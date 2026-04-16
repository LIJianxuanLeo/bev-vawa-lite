"""Habitat-sim PointGoal navigation environment (remote GPU track).

This is a **drop-in replacement** for ``NavEnv`` (the MuJoCo environment) used
for future research on realistic 3D scene datasets (HSSD, HM3D, ProcTHOR-HAB,
Gibson). It is **not** intended to run on the Apple M4 — habitat-sim has no
EGL headless path on macOS and requires a CUDA + EGL-capable Linux host.

Design principles:

* **Lazy import** — ``habitat_sim`` is only imported when an instance is
  constructed, so the rest of the project (including ``pytest tests/``) stays
  fully importable on macOS where habitat-sim is not installed.
* **Shared observation schema** — ``reset()`` / ``step()`` return exactly the
  same dict shape as ``NavEnv`` (keys ``depth``, ``goal_vec``, ``pose``), so
  downstream code (data rollout, training, evaluation) does not branch on the
  simulator backend.
* **Same action space** — ``(v, omega)`` continuous differential-drive, clipped
  to the same platform limits as the MuJoCo env.

Typical usage (on the remote box):

    from bev_vawa.envs.habitat_env import HabitatNavEnv
    env = HabitatNavEnv(cfg["env"], scene_glb="data/scene_datasets/hssd/106366104.glb")
    obs = env.reset(seed=0)
    for _ in range(300):
        obs, reward, done, info = env.step((0.2, 0.1))
        if done:
            break
    env.close()
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

# --------------------------------------------------------------------------------------
# Lazy-import guard. We *do not* raise at module import time so that static
# tools (pytest collection, type-checkers, module discovery) keep working on
# laptops that do not have habitat-sim. The error is deferred to construction.
# --------------------------------------------------------------------------------------
try:
    import habitat_sim  # type: ignore
    _HAS_HABITAT = True
    _IMPORT_ERR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - depends on host
    habitat_sim = None  # type: ignore
    _HAS_HABITAT = False
    _IMPORT_ERR = _e


@dataclass
class StepResult:
    obs: dict
    reward: float
    done: bool
    info: dict


def _require_habitat() -> None:
    if not _HAS_HABITAT:
        raise ImportError(
            "habitat-sim is not available on this host. This environment is "
            "intended for the remote GPU training track; please use the "
            "docker/habitat.Dockerfile image, or run the MuJoCo NavEnv locally. "
            f"Underlying import error: {_IMPORT_ERR!r}"
        )


class HabitatNavEnv:
    """PointGoal navigation on a habitat-sim scene with a depth sensor.

    Parameters
    ----------
    env_cfg
        Same ``cfg['env']`` block used by the MuJoCo env (see configs/default.yaml).
        Fields consumed: ``depth_wh``, ``depth_fov_deg``, ``depth_max_m``,
        ``control_dt``, ``max_lin_vel``, ``max_ang_vel``, ``goal_tol_m``,
        ``max_collisions``, ``max_episode_steps``.
    scene_glb
        Absolute path to the scene .glb / .ply / .json (habitat scene config).
    seed
        RNG seed (used for start/goal sampling).
    agent_height
        Camera height above the navmesh, metres. Default 1.0 to match a small
        indoor robot.
    gpu_device_id
        CUDA device id for the renderer. Use ``0`` unless you run multi-GPU.
    """

    def __init__(
        self,
        env_cfg: dict,
        scene_glb: str,
        seed: int = 0,
        agent_height: float = 1.0,
        gpu_device_id: int = 0,
    ):
        _require_habitat()
        self.cfg = env_cfg
        self.scene_glb = scene_glb
        self.agent_height = float(agent_height)
        self.gpu_device_id = int(gpu_device_id)

        self.depth_wh = tuple(env_cfg["depth_wh"])
        self.depth_max = float(env_cfg["depth_max_m"])
        self.fov = float(env_cfg["depth_fov_deg"])
        self.control_dt = float(env_cfg["control_dt"])
        self.max_lin = float(env_cfg["max_lin_vel"])
        self.max_ang = float(env_cfg["max_ang_vel"])
        self.goal_tol = float(env_cfg["goal_tol_m"])
        self.max_collisions = int(env_cfg["max_collisions"])
        self.max_steps = int(env_cfg["max_episode_steps"])

        self._rng = np.random.default_rng(seed)
        self._sim = None
        self._agent = None
        self._vel_ctrl = None
        self._build_sim()

        self._goal_xyz: Optional[np.ndarray] = None   # world-frame goal (3,)
        self._shortest_dist: float = float("nan")     # geodesic at reset
        self.n_steps = 0
        self.n_collisions = 0

    # -------------------------------------------------------------- simulator
    def _build_sim(self) -> None:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_glb
        sim_cfg.gpu_device_id = self.gpu_device_id
        sim_cfg.enable_physics = True

        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [self.depth_wh[1], self.depth_wh[0]]  # (H, W)
        depth_sensor.position = [0.0, self.agent_height, 0.0]
        depth_sensor.hfov = self.fov

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [depth_sensor]
        # we drive the agent with VelocityControl, so no discrete actions needed
        agent_cfg.action_space = {}

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)
        self._agent = self._sim.get_agent(0)

        self._vel_ctrl = habitat_sim.physics.VelocityControl()
        self._vel_ctrl.controlling_lin_vel = True
        self._vel_ctrl.controlling_ang_vel = True
        self._vel_ctrl.lin_vel_is_local = True
        self._vel_ctrl.ang_vel_is_local = True

    # ----------------------------------------------------------------- reset
    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not self._sim.pathfinder.is_loaded:
            raise RuntimeError(
                f"no navmesh loaded for scene {self.scene_glb}. Run "
                "`habitat_sim.nav.NavMeshSettings()` + `sim.recompute_navmesh()` "
                "or use scenes shipped with a .navmesh file."
            )

        # sample a solvable (start, goal) pair with enough geodesic distance
        start, goal, dist = None, None, 0.0
        for _ in range(50):
            s = self._sim.pathfinder.get_random_navigable_point()
            g = self._sim.pathfinder.get_random_navigable_point()
            path = habitat_sim.ShortestPath()
            path.requested_start = s
            path.requested_end = g
            if self._sim.pathfinder.find_path(path) and math.isfinite(path.geodesic_distance):
                if 2.0 < path.geodesic_distance < 15.0:  # avoid trivial / unreachable
                    start, goal, dist = s, g, float(path.geodesic_distance)
                    break
        if start is None:
            raise RuntimeError("failed to sample a navigable (start, goal) pair after 50 tries")

        # random yaw
        yaw = float(self._rng.uniform(-np.pi, np.pi))
        self.teleport_xyz(np.asarray(start), yaw)
        self._goal_xyz = np.asarray(goal, dtype=np.float32)
        self._shortest_dist = float(dist)
        self.n_steps = 0
        self.n_collisions = 0
        return self._get_obs()

    # ---------------------------------------------------------------- helpers
    def _pose(self) -> Tuple[float, float, float]:
        st = self._agent.state
        x = float(st.position[0])
        z = float(st.position[2])   # Habitat uses Y-up; we treat X–Z as the floor plane
        # extract yaw (rotation around Y). quat: (w, x, y, z) in habitat
        q = st.rotation
        # use atan2 of rotation applied to forward vector (-Z in habitat)
        fwd = np.array([0.0, 0.0, -1.0])
        rot = habitat_sim.utils.common.quat_rotate_vector(q, fwd)
        yaw = float(math.atan2(-rot[0], -rot[2]))  # match robot-frame x-forward convention
        return x, z, yaw

    def _goal_in_robot_frame(self) -> np.ndarray:
        x, z, yaw = self._pose()
        gx, _, gz = self._goal_xyz.tolist()
        dx, dz = gx - x, gz - z
        dist = float(math.hypot(dx, dz))
        bearing = float(math.atan2(dz, dx) - yaw)
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return np.asarray([dist, bearing], dtype=np.float32)

    def _render_depth(self) -> np.ndarray:
        obs = self._sim.get_sensor_observations()
        depth = np.asarray(obs["depth"], dtype=np.float32)
        depth = np.nan_to_num(depth, nan=self.depth_max, posinf=self.depth_max, neginf=0.0)
        depth = np.clip(depth, 0.0, self.depth_max)
        return depth

    def _get_obs(self) -> dict:
        x, z, yaw = self._pose()
        return {
            "depth": self._render_depth(),
            "goal_vec": self._goal_in_robot_frame(),
            "pose": np.asarray([x, z, yaw], dtype=np.float32),
        }

    # ----------------------------------------------------------------- step
    def step(self, action) -> StepResult:
        v = float(np.clip(action[0], -self.max_lin, self.max_lin))
        w = float(np.clip(action[1], -self.max_ang, self.max_ang))

        # body-frame velocity. Habitat uses -Z as forward.
        self._vel_ctrl.linear_velocity = np.array([0.0, 0.0, -v], dtype=np.float32)
        self._vel_ctrl.angular_velocity = np.array([0.0, w, 0.0], dtype=np.float32)

        state = self._agent.state
        new_rigid = self._vel_ctrl.integrate_transform(self.control_dt, state.rigid_state())
        proposed_pos = np.asarray(new_rigid.translation)
        # snap to navmesh: habitat returns a legal final pose given collisions
        end_pos = self._sim.pathfinder.try_step(state.position, proposed_pos)
        collided = bool(np.linalg.norm(end_pos - proposed_pos) > 1e-3)
        if collided:
            self.n_collisions += 1

        # write updated state back
        new_state = habitat_sim.AgentState()
        new_state.position = end_pos
        new_state.rotation = new_rigid.rotation
        self._agent.set_state(new_state, reset_sensors=False)
        self._sim.step_physics(self.control_dt)
        self.n_steps += 1

        obs = self._get_obs()
        goal_dist = float(obs["goal_vec"][0])
        reached = goal_dist < self.goal_tol
        too_many_collisions = self.n_collisions >= self.max_collisions
        timeout = self.n_steps >= self.max_steps
        done = bool(reached or too_many_collisions or timeout)

        reward = -0.01 - 0.5 * float(collided) + (5.0 if reached else 0.0)
        info = {
            "collided": collided,
            "n_collisions": self.n_collisions,
            "n_steps": self.n_steps,
            "reached": reached,
            "timeout": timeout,
            "geodesic_start": self._shortest_dist,
        }
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    # ----------------------------------------------------------- teleport
    def teleport_xyz(self, pos_xyz: np.ndarray, yaw: float) -> None:
        """Warp the agent to world (x, y, z) + yaw (rad around Y axis)."""
        st = habitat_sim.AgentState()
        st.position = np.asarray(pos_xyz, dtype=np.float32)
        half = 0.5 * yaw
        # quaternion around Y axis in habitat (w, x, y, z)
        st.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            yaw, np.array([0.0, 1.0, 0.0])
        )
        self._agent.set_state(st, reset_sensors=False)

    # ------------------------------------------------------ expert path API
    def shortest_path(self, start_xyz: np.ndarray, goal_xyz: np.ndarray):
        """Return a ``habitat_sim.ShortestPath`` with ``.points`` and
        ``.geodesic_distance`` populated, or ``None`` if unreachable.
        Used by the offline rollout to act as the A* equivalent."""
        p = habitat_sim.ShortestPath()
        p.requested_start = np.asarray(start_xyz, dtype=np.float32)
        p.requested_end = np.asarray(goal_xyz, dtype=np.float32)
        if not self._sim.pathfinder.find_path(p):
            return None
        if not math.isfinite(p.geodesic_distance):
            return None
        return p

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:
                pass
            self._sim = None
            self._agent = None

    # ---------------------------------------------------------- convenience
    @property
    def goal_xyz(self) -> Optional[np.ndarray]:
        return None if self._goal_xyz is None else self._goal_xyz.copy()

    @property
    def shortest_distance(self) -> float:
        return self._shortest_dist
