"""Differential-drive PointGoal navigation env wrapping MuJoCo.

Action: ``(v, omega)`` — forward linear velocity (m/s) and yaw rate (rad/s) in
the body frame. We turn this into world-frame velocity targets and drive the
planar joints via velocity actuators.

Observation:
  * ``depth`` : ``(H, W)`` float32, meters, clamped to ``depth_max_m``
  * ``goal_vec`` : ``(2,)`` float32 — ``(distance, bearing)`` in robot frame
  * ``pose`` : ``(3,)`` float32 — ``(x, y, yaw)`` in world
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import mujoco

from .pib_generator import RoomSpec, sample_room, build_xml


@dataclass
class StepResult:
    obs: dict
    reward: float
    done: bool
    info: dict


class NavEnv:
    def __init__(self, cfg: dict, room: Optional[RoomSpec] = None, seed: Optional[int] = None):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self.depth_wh = tuple(cfg["depth_wh"])
        self.depth_max = float(cfg["depth_max_m"])
        self.fov = float(cfg["depth_fov_deg"])
        self.control_dt = float(cfg["control_dt"])
        self.max_lin = float(cfg["max_lin_vel"])
        self.max_ang = float(cfg["max_ang_vel"])
        self.goal_tol = float(cfg["goal_tol_m"])
        self.max_collisions = int(cfg["max_collisions"])
        self.max_steps = int(cfg["max_episode_steps"])
        self._renderer = None
        self._depth_enabled = False
        self.reset(room=room)

    # ------------------------------------------------------------------
    def reset(self, room: Optional[RoomSpec] = None, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if room is None:
            room = sample_room(self._rng, self.cfg)
        self.room = room
        xml = build_xml(room, depth_wh=self.depth_wh, fov_deg=self.fov)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._jid_x = self.model.joint("slide_x").id
        self._jid_y = self.model.joint("slide_y").id
        self._jid_z = self.model.joint("hinge_z").id
        self._qpos_adr = [self.model.jnt_qposadr[j] for j in (self._jid_x, self._jid_y, self._jid_z)]
        self._qvel_adr = [self.model.jnt_dofadr[j] for j in (self._jid_x, self._jid_y, self._jid_z)]
        self._body_id = self.model.body("robot").id
        # robot body's initial world-frame xy (from the XML pos="..."); slide
        # joints measure displacement from this offset.
        self._start_xy = np.array(self.model.body_pos[self._body_id, :2], dtype=np.float64).copy()
        # set yaw
        self.data.qpos[self._qpos_adr[2]] = room.start_yaw
        mujoco.mj_forward(self.model, self.data)

        # re-create renderer for new model (size is fixed, so could reuse across models
        # in principle; but MuJoCo renderer is model-bound)
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        self._renderer = mujoco.Renderer(self.model, height=self.depth_wh[1], width=self.depth_wh[0])
        self._renderer.enable_depth_rendering()
        self._depth_enabled = True

        self.n_steps = 0
        self.n_collisions = 0
        self._goal_xy = np.asarray(room.goal, dtype=np.float32)
        return self._get_obs()

    # ------------------------------------------------------------------
    def _pose(self) -> Tuple[float, float, float]:
        # slide joints give displacement from body-home; add body-home to get world xy
        x = float(self.data.qpos[self._qpos_adr[0]]) + float(self._start_xy[0])
        y = float(self.data.qpos[self._qpos_adr[1]]) + float(self._start_xy[1])
        yaw = float(self.data.qpos[self._qpos_adr[2]])
        return x, y, yaw

    def _goal_in_robot_frame(self) -> np.ndarray:
        x, y, yaw = self._pose()
        dx, dy = self._goal_xy[0] - x, self._goal_xy[1] - y
        dist = float(np.hypot(dx, dy))
        bearing = float(np.arctan2(dy, dx) - yaw)
        # wrap
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        return np.asarray([dist, bearing], dtype=np.float32)

    def _render_depth(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera="cam_depth")
        depth = self._renderer.render()
        depth = np.asarray(depth, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=self.depth_max, posinf=self.depth_max, neginf=0.0)
        depth = np.clip(depth, 0.0, self.depth_max)
        return depth

    def _get_obs(self) -> dict:
        x, y, yaw = self._pose()
        return {
            "depth": self._render_depth(),
            "goal_vec": self._goal_in_robot_frame(),
            "pose": np.asarray([x, y, yaw], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    def step(self, action) -> StepResult:
        v = float(np.clip(action[0], -self.max_lin, self.max_lin))
        w = float(np.clip(action[1], -self.max_ang, self.max_ang))
        x, y, yaw = self._pose()
        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)
        self.data.ctrl[:] = [vx, vy, w]

        n_sub = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        pre_pos = np.array([x, y])
        collided_this_step = False
        for _ in range(n_sub):
            mujoco.mj_step(self.model, self.data)
            # detect contact with walls/obstacles (excluding floor)
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1 = self.model.geom(c.geom1).name
                g2 = self.model.geom(c.geom2).name
                if "floor" in (g1, g2):
                    continue
                if g1 == "base" or g2 == "base":
                    collided_this_step = True
                    break
            if collided_this_step:
                break

        if collided_this_step:
            self.n_collisions += 1
        self.n_steps += 1

        obs = self._get_obs()
        goal_dist = float(obs["goal_vec"][0])
        reached = goal_dist < self.goal_tol
        too_many_collisions = self.n_collisions >= self.max_collisions
        timeout = self.n_steps >= self.max_steps
        done = bool(reached or too_many_collisions or timeout)

        # reward is not used for training but convenient for logging
        moved = float(np.linalg.norm([self._pose()[0] - pre_pos[0], self._pose()[1] - pre_pos[1]]))
        reward = -0.01 - 0.5 * float(collided_this_step) + (5.0 if reached else 0.0)

        info = {
            "collided": collided_this_step,
            "n_collisions": self.n_collisions,
            "n_steps": self.n_steps,
            "moved": moved,
            "reached": reached,
            "timeout": timeout,
        }
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    # ------------------------------------------------------------------
    def teleport(self, x: float, y: float, yaw: float) -> None:
        """Warp the robot to world (x, y, yaw). Slide joints store displacement
        from the body-home pose, so subtract that offset."""
        self.data.qpos[self._qpos_adr[0]] = x - float(self._start_xy[0])
        self.data.qpos[self._qpos_adr[1]] = y - float(self._start_xy[1])
        self.data.qpos[self._qpos_adr[2]] = yaw
        self.data.qvel[self._qvel_adr[0]] = 0.0
        self.data.qvel[self._qvel_adr[1]] = 0.0
        self.data.qvel[self._qvel_adr[2]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
