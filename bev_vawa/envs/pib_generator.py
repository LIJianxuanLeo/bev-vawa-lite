"""Procedural Indoor Benchmark (PIB) room generator.

Produces a MuJoCo XML with:
  * a flat floor
  * 4 axis-aligned boundary walls
  * N random rectangular obstacles
  * a differential-drive robot base (planar free body: slide-X, slide-Y, hinge-Z)
  * a forward-facing depth camera mounted on the base

Coordinate convention: world frame, z up, floor at z=0. The robot base has
``robot_radius``; obstacles and walls extend from the floor up to ``wall_height``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class Obstacle:
    cx: float
    cy: float
    sx: float  # half-extent x
    sy: float  # half-extent y


@dataclass
class RoomSpec:
    width: float          # full x-size in meters
    depth: float          # full y-size in meters
    wall_height: float
    robot_radius: float
    obstacles: List[Obstacle] = field(default_factory=list)
    start: Tuple[float, float] = (0.0, 0.0)
    goal: Tuple[float, float] = (1.0, 1.0)
    start_yaw: float = 0.0


def _aabb_overlap(a_cx, a_cy, a_sx, a_sy, b_cx, b_cy, b_sx, b_sy, margin=0.0) -> bool:
    return (abs(a_cx - b_cx) <= a_sx + b_sx + margin) and (abs(a_cy - b_cy) <= a_sy + b_sy + margin)


def _point_inside_obstacle(x, y, obs: Obstacle, margin: float) -> bool:
    return (abs(x - obs.cx) <= obs.sx + margin) and (abs(y - obs.cy) <= obs.sy + margin)


def sample_room(rng: np.random.Generator, cfg: dict) -> RoomSpec:
    """Sample a random RoomSpec from an env config block (see configs/default.yaml)."""
    w = float(rng.uniform(*cfg["room_size_m"]))
    d = float(rng.uniform(*cfg["room_size_m"]))
    n_obs = int(rng.integers(cfg["n_obstacles"][0], cfg["n_obstacles"][1] + 1))
    obs: List[Obstacle] = []
    tries = 0
    while len(obs) < n_obs and tries < 200:
        tries += 1
        sx = float(rng.uniform(*cfg["obstacle_size_m"])) / 2.0
        sy = float(rng.uniform(*cfg["obstacle_size_m"])) / 2.0
        cx = float(rng.uniform(-w / 2 + sx + 0.3, w / 2 - sx - 0.3))
        cy = float(rng.uniform(-d / 2 + sy + 0.3, d / 2 - sy - 0.3))
        cand = Obstacle(cx, cy, sx, sy)
        if any(_aabb_overlap(cand.cx, cand.cy, cand.sx, cand.sy, o.cx, o.cy, o.sx, o.sy, margin=0.3) for o in obs):
            continue
        obs.append(cand)

    robot_r = cfg["robot_radius_m"]

    def _sample_free_point() -> Tuple[float, float]:
        for _ in range(500):
            x = float(rng.uniform(-w / 2 + robot_r + 0.2, w / 2 - robot_r - 0.2))
            y = float(rng.uniform(-d / 2 + robot_r + 0.2, d / 2 - robot_r - 0.2))
            if not any(_point_inside_obstacle(x, y, o, margin=robot_r + 0.05) for o in obs):
                return x, y
        return 0.0, 0.0

    start = _sample_free_point()
    for _ in range(50):
        goal = _sample_free_point()
        if (goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2 > (0.5 * min(w, d)) ** 2:
            break
    yaw = float(rng.uniform(-np.pi, np.pi))
    return RoomSpec(
        width=w, depth=d, wall_height=cfg["wall_height_m"], robot_radius=robot_r,
        obstacles=obs, start=start, goal=goal, start_yaw=yaw,
    )


def build_xml(room: RoomSpec, depth_wh=(128, 128), fov_deg: float = 90.0) -> str:
    """Return a MuJoCo XML string describing ``room`` + the diff-drive robot."""
    W, D, H = room.width, room.depth, room.wall_height
    thickness = 0.05

    walls = []
    # +X wall
    walls.append(f'<geom name="wall_px" type="box" pos="{W/2+thickness/2} 0 {H/2}" size="{thickness/2} {D/2} {H/2}" rgba="0.75 0.75 0.78 1"/>')
    walls.append(f'<geom name="wall_nx" type="box" pos="{-W/2-thickness/2} 0 {H/2}" size="{thickness/2} {D/2} {H/2}" rgba="0.75 0.75 0.78 1"/>')
    walls.append(f'<geom name="wall_py" type="box" pos="0 {D/2+thickness/2} {H/2}" size="{W/2} {thickness/2} {H/2}" rgba="0.75 0.75 0.78 1"/>')
    walls.append(f'<geom name="wall_ny" type="box" pos="0 {-D/2-thickness/2} {H/2}" size="{W/2} {thickness/2} {H/2}" rgba="0.75 0.75 0.78 1"/>')

    obs_geoms = []
    for i, o in enumerate(room.obstacles):
        obs_geoms.append(
            f'<geom name="obs_{i}" type="box" pos="{o.cx} {o.cy} {H/2}" '
            f'size="{o.sx} {o.sy} {H/2}" rgba="0.55 0.35 0.25 1"/>'
        )

    goal_marker = (
        f'<site name="goal" pos="{room.goal[0]} {room.goal[1]} 0.02" '
        f'size="0.12" rgba="0 1 0 0.5"/>'
    )

    r = room.robot_radius
    robot_body = f"""
    <body name="robot" pos="{room.start[0]} {room.start[1]} {r}">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="2.0"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="2.0"/>
      <joint name="hinge_z" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom name="base" type="cylinder" size="{r} {r*0.6}" rgba="0.1 0.5 0.9 1" mass="2.0"/>
      <geom name="heading" type="box" pos="{r*0.7} 0 0" size="{r*0.25} {r*0.1} {r*0.1}" rgba="1 1 0 1" mass="0.01"/>
      <site name="robot_site" pos="0 0 0"/>
      <camera name="cam_depth" pos="{r*0.4} 0 {r*0.4}" xyaxes="0 -1 0 0 0 1" fovy="{fov_deg}"/>
    </body>
    """

    # Velocity actuators so the controller can command v_forward and omega directly.
    # We emulate forward velocity by driving slide_x / slide_y jointly via the python-side
    # controller (applying a ctrl signal in the body frame is easier than mujoco-side kinematics).
    actuators = """
    <actuator>
      <velocity name="vx" joint="slide_x" kv="30"/>
      <velocity name="vy" joint="slide_y" kv="30"/>
      <velocity name="wz" joint="hinge_z" kv="10"/>
    </actuator>
    """

    xml = f"""<mujoco model="pib_nav">
      <option timestep="0.01" integrator="implicitfast" gravity="0 0 -9.81" impratio="2"/>
      <visual><global offwidth="{depth_wh[0]}" offheight="{depth_wh[1]}"/></visual>
      <default>
        <geom solref="0.005 1" solimp="0.9 0.95 0.001"/>
      </default>
      <asset>
        <material name="floor_mat" rgba="0.9 0.9 0.85 1"/>
      </asset>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="{max(W, D)} {max(W, D)} 0.1" material="floor_mat"/>
        {"".join(walls)}
        {"".join(obs_geoms)}
        {goal_marker}
        {robot_body}
      </worldbody>
      {actuators}
    </mujoco>"""
    return xml
