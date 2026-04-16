"""Simple heading + pure-pursuit controller.

Given a target waypoint expressed in the **robot frame** (x-forward, y-left),
return ``(v, omega)`` control commands for a differential drive.
"""
from __future__ import annotations
import math


def pure_pursuit_cmd(wp_robot_frame, max_lin: float, max_ang: float,
                     lookahead_min: float = 0.15) -> tuple[float, float]:
    wx, wy = float(wp_robot_frame[0]), float(wp_robot_frame[1])
    dist = math.hypot(wx, wy)
    if dist < 1e-3:
        return 0.0, 0.0
    heading_err = math.atan2(wy, wx)            # angle to waypoint in robot frame
    # proportional heading controller with distance-tapered velocity
    omega = 3.0 * heading_err
    omega = max(-max_ang, min(max_ang, omega))
    # reduce forward speed when we are mis-aimed (cos^3 is sharper than cos)
    align = max(0.0, math.cos(heading_err))
    align = align ** 2
    v = max_lin * align * min(1.0, dist / max(lookahead_min, 1e-3))
    v = max(0.0, min(max_lin, v))
    return v, omega
