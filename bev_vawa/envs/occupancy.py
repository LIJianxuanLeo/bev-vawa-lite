"""Occupancy grid rasterization + A* path planning on PIB rooms.

The grid lives in the room's world frame. Origin is the room center; axes are
world X (col) and world Y (row). Cells marked 1 are occupied (walls / obstacles
inflated by the robot radius), cells marked 0 are free.
"""
from __future__ import annotations
import heapq
from typing import List, Optional, Tuple
import numpy as np
from .pib_generator import RoomSpec


def rasterize(room: RoomSpec, cell_m: float = 0.05, inflate_m: Optional[float] = None) -> np.ndarray:
    """Return a uint8 HxW occupancy grid. Inflation defaults to the robot radius."""
    inflate = room.robot_radius if inflate_m is None else inflate_m
    W, D = room.width, room.depth
    w_cells = int(np.ceil(W / cell_m))
    h_cells = int(np.ceil(D / cell_m))
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)

    # boundary border (the walls are just outside, but inflation pulls them in)
    inf_cells = int(np.ceil(inflate / cell_m))
    if inf_cells > 0:
        grid[:inf_cells, :] = 1
        grid[-inf_cells:, :] = 1
        grid[:, :inf_cells] = 1
        grid[:, -inf_cells:] = 1

    for o in room.obstacles:
        cx_cell = int((o.cx + W / 2) / cell_m)
        cy_cell = int((o.cy + D / 2) / cell_m)
        rx = int(np.ceil((o.sx + inflate) / cell_m))
        ry = int(np.ceil((o.sy + inflate) / cell_m))
        y0, y1 = max(0, cy_cell - ry), min(h_cells, cy_cell + ry + 1)
        x0, x1 = max(0, cx_cell - rx), min(w_cells, cx_cell + rx + 1)
        grid[y0:y1, x0:x1] = 1
    return grid


def world_to_cell(x: float, y: float, room: RoomSpec, cell_m: float) -> Tuple[int, int]:
    col = int((x + room.width / 2) / cell_m)
    row = int((y + room.depth / 2) / cell_m)
    return row, col


def cell_to_world(row: int, col: int, room: RoomSpec, cell_m: float) -> Tuple[float, float]:
    x = (col + 0.5) * cell_m - room.width / 2
    y = (row + 0.5) * cell_m - room.depth / 2
    return x, y


_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.41421356), (-1, 1, 1.41421356),
              (1, -1, 1.41421356), (1, 1, 1.41421356)]


def astar_path(grid: np.ndarray, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """8-connected A* on an occupancy grid. Returns list of (row, col) or None."""
    H, W = grid.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return None
    if grid[sr, sc] or grid[gr, gc]:
        return None
    if start_rc == goal_rc:
        return [start_rc]

    def h(r, c):
        return np.hypot(r - gr, c - gc)

    open_heap: list = []
    heapq.heappush(open_heap, (h(sr, sc), 0.0, sr, sc))
    came: dict = {}
    g_cost = {(sr, sc): 0.0}

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came:
                r, c = came[(r, c)]
                path.append((r, c))
            path.reverse()
            return path
        if g > g_cost.get((r, c), np.inf):
            continue
        for dr, dc, step in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc]:
                continue
            # disallow corner-cutting
            if dr != 0 and dc != 0:
                if grid[r + dr, c] or grid[r, c + dc]:
                    continue
            ng = g + step
            if ng < g_cost.get((nr, nc), np.inf):
                g_cost[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))
    return None


def path_feasible(room: RoomSpec, cell_m: float = 0.05) -> bool:
    grid = rasterize(room, cell_m=cell_m)
    s = world_to_cell(*room.start, room, cell_m)
    g = world_to_cell(*room.goal, room, cell_m)
    return astar_path(grid, s, g) is not None


def path_world(room: RoomSpec, cell_m: float = 0.05) -> Optional[np.ndarray]:
    """Return the A* path in world coordinates as (N, 2) float array, or None."""
    grid = rasterize(room, cell_m=cell_m)
    s = world_to_cell(*room.start, room, cell_m)
    g = world_to_cell(*room.goal, room, cell_m)
    cells = astar_path(grid, s, g)
    if cells is None:
        return None
    pts = np.array([cell_to_world(r, c, room, cell_m) for (r, c) in cells], dtype=np.float32)
    return pts
