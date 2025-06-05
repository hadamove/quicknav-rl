"""
Lidar simulation module for robot environment.
Provides ray-casting functionality and collision detection.
"""

from typing import Protocol, Tuple

import numpy as np

from quicknav_utils.collision import Collision


class LidarParams(Protocol):
    """Protocol for parameters needed by the lidar simulator."""

    lidar_num_beams: int
    lidar_fov: float
    lidar_max_distance: float
    goal_tolerance: float


def simulate_lidar(
    x: float,
    y: float,
    theta: float,
    obstacles: np.ndarray,
    goal_x: float,
    goal_y: float,
    params: LidarParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a lidar sensor using ray casting.

    Args:
        x: Robot x position
        y: Robot y position
        theta: Robot orientation (radians)
        obstacles: Array of obstacle definitions, shape (n, 4) with [x, y, width, height]
        goal_x: Goal x position
        goal_y: Goal y position
        params: Environment parameters with lidar configuration

    Returns:
        Tuple of (distances, collision_types):
            - distances: Array of distances for each beam
            - collision_types: Array of collision types for each beam
    """
    # Calculate beam directions
    angles = theta + np.linspace(
        -params.lidar_fov * np.pi / 360.0, params.lidar_fov * np.pi / 360.0, params.lidar_num_beams
    )

    # Apply ray_cast to all angles
    distances = np.zeros(params.lidar_num_beams)
    collision_types = np.zeros(params.lidar_num_beams, dtype=np.int32)

    for i, angle in enumerate(angles):
        distances[i], collision_types[i] = ray_cast(angle, x, y, obstacles, goal_x, goal_y, params)

    return distances, collision_types


def ray_cast(
    angle: float,
    x: float,
    y: float,
    obstacles: np.ndarray,
    goal_x: float,
    goal_y: float,
    params: LidarParams,
) -> Tuple[float, int]:
    """
    Cast a single ray and find the closest intersection.

    Args:
        angle: Ray angle in radians
        x: Robot x position
        y: Robot y position
        obstacles: Array of obstacle definitions
        goal_x: Goal x position
        goal_y: Goal y position
        params: Environment parameters

    Returns:
        Tuple of (distance, collision_type)
    """
    # Direction vector
    dx, dy = np.cos(angle), np.sin(angle)

    # Obstacle intersections (simplified box collision)
    obs_t = calculate_obstacle_intersections(x, y, dx, dy, obstacles)

    # Goal intersection
    goal_t = calculate_goal_intersection(x, y, dx, dy, goal_x, goal_y, params.goal_tolerance)

    # Find minimum distance and collision type
    distances = np.array([obs_t, goal_t, params.lidar_max_distance])
    idx = np.argmin(distances)
    min_dist = distances[idx]

    # Map index to collision type
    collision_types = np.array([Collision.Obstacle, Collision.Goal, Collision.MaxDist])
    collision_type = collision_types[idx]

    return min_dist, collision_type


def calculate_obstacle_intersections(x: float, y: float, dx: float, dy: float, obstacles: np.ndarray) -> float:
    """
    Calculate intersection with obstacles.

    Args:
        x: Origin x position
        y: Origin y position
        dx: Direction vector x component
        dy: Direction vector y component
        obstacles: Array of obstacles, each [x, y, width, height]

    Returns:
        Minimum positive distance to an obstacle
    """
    best_t = np.inf

    for obstacle in obstacles:
        ox, oy, ow, oh = obstacle

        # Simplified AABB intersection test
        t_min_x, t_max_x = calculate_slab_intersection(x, dx, ox, ox + ow)
        t_min_y, t_max_y = calculate_slab_intersection(y, dy, oy, oy + oh)

        t_enter = max(t_min_x, t_min_y)
        t_exit = min(t_max_x, t_max_y)

        hit = (t_enter <= t_exit) and (t_enter > 0)
        if hit and (t_enter < best_t):
            best_t = t_enter

    return best_t


def calculate_slab_intersection(
    origin: float, direction: float, slab_min: float, slab_max: float
) -> Tuple[float, float]:
    """
    Calculate intersection with an axis-aligned slab.

    Args:
        origin: Origin coordinate (x or y)
        direction: Direction component (dx or dy)
        slab_min: Minimum slab coordinate
        slab_max: Maximum slab coordinate

    Returns:
        Tuple of (t_min, t_max) intersection parameters
    """
    # Use a small epsilon to avoid division by zero
    eps = 1e-10
    safe_dir = direction if abs(direction) >= eps else np.sign(direction) * eps

    # Calculate the intersection parameters
    t1 = (slab_min - origin) / safe_dir
    t2 = (slab_max - origin) / safe_dir

    # When direction is negative, swap t1 and t2
    is_neg = direction < 0
    t_min = t2 if is_neg else t1
    t_max = t1 if is_neg else t2

    # Special case for near-zero direction (parallel to slab)
    is_zero = abs(direction) < eps
    inside_slab = (origin >= slab_min) and (origin <= slab_max)

    if is_zero:
        if inside_slab:
            t_min = -np.inf
            t_max = np.inf
        else:
            t_min = np.inf
            t_max = -np.inf

    return t_min, t_max


def calculate_goal_intersection(
    x: float,
    y: float,
    dx: float,
    dy: float,
    goal_x: float,
    goal_y: float,
    goal_tolerance: float,
) -> float:
    """
    Calculate intersection with the goal circle.

    Args:
        x: Origin x position
        y: Origin y position
        dx: Direction vector x component
        dy: Direction vector y component
        goal_x: Goal x position
        goal_y: Goal y position
        goal_tolerance: Goal radius

    Returns:
        Minimum positive distance to the goal
    """
    # Vector from ray origin to goal center
    goal_dx, goal_dy = goal_x - x, goal_y - y

    # Project goal vector onto ray direction
    goal_proj = goal_dx * dx + goal_dy * dy

    # Distance squared from ray to goal center
    perp_dist_sq = goal_dx**2 + goal_dy**2 - goal_proj**2

    # Check if ray passes within goal radius
    goal_hit = (perp_dist_sq <= goal_tolerance**2) and (goal_proj > 0)

    # Distance along ray to closest point to goal
    if goal_hit:
        return goal_proj - np.sqrt(goal_tolerance**2 - perp_dist_sq)
    else:
        return np.inf
