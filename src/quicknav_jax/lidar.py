"""
Lidar simulation module for robot environment.
Provides ray-casting functionality and collision detection.
"""

from enum import IntEnum
from typing import Protocol, Tuple

import jax
import jax.numpy as jnp


class LidarParams(Protocol):
    """Protocol for parameters needed by the lidar simulator."""

    lidar_num_beams: int
    lidar_fov: float
    lidar_max_distance: float
    goal_tolerance: float
    robot_radius: float


# Enum for collision types
class Collision(IntEnum):
    """Enumeration of possible collision types for lidar beams."""

    MaxDist = 0  # No collision, max distance reached
    Obstacle = 1  # Collision with obstacle
    Goal = 2  # Collision with goal


def simulate_lidar(
    x: jnp.ndarray,
    y: jnp.ndarray,
    theta: jnp.ndarray,
    obstacles: jnp.ndarray,
    goal_x: jnp.ndarray,
    goal_y: jnp.ndarray,
    params: LidarParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    angles = theta + jnp.linspace(
        -params.lidar_fov * jnp.pi / 360.0, params.lidar_fov * jnp.pi / 360.0, params.lidar_num_beams
    )

    # Apply ray_cast to all angles
    distances, collision_types = jax.vmap(ray_cast, in_axes=(0, None, None, None, None, None, None))(
        angles, x, y, obstacles, goal_x, goal_y, params
    )

    return distances, collision_types


def ray_cast(
    angle: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    obstacles: jnp.ndarray,
    goal_x: jnp.ndarray,
    goal_y: jnp.ndarray,
    params: LidarParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cast a single ray from the perimeter of the robot and find the closest intersection.

    Args:
        angle: Ray angle in radians
        x: Robot center x position
        y: Robot center y position
        obstacles: Array of obstacle definitions
        goal_x: Goal x position
        goal_y: Goal y position
        params: Environment parameters

    Returns:
        Tuple of (distance, collision_type)
    """
    # Direction vector
    dx, dy = jnp.cos(angle), jnp.sin(angle)

    # Calculate start position on robot perimeter
    start_x = x + params.robot_radius * dx
    start_y = y + params.robot_radius * dy

    # Obstacle intersections (simplified box collision)
    obs_t = calculate_obstacle_intersections(start_x, start_y, dx, dy, obstacles)

    # Goal intersection
    goal_t = calculate_goal_intersection(start_x, start_y, dx, dy, goal_x, goal_y, params.goal_tolerance)

    # Find minimum distance and collision type
    distances = jnp.array([obs_t, goal_t, params.lidar_max_distance])
    idx = jnp.argmin(distances)
    min_dist = distances[idx]

    # Map index to collision type
    collision_types = jnp.array([Collision.Obstacle, Collision.Goal, Collision.MaxDist])
    collision_type = collision_types[idx]

    return min_dist, collision_type


def calculate_obstacle_intersections(
    x: jnp.ndarray, y: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray, obstacles: jnp.ndarray
) -> jnp.ndarray:
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

    def check_obstacle(obstacle: jnp.ndarray, best_t: jnp.ndarray) -> jnp.ndarray:
        """
        Check intersection with a single obstacle.
        Updates best_t if a closer intersection is found.
        """
        ox, oy, ow, oh = obstacle

        # Simplified AABB intersection test
        t_min_x, t_max_x = calculate_slab_intersection(x, dx, ox, ox + ow)
        t_min_y, t_max_y = calculate_slab_intersection(y, dy, oy, oy + oh)

        t_enter = jnp.maximum(t_min_x, t_min_y)
        t_exit = jnp.minimum(t_max_x, t_max_y)

        hit = (t_enter <= t_exit) & (t_enter > 0)
        return jnp.where(hit & (t_enter < best_t), t_enter, best_t)

    # Check all obstacles
    return jax.lax.fori_loop(0, obstacles.shape[0], lambda i, t: check_obstacle(obstacles[i], t), jnp.inf)


def calculate_slab_intersection(
    origin: jnp.ndarray, direction: jnp.ndarray, slab_min: jnp.ndarray, slab_max: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    safe_dir = jnp.where(jnp.abs(direction) < eps, jnp.sign(direction) * eps, direction)

    # Calculate the intersection parameters
    t1 = (slab_min - origin) / safe_dir
    t2 = (slab_max - origin) / safe_dir

    # When direction is negative, swap t1 and t2
    is_neg = direction < 0
    t_min = jnp.where(is_neg, t2, t1)
    t_max = jnp.where(is_neg, t1, t2)

    # Special case for near-zero direction (parallel to slab)
    is_zero = jnp.abs(direction) < eps
    inside_slab = (origin >= slab_min) & (origin <= slab_max)
    t_min = jnp.where(is_zero & inside_slab, -jnp.inf, t_min)
    t_max = jnp.where(is_zero & inside_slab, jnp.inf, t_max)
    t_min = jnp.where(is_zero & ~inside_slab, jnp.inf, t_min)
    t_max = jnp.where(is_zero & ~inside_slab, -jnp.inf, t_max)

    return t_min, t_max


def calculate_goal_intersection(
    x: jnp.ndarray,
    y: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    goal_x: jnp.ndarray,
    goal_y: jnp.ndarray,
    goal_tolerance: float,
) -> jnp.ndarray:
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
    goal_hit = (perp_dist_sq <= goal_tolerance**2) & (goal_proj > 0)

    # Distance along ray to closest point to goal
    return jnp.where(goal_hit, goal_proj - jnp.sqrt(goal_tolerance**2 - perp_dist_sq), jnp.inf)
