"""Geometry utility functions for the navigation environment.

This module provides functions to calculate distances between points and rectangles,
and sample valid positions for the robot and goal in the environment.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp


def point_to_rectangle_distance(point: jnp.ndarray, rect: jnp.ndarray) -> jnp.ndarray:
    """Calculate minimum distance from a point to a rectangle.

    Args:
        point: Point coordinates [x, y]
        rect: Rectangle coordinates [x, y, width, height]

    Returns:
        Minimum distance from point to rectangle
    """
    # Extract rectangle coordinates and dimensions
    rect_x, rect_y, rect_w, rect_h = rect

    # Calculate rectangle corners
    rect_x_min, rect_y_min = rect_x, rect_y
    rect_x_max, rect_y_max = rect_x + rect_w, rect_y + rect_h

    # Find closest point on rectangle to the query point
    closest_x = jnp.clip(point[0], rect_x_min, rect_x_max)
    closest_y = jnp.clip(point[1], rect_y_min, rect_y_max)

    # If point is inside the rectangle, distance is 0
    inside = (closest_x == point[0]) & (closest_y == point[1])

    # Calculate distance to closest point
    dx = closest_x - point[0]
    dy = closest_y - point[1]
    distance = jnp.sqrt(dx**2 + dy**2)

    # Return 0 if inside, otherwise the calculated distance
    return jnp.where(inside, 0.0, distance)


def sample_valid_positions(
    key: chex.PRNGKey, obstacles: jnp.ndarray, arena_size: float, clearance: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample valid positions for robot and goal that are not colliding with obstacles.

    Args:
        key: Random key
        obstacles: Obstacle coordinates [x, y, w, h]
        arena_size: Size of the arena
        clearance: Minimum distance from obstacles

    Returns:
        Robot position, and goal position
    """
    # Sample positions once
    key, key_robot, key_goal = jax.random.split(key, 3)
    buffer = clearance

    # Sample 10 positions at once for both robot and goal
    robot_samples = jax.random.uniform(key_robot, shape=(10, 2), minval=buffer, maxval=arena_size - buffer)

    goal_samples = jax.random.uniform(key_goal, shape=(10, 2), minval=buffer, maxval=arena_size - buffer)

    # Function to check if a position is valid
    def is_valid_position(pos):
        # Calculate distance to each obstacle
        distances = jax.vmap(point_to_rectangle_distance, in_axes=(None, 0))(pos, obstacles)
        return jnp.all(distances >= clearance)

    # Vectorize the validation function over all robot positions
    check_batch = jax.vmap(is_valid_position)

    # Check all robot and goal positions in one go
    valid_robots = check_batch(robot_samples)  # Shape: (10,)
    valid_goals = check_batch(goal_samples)  # Shape: (10,)

    # Find first valid position index, default to 0 if none are valid
    robot_idx = jnp.where(valid_robots, size=1, fill_value=0)[0][0]
    goal_idx = jnp.where(valid_goals, size=1, fill_value=0)[0][0]

    # Get the positions
    robot_pos = robot_samples[robot_idx]
    goal_pos = goal_samples[goal_idx]

    return robot_pos, goal_pos


def generate_obstacles(
    key: chex.PRNGKey, arena_size: float, num_obstacles: int, min_size: float, max_size: float
) -> jnp.ndarray:
    """Generate random obstacles for the navigation environment.

    Args:
        key: Random key
        arena_size: Size of the arena (square)
        num_obstacles: Number of obstacles to generate
        min_size: Minimum obstacle size (width/height)
        max_size: Maximum obstacle size (width/height)

    Returns:
        Obstacles array with shape (num_obstacles, 4) where each row is [x, y, width, height]
    """
    # Split key for obstacle sizes and positions
    key, obs_size_key = jax.random.split(key)

    # Sample obstacle sizes
    obstacle_sizes = jax.random.uniform(obs_size_key, shape=(num_obstacles, 2), minval=min_size, maxval=max_size)

    # Calculate maximum obstacle dimension to ensure they stay within arena
    max_obs_dim = jnp.max(obstacle_sizes)

    # Sample obstacle positions
    obstacle_positions = jax.random.uniform(
        obs_size_key, shape=(num_obstacles, 2), minval=0.0, maxval=arena_size - max_obs_dim
    )

    # Stack as [x, y, width, height] and return
    obstacles = jnp.concatenate([obstacle_positions, obstacle_sizes], axis=1)

    return obstacles
