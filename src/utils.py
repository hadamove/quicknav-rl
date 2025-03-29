"""Utilities for sampling valid positions in navigation environments."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp


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

    # Calculate obstacle centers and radii
    obstacle_centers = obstacles[:, :2] + obstacles[:, 2:] / 2
    obstacle_radii = jnp.minimum(obstacles[:, 2], obstacles[:, 3]) / 2

    # Function to check if a position is valid
    def is_valid_position(pos):
        # Use broadcasting to calculate distances to all obstacles at once
        deltas = pos - obstacle_centers  # Shape: (num_obstacles, 2)
        distances = jnp.sqrt(jnp.sum(deltas**2, axis=1))  # Shape: (num_obstacles,)
        return jnp.all(distances >= (obstacle_radii + clearance))

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
