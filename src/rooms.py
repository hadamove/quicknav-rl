"""Room generation utilities optimized for JAX JIT compilation.

This module provides functions to generate room layouts using approaches
that are fully compatible with JAX's JIT compiler.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp


def generate_room(
    key: Any, arena_size: float, grid_size: int, target_carved_percent: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a room layout
    # TODO: Explain the algorithm in detail

    Args:
        key: Random key
        arena_size: Size of the arena
        grid_size: Size of the grid (must be a concrete value for JIT)
        target_carved_percent: Target percentage of carved out space

    Returns:
        Tuple of (obstacles, free_positions):
            - obstacles: Array with obstacles as [x, y, width, height]
            - free_positions: Array with free space coordinates
    """
    # Compute tile size
    tile_size = arena_size / grid_size

    # Determine target count of cells to carve
    total_cells = grid_size * grid_size
    target_count = jnp.floor(total_cells * target_carved_percent).astype(jnp.int32)
    max_steps = jnp.floor(total_cells * target_carved_percent * 2).astype(jnp.int32)

    # Initialize grid (1 = wall, 0 = free space)
    grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32)

    # Create carving mask (inner cells only)
    mask = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    mask = mask.at[1:-1, 1:-1].set(1)

    # Center position
    center_x = grid_size // 2
    center_y = grid_size // 2

    # Carve initial center position
    grid = grid.at[center_y, center_x].set(0)

    # Movement directions
    dx = jnp.array([0, 1, 0, -1])
    dy = jnp.array([-1, 0, 1, 0])

    # Main loop state
    def body_fun(state_tuple):
        grid, key, carved_count, current_x, current_y, steps = state_tuple

        # Choose random direction
        key, subkey = jax.random.split(key)
        direction = jax.random.randint(subkey, (), 0, 4)

        # Calculate new position
        new_x = current_x + dx[direction]
        new_y = current_y + dy[direction]

        # Ensure new position is valid
        valid_move = mask[new_y, new_x] > 0

        # Update position
        current_x = jnp.where(valid_move, new_x, current_x)
        current_y = jnp.where(valid_move, new_y, current_y)

        # Carve the tile
        old_value = grid[current_y, current_x]
        grid = grid.at[current_y, current_x].set(0)

        # Update carved count
        carved_count = carved_count + (old_value > 0)

        # Random return to center
        key, subkey = jax.random.split(key)
        return_to_center = jax.random.uniform(subkey) < 0.05
        current_x = jnp.where(return_to_center, center_x, current_x)
        current_y = jnp.where(return_to_center, center_y, current_y)

        return grid, key, carved_count, current_x, current_y, steps + 1

    # Condition function
    def cond_fun(state_tuple):
        _, _, carved_count, _, _, step_count = state_tuple
        return (carved_count < target_count) & (step_count < max_steps)

    # Run the loop
    initial_state = (grid, key, 1, center_x, center_y, 0)
    final_grid, _, _, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_state)

    # Create coordinate arrays for all grid cells
    y_indices, x_indices = jnp.mgrid[0:grid_size, 0:grid_size]
    y_indices = y_indices.reshape(-1)
    x_indices = x_indices.reshape(-1)

    # Create obstacle mask
    grid_flat = final_grid.reshape(-1)
    obstacle_mask = (grid_flat == 1).astype(jnp.float32)
    free_mask = (grid_flat == 0).astype(jnp.float32)

    # Create obstacle rectangles
    all_rects = jnp.stack(
        [
            x_indices * tile_size,  # x
            y_indices * tile_size,  # y
            jnp.ones_like(x_indices) * tile_size,  # width
            jnp.ones_like(y_indices) * tile_size,  # height
        ],
        axis=1,
    )

    # Create free positions
    all_positions = jnp.stack(
        [x_indices * tile_size + tile_size / 2, y_indices * tile_size + tile_size / 2],  # x (center)  # y (center)
        axis=1,
    )

    # Use masking to get valid positions
    obstacle_rects = all_rects * obstacle_mask[:, jnp.newaxis]
    free_positions = all_positions * free_mask[:, jnp.newaxis]

    return obstacle_rects, free_positions


def sample_positions(key: Any, positions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample two positions from an array of positions.

    Args:
        key: Random key
        positions: Array of positions with shape (n, 2)

    Returns:
        Tuple of (pos1, pos2) with two sampled positions
    """
    # Split key
    key1, key2 = jax.random.split(key)

    # Create a mask of valid positions (non-zero)
    free_mask = jnp.any(positions != 0, axis=1)

    # Get number of valid positions using sum
    n = jnp.sum(free_mask)
    n_safe = jnp.maximum(n, 1)  # Ensure we have at least 1 position to sample from

    # Sample indices
    idx1 = jax.random.randint(key1, (), 0, n_safe)
    idx2 = jax.random.randint(key2, (), 0, n_safe)

    # Create a running index for valid positions
    running_idx = jnp.arange(positions.shape[0])

    # For each position, compute if it should be selected based on the running count
    # of valid positions up to that point matching our target indices
    def select_position(idx_target, running_idx, free_mask):
        # Count of valid positions up to current index (inclusive)
        valid_count = jnp.cumsum(free_mask)
        # This position is selected if:
        # 1. It's a valid position (free_mask is True)
        # 2. The count of valid positions up to here minus 1 equals our target index
        is_selected = free_mask & (valid_count - 1 == idx_target)
        # Return the position if selected, else zeros
        return jnp.sum(positions * is_selected[:, None], axis=0)

    # Select positions
    pos1 = select_position(idx1, running_idx, free_mask)
    pos2 = select_position(idx2, running_idx, free_mask)

    # Handle boundary case with fallback default
    empty = n == 0
    fallback_pos = jnp.array([1.0, 1.0])

    # Return positions with fallback
    pos1 = jnp.where(empty, fallback_pos, pos1)
    pos2 = jnp.where(empty, 2 * fallback_pos, pos2)

    return pos1, pos2
