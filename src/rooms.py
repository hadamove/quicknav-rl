"""Room generation utilities optimized for JAX JIT compilation.

This module provides functions to generate room layouts using approaches
that are fully compatible with JAX's JIT compiler.
"""

from enum import IntEnum
from typing import Any, Tuple, cast

import jax
import jax.numpy as jnp
from flax import struct


class TileType(IntEnum):
    """Enumeration of tile types used in the grid representation."""

    FREE = 0  # Empty space
    WALL = 1  # Wall/obstacle


@struct.dataclass
class RoomGenerationState:
    """State for the room generation random walk."""

    grid: jnp.ndarray  # Current grid state (1=wall, 0=free)
    key: Any  # Random key
    carved_count: jnp.ndarray  # Number of cells carved so far (as JAX array)
    current_x: jnp.ndarray  # Current x position (as JAX array)
    current_y: jnp.ndarray  # Current y position (as JAX array)
    steps: jnp.ndarray  # Number of steps taken (as JAX array)


def generate_room(
    key: Any, arena_size: float, grid_size: int, target_carved_percent: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a room layout using a random walk carving algorithm.

    The algorithm works as follows:
    1. Start with a grid where all cells are walls (1s)
    2. Begin at the center of the grid
    3. Repeatedly:
        - Choose a random direction (up, right, down, left)
        - If the move is valid (within inner grid), move there
        - Carve out the cell (set to 0)
        - Occasionally (5% chance) return to center to avoid getting stuck
        - Continue until target percentage of cells are carved or max steps reached
    4. Convert the grid to physical coordinates:
        - Walls become obstacle rectangles with position and size
        - Free spaces become center points for valid positions

    This approach generates connected rooms with organic-looking pathways,
    ensuring the robot can reach any free space from any other free space.
    The outer walls are always preserved as they're outside the carving mask.

    Args:
        key: JAX random key for reproducible generation
        arena_size: Physical size of the arena in meters
        grid_size: Number of grid cells in each dimension (must be concrete for JIT)
        target_carved_percent: Target fraction of inner cells to carve out (0 to 1)

    Returns:
        Tuple of (obstacles, free_positions):
            - obstacles: Array of [x, y, width, height] for each wall tile
            - free_positions: Array of [x, y] coordinates for free spaces
    """
    # Initialize grid parameters
    tile_size = arena_size / grid_size
    total_cells = grid_size * grid_size
    target_carved_cells = jnp.floor(total_cells * target_carved_percent).astype(jnp.int32)
    max_carving_steps = jnp.floor(total_cells * target_carved_percent * 2).astype(jnp.int32)

    # Initialize grid with all walls
    grid = jnp.ones((grid_size, grid_size), dtype=jnp.int32) * TileType.WALL

    # Create mask for valid carving area (inner cells only)
    carving_mask = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    carving_mask = carving_mask.at[1:-1, 1:-1].set(1)

    # Start at center
    center_x = grid_size // 2
    center_y = grid_size // 2
    grid = grid.at[center_y, center_x].set(TileType.FREE)  # Carve initial position

    # Define movement directions (up, right, down, left)
    directions = jnp.array(
        [
            [0, -1],  # up
            [1, 0],  # right
            [0, 1],  # down
            [-1, 0],  # left
        ]
    )

    def carve_step(state: RoomGenerationState) -> RoomGenerationState:
        """Perform one step of the random walk carving algorithm."""
        # Choose random direction
        key, move_key = jax.random.split(state.key)
        direction_idx = jax.random.randint(move_key, (), 0, 4)
        dx, dy = directions[direction_idx]

        # Calculate new position
        new_x = state.current_x + dx
        new_y = state.current_y + dy

        # Check if move is valid (within carving mask)
        is_valid_move = carving_mask[new_y, new_x] > 0

        # Update position if valid
        current_x = cast(jnp.ndarray, jnp.where(is_valid_move, new_x, state.current_x))
        current_y = cast(jnp.ndarray, jnp.where(is_valid_move, new_y, state.current_y))

        # Carve current position
        old_value = state.grid[current_y, current_x]
        new_grid = state.grid.at[current_y, current_x].set(TileType.FREE)
        new_carved_count = state.carved_count + (old_value == TileType.WALL)

        # Random return to center (5% chance)
        key, reset_key = jax.random.split(key)
        should_reset = jax.random.uniform(reset_key) < 0.05
        current_x = cast(jnp.ndarray, jnp.where(should_reset, center_x, current_x))
        current_y = cast(jnp.ndarray, jnp.where(should_reset, center_y, current_y))

        return RoomGenerationState(
            grid=new_grid,
            key=key,
            carved_count=new_carved_count,
            current_x=current_x,
            current_y=current_y,
            steps=state.steps + 1,
        )

    def should_continue_carving(state: RoomGenerationState) -> jnp.ndarray:
        """Check if we should continue carving."""
        return (state.carved_count < target_carved_cells) & (state.steps < max_carving_steps)

    # Run the carving loop
    initial_state = RoomGenerationState(
        grid=grid,
        key=key,
        carved_count=jnp.array(1),
        current_x=jnp.array(center_x),
        current_y=jnp.array(center_y),
        steps=jnp.array(0),
    )
    final_state = jax.lax.while_loop(should_continue_carving, carve_step, initial_state)
    final_grid = final_state.grid

    # Convert grid to physical coordinates
    y_coords, x_coords = jnp.mgrid[0:grid_size, 0:grid_size]
    y_coords = y_coords.reshape(-1)
    x_coords = x_coords.reshape(-1)

    # Create masks for obstacles and free spaces
    grid_flat = final_grid.reshape(-1)
    is_wall = (grid_flat == TileType.WALL).astype(jnp.float32)
    is_free = (grid_flat == TileType.FREE).astype(jnp.float32)

    # Create obstacle rectangles (for all wall cells)
    obstacles = (
        jnp.stack(
            [
                x_coords * tile_size,  # x position
                y_coords * tile_size,  # y position
                jnp.ones_like(x_coords) * tile_size,  # width
                jnp.ones_like(y_coords) * tile_size,  # height
            ],
            axis=1,
        )
        * is_wall[:, jnp.newaxis]
    )

    # Create free position coordinates (centers of free cells)
    free_positions = (
        jnp.stack(
            [
                x_coords * tile_size + tile_size / 2,  # x center
                y_coords * tile_size + tile_size / 2,  # y center
            ],
            axis=1,
        )
        * is_free[:, jnp.newaxis]
    )

    return obstacles, free_positions


def sample_positions(key: Any, positions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample two distinct positions from an array of valid positions.

    Args:
        key: JAX random key for reproducible sampling
        positions: Array of positions with shape (n, 2), where zeros indicate invalid positions

    Returns:
        Tuple of (pos1, pos2), each a [x, y] coordinate sampled from valid positions.
        If no valid positions exist, returns fallback positions.
    """
    # Split random key for two independent samples
    key_start, key_goal = jax.random.split(key)

    # Identify valid positions (non-zero coordinates)
    is_valid = jnp.any(positions != 0, axis=1)
    valid_count = jnp.sum(is_valid)
    safe_count = jnp.maximum(valid_count, 1)  # Avoid empty array issues

    # Sample random indices within valid range
    idx_start = jax.random.randint(key_start, (), 0, safe_count)
    idx_goal = jax.random.randint(key_goal, (), 0, safe_count)

    def select_position(target_idx: jnp.ndarray) -> jnp.ndarray:
        """Select a position corresponding to the nth valid position."""
        # Count valid positions up to each index
        valid_position_count = jnp.cumsum(is_valid)
        # Create selection mask for the target index
        is_selected = is_valid & (valid_position_count - 1 == target_idx)
        # Use multiplication and sum to select the position
        return jnp.sum(positions * is_selected[:, jnp.newaxis], axis=0)

    # Sample two positions
    pos_start = select_position(idx_start)
    pos_goal = select_position(idx_goal)

    # Handle empty array case
    is_empty = valid_count == 0
    fallback_pos = jnp.array([1.0, 1.0])
    pos_start = cast(jnp.ndarray, jnp.where(is_empty, fallback_pos, pos_start))
    pos_goal = cast(jnp.ndarray, jnp.where(is_empty, 2 * fallback_pos, pos_goal))

    return pos_start, pos_goal
