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
class RoomParams:
    """Parameters for room generation."""

    size: float = 8.0
    """Physical size of the room in meters"""
    grid_size: int = 16
    """Number of grid cells in each dimension"""
    target_carved_percent: float = 0.8
    """Target fraction of inner cells to carve out (0 to 1)"""
    num_rooms: int = 256
    """Number of rooms to generate"""


def generate_rooms(key: Any, params: RoomParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate multiple room layouts in parallel using vmap.

    Args:
        key: JAX random key for reproducible generation
        params: Parameters for room generation

    Returns:
        Tuple of (obstacles_batch, free_positions_batch) containing batched room data
    """
    # Create parallel random keys for each room
    keys = jax.random.split(key, params.num_rooms)

    # Vectorize the room generation function across the batch dimension
    batch_generate_room = jax.vmap(generate_room, in_axes=(0, None))

    # Generate rooms in parallel
    obstacles_batch, free_positions_batch = batch_generate_room(keys, params)

    return obstacles_batch, free_positions_batch


@struct.dataclass
class RoomGenerationState:
    """State for the room generation random walk."""

    grid: jnp.ndarray  # Current grid state (TileType.WALL=1, TileType.FREE=0)
    key: Any  # Random key for next step
    keys: jnp.ndarray  # Pre-generated random keys for all steps
    key_idx: jnp.ndarray  # Index into the keys array
    carved_count: jnp.ndarray  # Number of cells carved so far (as JAX array)
    current_x: jnp.ndarray  # Current x position (as JAX array)
    current_y: jnp.ndarray  # Current y position (as JAX array)
    steps: jnp.ndarray  # Number of steps taken (as JAX array)


def generate_room(key: Any, params: RoomParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a room layout using a random walk carving algorithm.

    The algorithm works as follows:
    1. Start with a grid where all cells are walls (1s)
    2. Begin at the center of the grid
    3. Repeatedly:
        - Choose a random direction (up, right, down, left)
        - If the move is valid (within inner grid), move there
        - Carve out the cell (set to 0)
        - Occasionally (5% chance) return to center to avoid getting stuck
        - Continue until target percentage (`target_carved_percent`) of cells are carved or max steps reached
    4. Convert the grid to physical coordinates:
        - Walls become obstacle rectangles with position and size
        - Free spaces become center points for valid positions

    This approach generates connected rooms with organic-looking pathways,
    ensuring the robot can reach any free space from any other free space.
    The outer walls are always preserved as they're outside the carving mask.

    Args:
        key: JAX random key for reproducible generation
        params: Parameters for room generation

    Returns:
        Tuple of (obstacles, free_positions):
            - obstacles: Array of [x, y, width, height] for each wall tile
            - free_positions: Array of [x, y] coordinates for free spaces
    """
    # Initialize grid parameters
    tile_size = params.size / params.grid_size
    total_cells = params.grid_size * params.grid_size
    target_carved_cells = jnp.floor(total_cells * params.target_carved_percent).astype(jnp.int32)
    max_carving_steps = jnp.floor(total_cells * params.target_carved_percent * 2).astype(jnp.int32)

    # Initialize grid with all walls (more efficient broadcasting)
    grid = jnp.full((params.grid_size, params.grid_size), TileType.WALL, dtype=jnp.int32)

    # Create mask for valid carving area (inner cells only)
    # Use jnp.pad instead of manually constructing the array
    carving_mask = jnp.pad(
        jnp.ones((params.grid_size - 2, params.grid_size - 2), dtype=jnp.int32),
        ((1, 1), (1, 1)),
        constant_values=0,
    )

    # Start at center
    center_x = params.grid_size // 2
    center_y = params.grid_size // 2
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

    # Pre-generate random keys for all possible steps
    # The maximum steps is bounded by total_cells (we can carve at most all cells)
    keys_per_step = 2  # Two random values per step (direction and reset)
    max_keys = total_cells * keys_per_step
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, max_keys)

    def carve_step(state: RoomGenerationState) -> RoomGenerationState:
        """Perform one step of the random walk carving algorithm."""
        # Get the next pre-generated keys (mod to ensure we don't go out of bounds)
        key_idx = state.key_idx % max_keys
        direction_key = state.keys[key_idx]
        reset_key = state.keys[(key_idx + 1) % max_keys]
        next_key_idx = key_idx + keys_per_step

        # Choose random direction
        direction_idx = jax.random.randint(direction_key, (), 0, 4)
        dx, dy = jnp.take(directions, direction_idx, axis=0)

        # Calculate new position
        new_x = state.current_x + dx
        new_y = state.current_y + dy

        # Check if move is valid (within carving mask)
        is_valid_move = carving_mask[new_y, new_x] > 0

        # Update position if valid
        current_x = cast(jnp.ndarray, jnp.where(is_valid_move, new_x, state.current_x))
        current_y = cast(jnp.ndarray, jnp.where(is_valid_move, new_y, state.current_y))

        # Carve current position and update count atomically
        old_value = state.grid[current_y, current_x]
        new_grid = state.grid.at[current_y, current_x].set(TileType.FREE)
        new_carved_count = state.carved_count + (old_value == TileType.WALL)

        # Random return to center (5% chance)
        should_reset = jax.random.uniform(reset_key) < 0.05
        current_x = cast(jnp.ndarray, jnp.where(should_reset, jnp.array(center_x), current_x))
        current_y = cast(jnp.ndarray, jnp.where(should_reset, jnp.array(center_y), current_y))

        return RoomGenerationState(
            grid=new_grid,
            key=state.key,
            keys=state.keys,
            key_idx=next_key_idx,
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
        keys=keys,
        key_idx=jnp.array(0),
        carved_count=jnp.array(1),
        current_x=jnp.array(center_x),
        current_y=jnp.array(center_y),
        steps=jnp.array(0),
    )
    final_state = jax.lax.while_loop(should_continue_carving, carve_step, initial_state)
    final_grid = final_state.grid

    # Create coordinate mesh once and reuse
    mesh_y, mesh_x = jnp.mgrid[0 : params.grid_size, 0 : params.grid_size]

    # Create the full set of obstacle rectangles
    # Every cell in the grid gets a rectangle with proper dimensions
    obstacle_x = mesh_x * tile_size
    obstacle_y = mesh_y * tile_size
    obstacle_width = jnp.full(mesh_x.shape, tile_size)
    obstacle_height = jnp.full(mesh_x.shape, tile_size)

    # Create mask arrays from grid
    wall_mask = (final_grid == TileType.WALL).astype(jnp.float32)
    free_mask = (final_grid == TileType.FREE).astype(jnp.float32)

    # Apply mask to zero out non-wall cells for obstacles
    # This is a JAX-friendly alternative to boolean indexing
    masked_obstacle_x = obstacle_x * wall_mask
    masked_obstacle_y = obstacle_y * wall_mask
    masked_obstacle_w = obstacle_width * wall_mask
    masked_obstacle_h = obstacle_height * wall_mask

    # Create obstacles array that includes all grid cells
    # Each entry is (x, y, width, height) with zeros for non-obstacle cells
    all_obstacles = jnp.stack(
        [
            masked_obstacle_x.flatten(),
            masked_obstacle_y.flatten(),
            masked_obstacle_w.flatten(),
            masked_obstacle_h.flatten(),
        ],
        axis=1,
    )

    # Create free position coordinates (centers of free cells)
    free_x = (mesh_x + 0.5) * tile_size
    free_y = (mesh_y + 0.5) * tile_size

    # Apply mask to zero out non-free cells
    masked_free_x = free_x * free_mask
    masked_free_y = free_y * free_mask

    # Create free positions array with zeros for non-free cells
    all_free_positions = jnp.stack([masked_free_x.flatten(), masked_free_y.flatten()], axis=1)

    return all_obstacles, all_free_positions


def sample_position(key: Any, positions: jnp.ndarray) -> jnp.ndarray:
    """Sample a single position from an array of valid positions.

    Args:
        key: JAX random key for reproducible sampling
        positions: Array of positions with shape (n, 2), where zeros indicate invalid positions

    Returns:
        A [x, y] coordinate sampled from valid positions.
        If no valid positions exist, returns a fallback position.
    """
    # Get position mask and count
    is_valid = jnp.any(positions != 0, axis=1)

    # Create random values for all positions
    rand_vals = jax.random.uniform(key, (positions.shape[0],))

    # Mask invalid positions (set their random values to -1 so they won't be selected)
    masked_rand = jnp.where(is_valid, rand_vals, -1.0)

    # Select position with highest random value
    return positions[jnp.argmax(masked_rand)]
