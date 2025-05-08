"""Room generation utilities for numpy implementation.

This module provides functions to generate room layouts using numpy.
"""

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class TileType(IntEnum):
    """Enumeration of tile types used in the grid representation."""

    FREE = 0  # Empty space
    WALL = 1  # Wall/obstacle


class RoomParams:
    """Parameters for room generation."""

    def __init__(
        self,
        size: float = 8.0,
        grid_size: int = 16,
        target_carved_percent: float = 0.8,
        num_rooms: int = 256,
    ):
        """
        Initialize room parameters.

        Args:
            size: Physical size of the room in meters
            grid_size: Number of grid cells in each dimension
            target_carved_percent: Target fraction of inner cells to carve out (0 to 1)
            num_rooms: Number of rooms to generate. Half used for training, half for evaluation.
        """
        self.size = size
        self.grid_size = grid_size
        self.target_carved_percent = target_carved_percent
        self.num_rooms = num_rooms


def generate_rooms(rng: np.random.Generator, params: RoomParams) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multiple room layouts sequentially.

    Args:
        rng: Numpy random number generator for reproducible generation
        params: Parameters for room generation

    Returns:
        Tuple of (obstacles_batch, free_positions_batch) containing batched room data
    """
    obstacles_batch = np.zeros((params.num_rooms, params.grid_size * params.grid_size, 4))
    free_positions_batch = np.zeros((params.num_rooms, params.grid_size * params.grid_size, 2))

    for i in range(params.num_rooms):
        # Create a new RNG for each room to ensure reproducibility
        room_seed = rng.integers(0, 2**32)
        room_rng = np.random.RandomState(room_seed)
        obstacles, free_positions = generate_room(room_rng, params)
        obstacles_batch[i] = obstacles
        free_positions_batch[i] = free_positions

    return obstacles_batch, free_positions_batch


def generate_room(rng: np.random.RandomState, params: RoomParams) -> Tuple[np.ndarray, np.ndarray]:
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

    Args:
        rng: Numpy random number generator for reproducible generation
        params: Parameters for room generation

    Returns:
        Tuple of (obstacles, free_positions):
            - obstacles: Array of [x, y, width, height] for each wall tile
            - free_positions: Array of [x, y] coordinates for free spaces
    """
    # Initialize grid parameters
    tile_size = params.size / params.grid_size
    total_cells = params.grid_size * params.grid_size
    target_carved_cells = int(np.floor(total_cells * params.target_carved_percent))
    max_carving_steps = int(np.floor(total_cells * params.target_carved_percent * 2))

    # Initialize grid with all walls
    grid = np.full((params.grid_size, params.grid_size), TileType.WALL, dtype=np.int32)

    # Create mask for valid carving area (inner cells only)
    carving_mask = np.pad(
        np.ones((params.grid_size - 2, params.grid_size - 2), dtype=np.int32),
        ((1, 1), (1, 1)),
        constant_values=0,
    )

    # Start at center
    center_x = params.grid_size // 2
    center_y = params.grid_size // 2
    grid[center_y, center_x] = TileType.FREE  # Carve initial position

    # Define movement directions (up, right, down, left)
    directions = np.array(
        [
            [0, -1],  # up
            [1, 0],  # right
            [0, 1],  # down
            [-1, 0],  # left
        ]
    )

    # Run the carving loop
    current_x, current_y = center_x, center_y
    carved_count = 1
    steps = 0

    while carved_count < target_carved_cells and steps < max_carving_steps:
        # Choose random direction
        direction_idx = rng.randint(0, 4)
        dx, dy = directions[direction_idx]

        # Calculate new position
        new_x = current_x + dx
        new_y = current_y + dy

        # Check if move is valid (within carving mask)
        is_valid_move = carving_mask[new_y, new_x] > 0

        # Update position if valid
        if is_valid_move:
            current_x, current_y = new_x, new_y

        # Carve current position
        if grid[current_y, current_x] == TileType.WALL:
            grid[current_y, current_x] = TileType.FREE
            carved_count += 1

        # Random return to center (5% chance)
        if rng.random() < 0.05:
            current_x, current_y = center_x, center_y

        steps += 1

    # Create coordinate mesh
    mesh_y, mesh_x = np.mgrid[0 : params.grid_size, 0 : params.grid_size]

    # Create the full set of obstacle rectangles
    obstacle_x = mesh_x * tile_size
    obstacle_y = mesh_y * tile_size
    obstacle_width = np.full(mesh_x.shape, tile_size)
    obstacle_height = np.full(mesh_x.shape, tile_size)

    # Create mask arrays from grid
    wall_mask = (grid == TileType.WALL).astype(np.float32)
    free_mask = (grid == TileType.FREE).astype(np.float32)

    # Apply mask to zero out non-wall cells for obstacles
    masked_obstacle_x = obstacle_x * wall_mask
    masked_obstacle_y = obstacle_y * wall_mask
    masked_obstacle_w = obstacle_width * wall_mask
    masked_obstacle_h = obstacle_height * wall_mask

    # Create obstacles array that includes all grid cells
    all_obstacles = np.stack(
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
    all_free_positions = np.stack([masked_free_x.flatten(), masked_free_y.flatten()], axis=1)

    return all_obstacles, all_free_positions


def sample_position(rng: np.random.RandomState, positions: np.ndarray) -> np.ndarray:
    """Sample a single position from an array of valid positions.

    Args:
        rng: Numpy random number generator
        positions: Array of positions with shape (n, 2), where zeros indicate invalid positions

    Returns:
        A [x, y] coordinate sampled from valid positions.
        If no valid positions exist, returns a fallback position.
    """
    # Get position mask and count
    is_valid = np.any(positions != 0, axis=1)

    if not np.any(is_valid):
        # Fallback if no valid positions
        return np.zeros(2)

    # Create random values for all positions
    rand_vals = rng.random(positions.shape[0])

    # Mask invalid positions
    masked_rand = np.where(is_valid, rand_vals, -1.0)

    # Select position with highest random value
    return positions[np.argmax(masked_rand)]
