"""
Shared enumerations for QuickNav environments.

Contains definitions that are common to both JAX and NumPy implementations.
"""

from enum import IntEnum


class Collision(IntEnum):
    """Enumeration of possible collision types for lidar beams."""

    MaxDist = 0  # No collision, max distance reached
    Obstacle = 1  # Collision with obstacle
    Goal = 2  # Collision with goal
