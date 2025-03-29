"""
Common constants shared across environment and visualization.
"""

# Robot dimensions
ROBOT_LENGTH = 0.3
ROBOT_WIDTH = 0.2
WHEEL_LENGTH = 0.15
WHEEL_WIDTH = 0.05

# Collision type constants
MAX_DIST = 0  # No collision, max distance reached
WALL = 1  # Collision with arena boundary
OBSTACLE = 2  # Collision with obstacle
GOAL = 3  # Collision with goal
