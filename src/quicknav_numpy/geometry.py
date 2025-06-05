"""Geometry utility functions for the navigation environment.

This module provides functions to calculate distances between points and rectangles,
and sample valid positions for the robot and goal in the environment.
"""

from typing import Tuple

import numpy as np


def point_to_rectangle_distance(point: np.ndarray, rect: np.ndarray) -> float:
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
    closest_x = np.clip(point[0], rect_x_min, rect_x_max)
    closest_y = np.clip(point[1], rect_y_min, rect_y_max)

    # If point is inside the rectangle, distance is 0
    inside = (closest_x == point[0]) and (closest_y == point[1])

    # Calculate distance to closest point
    dx = closest_x - point[0]
    dy = closest_y - point[1]
    distance = np.sqrt(dx**2 + dy**2)

    # Return 0 if inside, otherwise the calculated distance
    return 0.0 if inside else distance


def handle_collision_with_sliding(
    current_x: float,
    current_y: float,
    new_x: float,
    new_y: float,
    obstacles: np.ndarray,
    robot_radius: float,
) -> Tuple[float, float]:
    """Handle collision with sliding behavior.

    When colliding with obstacles, slides along the edge instead of stopping completely.

    Args:
        current_x: Current x position of the robot
        current_y: Current y position of the robot
        new_x: Proposed new x position (may be in collision)
        new_y: Proposed new y position (may be in collision)
        obstacles: Array of obstacle rectangles [x, y, width, height]
        robot_radius: Radius of the robot for collision detection

    Returns:
        Tuple of (x, y) position after applying sliding logic
    """
    # Try to slide horizontally or vertically depending on the movement direction
    slide_x_pos = np.array([new_x, current_y])  # Try moving only in x direction
    slide_x_distances = np.array([point_to_rectangle_distance(slide_x_pos, obstacle) for obstacle in obstacles])
    can_slide_x = np.all(slide_x_distances >= robot_radius)

    slide_y_pos = np.array([current_x, new_y])  # Try moving only in y direction
    slide_y_distances = np.array([point_to_rectangle_distance(slide_y_pos, obstacle) for obstacle in obstacles])
    can_slide_y = np.all(slide_y_distances >= robot_radius)

    # If moving more horizontally than vertically, prioritize sliding in y
    movement_dx = abs(new_x - current_x)
    movement_dy = abs(new_y - current_y)
    horizontal_dominant = movement_dx >= movement_dy

    # Apply sliding logic
    if horizontal_dominant:
        if can_slide_y:
            slide_x = current_x
            slide_y = new_y
        elif can_slide_x:
            slide_x = new_x
            slide_y = current_y
        else:
            slide_x = current_x
            slide_y = current_y
    else:
        if can_slide_x:
            slide_x = new_x
            slide_y = current_y
        elif can_slide_y:
            slide_x = current_x
            slide_y = new_y
        else:
            slide_x = current_x
            slide_y = current_y

    return slide_x, slide_y
