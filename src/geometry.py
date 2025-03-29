"""Geometry utility functions for the navigation environment.

This module provides functions to calculate distances between points and rectangles,
and sample valid positions for the robot and goal in the environment.
"""

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
