"""Geometry utility functions for the navigation environment.

This module provides functions to calculate distances between points and rectangles,
and sample valid positions for the robot and goal in the environment.
"""

from typing import Tuple

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


def handle_collision_with_sliding(
    current_x: jnp.ndarray,
    current_y: jnp.ndarray, 
    new_x: jnp.ndarray, 
    new_y: jnp.ndarray, 
    obstacles: jnp.ndarray, 
    robot_radius: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    slide_x_pos = jnp.array([new_x, current_y])  # Try moving only in x direction
    slide_x_distances = jax.vmap(point_to_rectangle_distance, in_axes=(None, 0))(slide_x_pos, obstacles)
    can_slide_x = jnp.all(slide_x_distances >= robot_radius)
    
    slide_y_pos = jnp.array([current_x, new_y])  # Try moving only in y direction
    slide_y_distances = jax.vmap(point_to_rectangle_distance, in_axes=(None, 0))(slide_y_pos, obstacles)
    can_slide_y = jnp.all(slide_y_distances >= robot_radius)
    
    # If moving more horizontally than vertically, prioritize sliding in y
    movement_dx = jnp.abs(new_x - current_x)
    movement_dy = jnp.abs(new_y - current_y)
    horizontal_dominant = movement_dx >= movement_dy
    
    # Apply sliding logic
    slide_x = jnp.where(
        horizontal_dominant,
        jnp.where(can_slide_y, current_x, jnp.where(can_slide_x, new_x, current_x)),
        jnp.where(can_slide_x, new_x, current_x),
    )
    
    slide_y = jnp.where(
        horizontal_dominant,
        jnp.where(can_slide_y, new_y, current_y),
        jnp.where(can_slide_x, current_y, jnp.where(can_slide_y, new_y, current_y)),
    )
    
    return slide_x, slide_y
