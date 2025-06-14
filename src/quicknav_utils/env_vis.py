"""Visualization utilities for the environment state

Works with both JAX and NumPy implementations of the environment.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import quicknav_jax as env_jax
    import quicknav_numpy as env_np

    EnvState = env_jax.EnvState | env_np.EnvState
    NavigationEnvParams = env_jax.NavigationEnvParams | env_np.NavigationEnvParams
else:
    EnvState = Any
    NavigationEnvParams = Any

from .collision import Collision

# Type alias for RGB color values
RGBColor = Tuple[int, int, int]


@dataclass
class Theme:
    """Theme configuration for environment visualization"""

    background: RGBColor = (255, 255, 255)
    chassis: RGBColor = (100, 100, 100)
    wheel: RGBColor = (50, 50, 50)
    goal: RGBColor = (63, 176, 0)
    obstacle: RGBColor = (200, 200, 200)
    default_beam: RGBColor = (30, 30, 30)
    wall_beam: RGBColor = (255, 50, 50)
    goal_beam: RGBColor = (70, 140, 40)
    text: RGBColor = (0, 0, 0)
    path: RGBColor = (173, 216, 230)  # Light blue color for robot path


Frame = np.ndarray
"""Type alias for a single frame of the environment visualization"""

Point = Tuple[float, float]
"""Type alias for a 2D point in the environment"""


def render_frame(
    state: EnvState,
    params: NavigationEnvParams,
    img_width: int = 600,
    img_height: int = 600,
    theme: Theme = Theme(),
    aa_scale: int = 4,
) -> Frame:
    """Render a frame of the environment state"""
    # Create high-resolution image for antialiasing
    aa_width, aa_height = img_width * aa_scale, img_height * aa_scale

    img = Image.new("RGB", (aa_width, aa_height), color=theme.background)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12 * aa_scale)
    scale = min(aa_width, aa_height) / params.rooms.size

    # Convert state fields to numpy for visualization
    x, y = float(state.x), float(state.y)
    theta = float(state.theta)
    goal_x, goal_y = float(state.goal_x), float(state.goal_y)
    obstacles = np.array(state.obstacles)

    # Draw environment elements
    _draw_obstacles(draw, obstacles, scale, aa_height, theme)
    _draw_goal(draw, (goal_x, goal_y), params.goal_tolerance, scale, aa_height, theme)

    # Draw the robot's path
    _draw_path(draw, state, scale, aa_height, theme)

    # Draw lidar beams
    _draw_lidar(
        draw,
        state,
        params.lidar_fov,
        params.lidar_num_beams,
        scale,
        aa_height,
        aa_scale,
        theme,
        params,
    )

    # Draw robot body and wheels
    _draw_robot(draw, (x, y), theta, params.robot_radius, scale, aa_height, theme)

    # Draw info
    draw.text((5 * aa_scale, 5 * aa_scale), f"T: {int(state.steps)}", fill=theme.text, font=font)
    draw.text((5 * aa_scale, 20 * aa_scale), f"R: {float(state.accumulated_reward):.2f}", fill=theme.text, font=font)

    # Downsample for antialiasing
    return np.array(img.resize((img_width, img_height), resample=Image.Resampling.LANCZOS))


def save_gif(frames: List[Frame] | List[List[Frame]], filename: Path, duration_per_frame: float = 0.1):
    """Save a sequence of frames as a GIF animation"""
    if len(frames) > 0 and isinstance(frames[0], list):
        # If frames is a list of lists, flatten it
        frames = [frame for episode in frames for frame in episode]

    try:
        os.makedirs(filename.parent, exist_ok=True)
        imageio.mimsave(filename, list(frames), duration=duration_per_frame, loop=0)
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def _draw_obstacles(draw: ImageDraw.ImageDraw, obstacles: np.ndarray, scale: float, img_height: int, theme: Theme):
    """Draw obstacles in the environment."""
    for ox, oy, ow, oh in obstacles:
        draw.rectangle(
            [_world_to_pixels(ox, oy + oh, scale, img_height), _world_to_pixels(ox + ow, oy, scale, img_height)],
            fill=theme.obstacle,
        )


def _draw_goal(
    draw: ImageDraw.ImageDraw, goal: Point, goal_tolerance: float, scale: float, img_height: int, theme: Theme
):
    """Draw the goal point with its tolerance radius."""
    goal_x, goal_y = goal
    gx, gy = _world_to_pixels(goal_x, goal_y, scale, img_height)
    r = int(goal_tolerance * scale)
    draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=theme.goal, outline=theme.goal)


def _draw_lidar(
    draw: ImageDraw.ImageDraw,
    state: EnvState,
    lidar_fov: float,
    lidar_num_beams: int,
    scale: float,
    img_height: int,
    aa_scale: int,
    theme: Theme,
    params: NavigationEnvParams,
):
    """Draw lidar beams with their collision types."""
    fov_rad = np.radians(lidar_fov)
    beam_angles = float(state.theta) + np.linspace(-fov_rad / 2, fov_rad / 2, lidar_num_beams)

    for i, angle in enumerate(beam_angles):
        dx, dy = np.cos(angle), np.sin(angle)
        d = float(state.lidar_distances[i])
        collision_type = int(state.lidar_collision_types[i])

        # Determine beam color based on collision type
        color = _get_beam_color(Collision(collision_type), theme)

        # Calculate starting point on robot perimeter
        x, y = float(state.x), float(state.y)

        # Start point is on the perimeter of the robot
        start_x = x + params.robot_radius * dx
        start_y = y + params.robot_radius * dy

        # End point is at the measured distance from the perimeter start point
        end_x = start_x + d * dx
        end_y = start_y + d * dy

        # Convert to pixel coordinates
        start = _world_to_pixels(start_x, start_y, scale, img_height)
        end = _world_to_pixels(end_x, end_y, scale, img_height)

        # Draw the beam
        draw.line([start, end], fill=color, width=max(1, int(0.5 * aa_scale)))


def _get_beam_color(collision_type: Collision, theme: Theme) -> RGBColor:
    """Get the color for a lidar beam based on its collision type"""
    if collision_type == Collision.Goal:
        return theme.goal_beam
    elif collision_type == Collision.Obstacle:
        return theme.wall_beam
    else:
        return theme.default_beam


def _draw_robot(
    draw: ImageDraw.ImageDraw,
    pos: Point,
    theta: float,
    robot_radius: float,
    scale: float,
    img_height: int,
    theme: Theme,
):
    """Draw the robot chassis and wheels."""
    x, y = pos
    chassis, wheels = _get_robot_dimensions(robot_radius)

    # Draw robot chassis
    draw.polygon(_get_polygon_pixels(chassis, x, y, theta, scale, img_height), fill=theme.chassis)

    # Draw wheels
    for wheel in wheels:
        draw.polygon(_get_polygon_pixels(wheel, x, y, theta, scale, img_height), fill=theme.wheel)


def _get_robot_dimensions(robot_radius: float) -> Tuple[List[Point], List[List[Point]]]:
    """Calculate robot chassis and wheel dimensions based on robot radius.

    Returns:
        Tuple containing:
        - List of chassis corner points
        - List of wheel corner points for each wheel
    """
    # Chassis dimensions: 80% of diameter for length, 60% of diameter for width
    chassis_length = robot_radius * 1.6  # 80% of diameter
    chassis_width = robot_radius * 1.2  # 60% of diameter

    # Wheel dimensions: 40% of chassis length, 20% of chassis width
    wheel_length = chassis_length * 0.4
    wheel_width = chassis_width * 0.2

    # Chassis corners (centered at origin)
    chassis = [
        (chassis_length / 2, chassis_width / 2),
        (-chassis_length / 2, chassis_width / 2),
        (-chassis_length / 2, -chassis_width / 2),
        (chassis_length / 2, -chassis_width / 2),
    ]

    # Wheel corners (centered at origin)
    wheels = [
        [
            (wheel_length / 2, chassis_width / 2 + wheel_width / 2),
            (-wheel_length / 2, chassis_width / 2 + wheel_width / 2),
            (-wheel_length / 2, chassis_width / 2 - wheel_width / 2),
            (wheel_length / 2, chassis_width / 2 - wheel_width / 2),
        ],
        [
            (wheel_length / 2, -chassis_width / 2 + wheel_width / 2),
            (-wheel_length / 2, -chassis_width / 2 + wheel_width / 2),
            (-wheel_length / 2, -chassis_width / 2 - wheel_width / 2),
            (wheel_length / 2, -chassis_width / 2 - wheel_width / 2),
        ],
    ]

    return chassis, wheels


def _world_to_pixels(x: float, y: float, scale: float, img_height: int) -> Tuple[int, int]:
    """Convert world coordinates to pixel coordinates"""
    return int(x * scale), int(img_height - y * scale)


def _rotate_point(x: float, y: float, angle: float) -> Point:
    """Rotate a point around the origin by the given angle"""
    ca, sa = np.cos(angle), np.sin(angle)
    return x * ca - y * sa, x * sa + y * ca


def _get_polygon_pixels(
    corners: List[Point], cx: float, cy: float, angle: float, scale: float, img_height: int
) -> List[Tuple[int, int]]:
    """Convert a polygon in world coordinates to pixel coordinates"""
    return [
        _world_to_pixels(cx + px, cy + py, scale, img_height)
        for (lx, ly) in corners
        for (px, py) in [_rotate_point(lx, ly, angle)]
    ]


def _draw_path(
    draw: ImageDraw.ImageDraw,
    state: EnvState,
    scale: float,
    img_height: int,
    theme: Theme,
):
    """Draw the robot's position history as a path."""
    # Only draw positions up to the current step
    path = np.array(state.position_history[: state.steps + 1])

    # Need at least 2 points to draw a line
    if len(path) >= 2:
        points = []
        for i in range(len(path)):
            if np.all(path[i] != 0) or i == 0:  # Skip zero entries (uninitialized positions)
                px, py = _world_to_pixels(path[i, 0], path[i, 1], scale, img_height)
                points.append((px, py))

        # Draw the path if we have points
        if len(points) >= 2:
            draw.line(points, fill=theme.path, width=max(1, int(1.5 * scale / 50)))
