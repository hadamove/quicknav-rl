"""Visualization utilities for the environment state"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env import EnvParams, EnvState
from lidar import Collision

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


Frame = np.ndarray
"""Type alias for a single frame of the environment visualization"""

Point = Tuple[float, float]
"""Type alias for a 2D point in the environment"""


def render_frame(
    state: EnvState,
    params: EnvParams,
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
    scale = min(aa_width, aa_height) / params.arena_size

    # Convert state fields to numpy for visualization
    x, y = float(state.x), float(state.y)
    theta = float(state.theta)
    goal_x, goal_y = float(state.goal_x), float(state.goal_y)
    obstacles = np.array(state.obstacles)

    # Draw environment elements
    _draw_obstacles(draw, obstacles, scale, aa_height, theme)
    _draw_goal(draw, (goal_x, goal_y), params.goal_tolerance, scale, aa_height, theme)

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
    )

    # Draw robot body and wheels
    _draw_robot(draw, (x, y), theta, params.robot_radius, scale, aa_height, theme)

    # Draw info
    draw.text((5 * aa_scale, 5 * aa_scale), f"T: {int(state.time)}", fill=theme.text, font=font)
    draw.text((5 * aa_scale, 20 * aa_scale), f"R: {float(state.accumulated_reward):.2f}", fill=theme.text, font=font)

    # Downsample for antialiasing
    return np.array(img.resize((img_width, img_height), resample=Image.Resampling.LANCZOS))


def save_gif(frames: Sequence[Frame], filename: str, duration_per_frame: float = 0.1):
    """Save a sequence of frames as a GIF animation"""
    try:
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
):
    """Draw lidar beams with their collision types."""
    fov_rad = np.radians(lidar_fov)
    beam_angles = float(state.theta) + np.linspace(-fov_rad / 2, fov_rad / 2, lidar_num_beams)

    for i, angle in enumerate(beam_angles):
        dx, dy = np.cos(angle), np.sin(angle)
        d = float(state.lidar_distances[i])
        collision_type = state.lidar_collision_types[i]

        # Determine beam color based on collision type
        color = _get_beam_color(Collision(collision_type), theme)

        # Draw the beam
        x, y = float(state.x), float(state.y)
        start = _world_to_pixels(x, y, scale, img_height)
        end = _world_to_pixels(x + d * dx, y + d * dy, scale, img_height)
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
