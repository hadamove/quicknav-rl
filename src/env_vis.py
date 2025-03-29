from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constants import GOAL, OBSTACLE, ROBOT_LENGTH, ROBOT_WIDTH, WALL, WHEEL_LENGTH, WHEEL_WIDTH
from env import EnvParams, EnvState

# Type alias for RGB color values
RGBColor = Tuple[int, int, int]


@dataclass
class VisualizationTheme:
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


# Default theme
DEFAULT_THEME = VisualizationTheme()

# Anti-aliasing scale factor
AA_SCALE = 4

Frame = np.ndarray


def world_to_pixels(x: float, y: float, scale: float, img_height: int) -> Tuple[int, int]:
    """Convert world coordinates to pixel coordinates"""
    return int(x * scale), int(img_height - y * scale)


def rotate_point(x: float, y: float, angle: float) -> Tuple[float, float]:
    """Rotate a point around the origin by the given angle"""
    ca, sa = np.cos(angle), np.sin(angle)
    return x * ca - y * sa, x * sa + y * ca


def get_polygon_pixels(
    corners: List[Tuple[float, float]], cx: float, cy: float, angle: float, scale: float, img_height: int
) -> List[Tuple[int, int]]:
    """Convert a polygon in world coordinates to pixel coordinates"""
    return [
        world_to_pixels(cx + px, cy + py, scale, img_height)
        for (lx, ly) in corners
        for (px, py) in [rotate_point(lx, ly, angle)]
    ]


def render_frame(
    state: EnvState,
    params: EnvParams,
    img_width: int = 600,
    img_height: int = 600,
    theme: Optional[VisualizationTheme] = None,
) -> Frame:
    """Render a frame of the environment state"""
    theme = theme or DEFAULT_THEME

    # Create high-resolution image for antialiasing
    high_width, high_height = img_width * AA_SCALE, img_height * AA_SCALE
    img = Image.new("RGB", (high_width, high_height), color=theme.background)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12 * AA_SCALE)
    scale = min(high_width, high_height) / params.arena_size

    # Convert state fields to numpy for visualization
    x_np = float(state.x)
    y_np = float(state.y)
    theta_np = float(state.theta)
    goal_x_np = float(state.goal_x)
    goal_y_np = float(state.goal_y)
    obstacles_np = np.array(state.obstacles) if state.obstacles is not None else np.empty((0, 4))

    # Draw obstacles
    for ox, oy, ow, oh in obstacles_np:
        draw.rectangle(
            [world_to_pixels(ox, oy + oh, scale, high_height), world_to_pixels(ox + ow, oy, scale, high_height)],
            fill=theme.obstacle,
        )

    # Draw goal
    gx, gy = world_to_pixels(goal_x_np, goal_y_np, scale, high_height)
    r = int(params.goal_tolerance * scale)
    draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=theme.goal, width=1)

    # Get lidar data from state
    lidar_distances = np.array(state.lidar_distances)
    lidar_collision_types = np.array(state.lidar_collision_types)

    # Simulate lidar beams with precomputed distances
    fov_rad = np.radians(params.lidar_fov)
    beam_angles = theta_np + np.linspace(-fov_rad / 2, fov_rad / 2, params.lidar_num_beams)

    for i, angle in enumerate(beam_angles):
        dx, dy = np.cos(angle), np.sin(angle)
        # Get precomputed distance and collision type
        d = lidar_distances[i]
        collision_type = lidar_collision_types[i]

        # Determine beam color based on collision type
        if collision_type == GOAL:
            color = theme.goal_beam
        elif collision_type == WALL or collision_type == OBSTACLE:
            color = theme.wall_beam
        else:
            color = theme.default_beam

        # Draw the beam
        start = world_to_pixels(x_np, y_np, scale, high_height)
        end = world_to_pixels(x_np + d * dx, y_np + d * dy, scale, high_height)
        draw.line([start, end], fill=color, width=max(1, int(0.5 * AA_SCALE)))

    # Draw robot chassis
    chassis = [
        (ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
        (-ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
        (-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
        (ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
    ]
    draw.polygon(get_polygon_pixels(chassis, x_np, y_np, theta_np, scale, high_height), fill=theme.chassis)

    # Draw wheels
    wheels = [
        [
            (WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
            (-WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
            (-WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
            (WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
        ],
        [
            (WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
            (-WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
            (-WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
            (WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
        ],
    ]
    for wheel in wheels:
        draw.polygon(get_polygon_pixels(wheel, x_np, y_np, theta_np, scale, high_height), fill=theme.wheel)

    # Draw text info
    reward_val = float(state.accumulated_reward)
    draw.text((5 * AA_SCALE, 5 * AA_SCALE), f"T: {int(state.time)}", fill=theme.text, font=font)
    draw.text((5 * AA_SCALE, 20 * AA_SCALE), f"R: {reward_val:.2f}", fill=theme.text, font=font)

    # Downsample for antialiasing
    return np.array(img.resize((img_width, img_height), resample=Image.Resampling.LANCZOS))


def save_gif(frames: Sequence[Frame], filename: str, duration_per_frame: float = 0.1):
    """Save a sequence of frames as a GIF animation"""
    try:
        imageio.mimsave(filename, list(frames), duration=duration_per_frame, loop=0)
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def create_episode_animation(
    env_states: List[EnvState],
    params: EnvParams,
    filename: str = "episode.gif",
    width: int = 600,
    height: int = 600,
    fps: float = 10,
    theme: Optional[VisualizationTheme] = None,
):
    """Create and save an animation of an episode from a list of environment states"""
    frames = [render_frame(state, params, width, height, theme) for state in env_states]
    save_gif(frames, filename, 1.0 / fps)
    return frames
