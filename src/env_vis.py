from typing import List, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env import EnvParams, EnvState

# TODO: refactor this whole file

# Dimensions and Colors
ROBOT_LENGTH = 0.3
ROBOT_WIDTH = 0.2
WHEEL_LENGTH = 0.15
WHEEL_WIDTH = 0.05

COLORS = {
    "background": (255, 255, 255),
    "chassis": (100, 100, 100),
    "wheel": (50, 50, 50),
    "goal": (63, 176, 0),
    "default_beam": (30, 30, 30),
    "wall_beam": (255, 50, 50),
    "goal_beam": (70, 140, 40),
    "text": (0, 0, 0),
}

Frame = np.ndarray
AA_SCALE = 4


def world_to_pixels(x: float, y: float, scale: float, img_height: int) -> Tuple[int, int]:
    return int(x * scale), int(img_height - y * scale)


def rotate_point(x: float, y: float, angle: float) -> Tuple[float, float]:
    ca, sa = np.cos(angle), np.sin(angle)
    return x * ca - y * sa, x * sa + y * ca


def get_polygon_pixels(
    corners: List[Tuple[float, float]], cx: float, cy: float, angle: float, scale: float, img_height: int
) -> List[Tuple[int, int]]:
    return [
        world_to_pixels(cx + px, cy + py, scale, img_height)
        for (lx, ly) in corners
        for (px, py) in [rotate_point(lx, ly, angle)]
    ]


def compute_obstacle_distance(x: float, y: float, dx: float, dy: float, obstacles: np.ndarray) -> float:
    eps = 1e-6
    distances = []
    for ox, oy, ow, oh in obstacles:
        if abs(dx) < eps:
            tmin_x, tmax_x = -np.inf, np.inf
        else:
            t1, t2 = (ox - x) / dx, ((ox + ow) - x) / dx
            tmin_x, tmax_x = min(t1, t2), max(t1, t2)
        if abs(dy) < eps:
            tmin_y, tmax_y = -np.inf, np.inf
        else:
            t1, t2 = (oy - y) / dy, ((oy + oh) - y) / dy
            tmin_y, tmax_y = min(t1, t2), max(t1, t2)
        t_entry, t_exit = max(tmin_x, tmin_y), min(tmax_x, tmax_y)
        if t_exit >= t_entry > 0:
            distances.append(t_entry)
    return min(distances) if distances else np.inf


def render_frame(state: EnvState, params: EnvParams, img_width: int = 600, img_height: int = 600) -> Frame:
    # Create high-resolution image for antialiasing
    high_width, high_height = img_width * AA_SCALE, img_height * AA_SCALE
    img = Image.new("RGB", (high_width, high_height), color=COLORS["background"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    scale = min(high_width, high_height) / params.arena_size

    # Draw obstacles
    if state.obstacles is not None:
        for ox, oy, ow, oh in np.array(state.obstacles):
            draw.rectangle(
                [world_to_pixels(ox, oy + oh, scale, high_height), world_to_pixels(ox + ow, oy, scale, high_height)],
                fill=(200, 200, 200),
            )

    # Draw goal
    gx, gy = world_to_pixels(state.goal_x, state.goal_y, scale, high_height)
    r = int(0.02 * min(high_width, high_height))
    draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=COLORS["goal"], width=1)

    # Lidar beams with collision detection
    obs = np.array(state.obstacles) if state.obstacles is not None else np.empty((0, 4))
    fov_rad = np.radians(params.lidar_fov)
    beam_angles = state.theta + np.linspace(-fov_rad / 2, fov_rad / 2, params.lidar_num_beams)
    for angle in beam_angles:
        dx, dy = np.cos(angle), np.sin(angle)
        wall_d = compute_obstacle_distance(state.x, state.y, dx, dy, obs)
        ocx, ocy = state.x - state.goal_x, state.y - state.goal_y
        b = 2 * (ocx * dx + ocy * dy)
        c = ocx**2 + ocy**2 - params.goal_tolerance**2
        disc = b * b - 4 * c
        t_goal = min(
            [t for t in [(-b - np.sqrt(disc)) / 2, (-b + np.sqrt(disc)) / 2] if disc >= 0 and t >= 0] or [np.inf]
        )
        d = min(wall_d, t_goal, params.lidar_max_distance)
        if d == t_goal and d < params.lidar_max_distance:
            color = COLORS["goal_beam"]
        elif d == wall_d and d < params.lidar_max_distance:
            color = COLORS["wall_beam"]
        else:
            color = COLORS["default_beam"]
        start = world_to_pixels(state.x, state.y, scale, high_height)
        end = world_to_pixels(state.x + d * dx, state.y + d * dy, scale, high_height)
        draw.line([start, end], fill=color, width=1)

    # Draw robot chassis
    chassis = [
        (ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
        (-ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
        (-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
        (ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
    ]
    draw.polygon(get_polygon_pixels(chassis, state.x, state.y, state.theta, scale, high_height), fill=COLORS["chassis"])

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
        draw.polygon(get_polygon_pixels(wheel, state.x, state.y, state.theta, scale, high_height), fill=COLORS["wheel"])

    # Draw text info
    draw.text((5 * AA_SCALE, 5 * AA_SCALE), f"T: {int(state.time)}", fill=COLORS["text"], font=font)
    draw.text((5 * AA_SCALE, 20 * AA_SCALE), f"R: {state.accumulated_reward:.2f}", fill=COLORS["text"], font=font)

    # Downsample for antialiasing
    return np.array(img.resize((img_width, img_height), resample=Image.Resampling.LANCZOS))


def save_gif(frames: Sequence[Frame], filename: str, duration_per_frame: float = 0.1):
    try:
        imageio.mimsave(filename, list(frames), duration=duration_per_frame, loop=0)
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
