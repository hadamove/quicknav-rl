from typing import List, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env import EnvParams, EnvState

# Robot dimensions
ROBOT_LENGTH = 0.3
ROBOT_WIDTH = 0.2
WHEEL_LENGTH = 0.15
WHEEL_WIDTH = 0.05

# Rendering colors
BACKGROUND_COLOR = (255, 255, 255)
ROBOT_CHASSIS_COLOR = (100, 100, 100)
ROBOT_WHEEL_COLOR = (50, 50, 50)
GOAL_COLOR = (0, 255, 0)
LIDAR_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 0, 0)

Frame = np.ndarray


def compute_obstacle_distance_np(x: float, y: float, dx: float, dy: float, obstacles: np.ndarray) -> float:
    eps = 1e-6
    distances = []
    for obs in obstacles:
        ox, oy, ow, oh = obs
        # x slabs
        if abs(dx) < eps:
            t_min_x, t_max_x = -float("inf"), float("inf")
        else:
            t1 = (ox - x) / dx
            t2 = ((ox + ow) - x) / dx
            t_min_x, t_max_x = min(t1, t2), max(t1, t2)
        # y slabs
        if abs(dy) < eps:
            t_min_y, t_max_y = -float("inf"), float("inf")
        else:
            t3 = (oy - x) / dy  # Note: use x? Actually, we need to compute based on y.
            # Correction: use y for y-slabs:
            t3 = (oy - y) / dy
            t4 = ((oy + oh) - y) / dy
            t_min_y, t_max_y = min(t3, t4), max(t3, t4)
        t_entry = max(t_min_x, t_min_y)
        t_exit = min(t_max_x, t_max_y)
        if t_exit >= t_entry and t_exit > 0:
            t_entry = t_entry if t_entry > 0 else float("inf")
            distances.append(t_entry)
    if distances:
        return min(distances)
    else:
        return float("inf")


def render_frame(state: EnvState, params: EnvParams, img_width: int = 600, img_height: int = 600) -> Frame:
    """Renders the current state including obstacles and lidar beams with collisions."""
    img = Image.new("RGB", (img_width, img_height), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    scale = min(img_width, img_height) / params.arena_size

    def world_to_pixels(x, y) -> Tuple[int, int]:
        px = int(float(x) * scale)
        py = int(img_height - float(y) * scale)
        return px, py

    def rotate_point(x, y, angle_rad) -> Tuple[float, float]:
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return x * cos_a - y * sin_a, x * sin_a + y * cos_a

    def get_polygon_pixels(
        local_corners: List[Tuple[float, float]], center_x: float, center_y: float, angle_rad: float
    ) -> List[Tuple[int, int]]:
        return [
            world_to_pixels(
                center_x + rotate_point(lx, ly, angle_rad)[0], center_y + rotate_point(lx, ly, angle_rad)[1]
            )
            for lx, ly in local_corners
        ]

    # --- Draw obstacles ---
    if state.obstacles is not None:
        obstacles = np.array(state.obstacles)
        for obs in obstacles:
            ox, oy, ow, oh = obs
            top_left = world_to_pixels(ox, oy + oh)
            bottom_right = world_to_pixels(ox + ow, oy)
            draw.rectangle([top_left, bottom_right], fill=(200, 200, 200), outline=(0, 0, 0))

    # --- Draw Goal ---
    goal_px, goal_py = world_to_pixels(state.goal_x, state.goal_y)
    goal_radius_px = int(0.02 * min(img_width, img_height))
    goal_bbox = [goal_px - goal_radius_px, goal_py - goal_radius_px, goal_px + goal_radius_px, goal_py + goal_radius_px]
    draw.ellipse(goal_bbox, fill=GOAL_COLOR, width=1)

    # --- Draw Robot Chassis ---
    hl, hw = ROBOT_LENGTH / 2, ROBOT_WIDTH / 2
    chassis_local = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
    chassis_pixels = get_polygon_pixels(chassis_local, state.x, state.y, state.theta)
    draw.polygon(chassis_pixels, fill=ROBOT_CHASSIS_COLOR, width=1)

    # --- Draw Robot Wheels ---
    left_wheel_local = [
        (WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
        (-WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
        (-WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
        (WHEEL_LENGTH / 2, ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
    ]
    right_wheel_local = [
        (WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
        (-WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 + WHEEL_WIDTH / 2),
        (-WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
        (WHEEL_LENGTH / 2, -ROBOT_WIDTH / 2 - WHEEL_WIDTH / 2),
    ]
    left_wheel_pixels = get_polygon_pixels(left_wheel_local, state.x, state.y, state.theta)
    right_wheel_pixels = get_polygon_pixels(right_wheel_local, state.x, state.y, state.theta)
    draw.polygon(left_wheel_pixels, fill=ROBOT_WHEEL_COLOR, width=1)
    draw.polygon(right_wheel_pixels, fill=ROBOT_WHEEL_COLOR, width=1)

    # --- Draw Lidar Beams with collision detection ---
    num_beams = params.lidar_num_beams
    fov_rad = params.lidar_fov * np.pi / 180.0
    beam_offsets = np.linspace(-fov_rad / 2, fov_rad / 2, num_beams)
    beam_angles = state.theta + beam_offsets
    obstacles = np.array(state.obstacles) if state.obstacles is not None else np.empty((0, 4))

    for angle in beam_angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        obs_dist = compute_obstacle_distance_np(state.x, state.y, dx, dy, obstacles)
        d = min(obs_dist, obs_dist, params.lidar_max_distance)
        end_x = state.x + d * dx
        end_y = state.y + d * dy
        start_px = world_to_pixels(state.x, state.y)
        end_px = world_to_pixels(end_x, end_y)
        draw.line([start_px, end_px], fill=LIDAR_COLOR, width=1)

    # --- Draw Text Info ---
    draw.text((5, 5), f"T: {int(state.time)}", fill=TEXT_COLOR, font=font)
    draw.text((5, 20), f"R: {float(state.accumulated_reward):.2f}", fill=TEXT_COLOR, font=font)

    return np.array(img, dtype=np.uint8)


def save_gif(frames: Sequence[Frame], filename: str, duration_per_frame: float = 0.1):
    try:
        imageio.mimsave(filename, list(frames), duration=duration_per_frame, loop=0)
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
