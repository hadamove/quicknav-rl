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
TEXT_COLOR = (0, 0, 0)

Frame = np.ndarray
"""Type alias for a frame, which is a NumPy array representing an image."""


def render_frame(state: EnvState, params: EnvParams, img_width: int = 600, img_height: int = 600) -> Frame:
    """Renders the current state of the environment into a NumPy array image using Pillow."""

    # Create a blank white image
    img = Image.new("RGB", (img_width, img_height), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)  # Use default font

    # --- Calculate scaling and transformation ---
    scale = min(img_width, img_height) / params.arena_size

    def world_to_pixels(x, y) -> Tuple[int, int]:
        """Converts world coordinates to integer pixel coordinates."""
        x_f, y_f = float(x), float(y)  # Ensure input is float
        px = int(x_f * scale)
        py = int(img_height - y_f * scale)  # Flip y-axis
        return px, py

    def rotate_point(x, y, angle_rad) -> Tuple[float, float]:
        """Rotates a point (x, y) counter-clockwise by angle_rad around the origin."""
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        return x_new, y_new

    def get_polygon_pixels(
        local_corners: List[Tuple[float, float]], center_x: float, center_y: float, angle_rad: float
    ) -> List[Tuple[int, int]]:
        """Rotates, translates, and converts local corners to pixel coordinates."""
        pixel_corners = []
        for lx, ly in local_corners:
            # Rotate around origin
            rx, ry = rotate_point(lx, ly, angle_rad)
            # Translate to world center
            wx, wy = center_x + rx, center_y + ry
            # Convert to pixels
            px, py = world_to_pixels(wx, wy)
            pixel_corners.append((px, py))
        return pixel_corners

    # --- Define Local Robot Component Coordinates (around 0,0 with theta=0) ---
    hl, hw = ROBOT_LENGTH / 2, ROBOT_WIDTH / 2
    whl, whw = WHEEL_LENGTH / 2, WHEEL_WIDTH / 2

    chassis_local_corners = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]

    # Wheels positioned along the sides, centered longitudinally
    left_wheel_local_corners = [(whl, hw + whw), (-whl, hw + whw), (-whl, hw - whw), (whl, hw - whw)]
    right_wheel_local_corners = [(whl, -hw + whw), (-whl, -hw + whw), (-whl, -hw - whw), (whl, -hw - whw)]

    # --- Draw Goal ---
    goal_px, goal_py = world_to_pixels(state.goal_x, state.goal_y)
    # Calculate goal radius in pixels based on world size? Let's use a fixed pixel size for now.
    goal_radius_px = int(0.02 * min(img_width, img_height))  # Example: 2% of image size
    goal_bbox = [goal_px - goal_radius_px, goal_py - goal_radius_px, goal_px + goal_radius_px, goal_py + goal_radius_px]
    draw.ellipse(goal_bbox, fill=GOAL_COLOR, width=1)

    # --- Draw Robot Chassis ---
    chassis_pixel_corners = get_polygon_pixels(chassis_local_corners, state.x, state.y, state.theta)
    draw.polygon(chassis_pixel_corners, fill=ROBOT_CHASSIS_COLOR, width=1)

    # --- Draw Robot Wheels ---
    left_wheel_pixel_corners = get_polygon_pixels(left_wheel_local_corners, state.x, state.y, state.theta)
    draw.polygon(left_wheel_pixel_corners, fill=ROBOT_WHEEL_COLOR, width=1)

    right_wheel_pixel_corners = get_polygon_pixels(right_wheel_local_corners, state.x, state.y, state.theta)
    draw.polygon(right_wheel_pixel_corners, fill=ROBOT_WHEEL_COLOR, width=1)

    # --- Display Text Info (Timestamp & Reward) ---
    time_text = f"T: {int(state.time)}"
    reward_text = f"R: {float(state.accumulated_reward):.2f}"
    draw.text((5, 5), time_text, fill=TEXT_COLOR, font=font)
    draw.text((5, 20), reward_text, fill=TEXT_COLOR, font=font)

    # Convert Pillow Image to NumPy array (HxWxRGB format)
    frame = np.array(img, dtype=np.uint8)  # Ensure uint8 dtype

    return frame


def save_gif(frames: Sequence[Frame], filename: str, duration_per_frame: float = 100):
    """Saves a list of NumPy array frames as a GIF.

    Args:
        frames: List of frames (HxWx3 NumPy arrays).
        filename: Path to save the GIF file.
        duration_per_frame: Duration (in milliseconds) for each frame.
    """
    duration_per_frame_sec = duration_per_frame / 1000.0
    try:
        imageio.mimsave(
            filename,
            list(frames),
            duration=duration_per_frame_sec,
            loop=0,  # loop=0 means loop indefinitely
        )
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Ensure imageio is installed correctly.")
