# Remove pygame import
# import pygame
from typing import Sequence

import imageio
import numpy as np
from numpy.typing import NDArray  # Import NDArray for more specific NumPy type hinting
from PIL import Image, ImageDraw, ImageFont

from env import EnvParams, EnvState

# Constants for visualization
ROBOT_RADIUS_PX = 10  # Robot radius in pixels on the image
GOAL_RADIUS_PX = 12  # Goal radius in pixels on the image

# Rendering colors
BACKGROUND_COLOR = (255, 255, 255)
ROBOT_COLOR = (0, 0, 255)
GOAL_COLOR = (0, 255, 0)
ORIENTATION_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 0, 0)

Frame = NDArray[np.uint8]
"""Type alias for a frame, which is a NumPy array representing an image."""


def render_frame(state: EnvState, params: EnvParams, img_width: int = 600, img_height: int = 600) -> Frame:
    """Renders the current state of the environment into a NumPy array image using Pillow."""

    # Create a blank white image
    img = Image.new("RGB", (img_width, img_height), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # --- Calculate scaling and transformation ---
    scale = min(img_width, img_height) / params.arena_size

    # Function to transform world (x, y) to image pixel coordinates (px, py)
    def world_to_pixels(x, y):
        # Convert JAX arrays/scalars to standard Python floats first
        x_f, y_f = float(x), float(y)
        px = int(x_f * scale)
        py = int(img_height - y_f * scale)  # Flip y-axis
        # Clamp to image bounds just in case
        px = max(0, min(img_width - 1, px))
        py = max(0, min(img_height - 1, py))
        return px, py

    # --- Draw Arena Boundaries (optional) ---
    draw.rectangle([0, 0, img_width - 1, img_height - 1], width=1)

    # --- Draw Goal ---
    goal_px, goal_py = world_to_pixels(state.goal_x, state.goal_y)
    goal_bbox = [
        goal_px - GOAL_RADIUS_PX,
        goal_py - GOAL_RADIUS_PX,
        goal_px + GOAL_RADIUS_PX,
        goal_py + GOAL_RADIUS_PX,
    ]
    draw.ellipse(goal_bbox, fill=GOAL_COLOR, width=1)

    # --- Draw Robot ---
    robot_px, robot_py = world_to_pixels(state.x, state.y)
    robot_bbox = [
        robot_px - ROBOT_RADIUS_PX,
        robot_py - ROBOT_RADIUS_PX,
        robot_px + ROBOT_RADIUS_PX,
        robot_py + ROBOT_RADIUS_PX,
    ]
    draw.ellipse(robot_bbox, fill=ROBOT_COLOR, width=1)

    # --- Draw Orientation line ---
    line_len_world = (ROBOT_RADIUS_PX * 1.5) / scale  # Length in world units

    # Convert state elements to float for numpy trig functions
    state_x_f, state_y_f, state_theta_f = float(state.x), float(state.y), float(state.theta)
    end_x_world = state_x_f + line_len_world * np.cos(state_theta_f)
    end_y_world = state_y_f + line_len_world * np.sin(state_theta_f)
    end_px, end_py = world_to_pixels(end_x_world, end_y_world)
    draw.line([robot_px, robot_py, end_px, end_py], fill=ORIENTATION_COLOR, width=2)

    time_text = f"T: {int(state.time)}"  # Use int() just in case
    draw.text((5, 5), time_text, fill=TEXT_COLOR, font=font)

    # Convert Pillow Image to NumPy array (HxWxRGB format)
    frame = np.array(img)

    return frame


def save_gif(frames: Sequence[Frame], filename: str, duration_per_frame: float = 100):
    """Saves a list of NumPy array frames as a GIF.

    Args:
        frames: List of frames (HxWx3 NumPy arrays).
        filename: Path to save the GIF file.
        duration: Duration (in milliseconds) for each frame.
    """
    duration_per_frame_sec = duration_per_frame / 1000.0
    try:
        imageio.mimsave(
            filename,
            list(frames),
            duration=duration_per_frame_sec,
            loop=0,
        )
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Ensure imageio is installed correctly.")
