# Remove pygame import
# import pygame
from typing import Any, Dict, List, Optional, Tuple

import chex
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

# Import Pillow and imageio
from PIL import Image, ImageDraw, ImageFont

# TODO: move this to utils/vis
# Define constants for visualization
IMG_WIDTH = 600
IMG_HEIGHT = 600
ROBOT_RADIUS_PX = 10  # Robot radius in pixels on the image
GOAL_RADIUS_PX = 12  # Goal radius in pixels on the image

# Define colors as RGB tuples
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


@struct.dataclass
class EnvState(environment.EnvState):
    """Environment state for the differential drive robot."""

    x: float  # Robot x position
    y: float  # Robot y position
    theta: float  # Robot orientation (angle in radians)
    goal_x: float  # Goal x position
    goal_y: float  # Goal y position
    time: int  # Current timestep
    terminal: bool  # Flag indicating if the episode ended


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters for the differential drive robot."""

    max_steps_in_episode: int = 200
    goal_tolerance: float = 0.1  # meters
    arena_size: float = 5.0  # meters (square arena from 0 to arena_size)
    dt: float = 0.1  # simulation timestep
    # Simplified action parameters (can be adjusted)
    linear_speed: float = 1.0  # m/s for 'forward' action
    angular_speed: float = jnp.pi / 4  # rad/s for 'turn' actions
    step_penalty: float = 0.01  # Small penalty for each step
    goal_reward: float = 10.0  # Large reward for reaching the goal


class DiffDriveEnv(environment.Environment[EnvState, EnvParams]):
    """
    JAX compatible environment for a differential drive robot navigating to a goal.

    ENVIRONMENT DESCRIPTION:
    - A simple two-wheeled robot operates in a square 2D arena.
    - The robot starts at a predefined position A and must navigate to a goal B.
    - Observations include the robot's current pose (x, y, theta) and the goal position (gx, gy).
    - Actions are discrete: 0: Turn Left, 1: Move Forward, 2: Turn Right.
    - Reward is dense, based on the change in distance to the goal, plus a large bonus for reaching the goal and a small step penalty.
    - Termination occurs when the goal is reached or the maximum number of steps is exceeded.
    """

    def __init__(self):
        super().__init__()
        # No visualization instance variables needed

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Numeric,  # Typically an int for Discrete action space
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        # --- Calculate robot motion based on action ---
        # Action 0: Turn Left
        # Action 1: Move Forward
        # Action 2: Turn Right

        # Define possible linear speeds based on action
        possible_vs = jnp.array(
            [
                params.linear_speed * 0.2,  # Small forward speed while turning left
                params.linear_speed,  # Full forward speed
                params.linear_speed * 0.2,  # Small forward speed while turning right
            ],
            dtype=jnp.float32,
        )

        # Define possible angular speeds based on action
        possible_omegas = jnp.array(
            [params.angular_speed, 0.0, -params.angular_speed],  # Turn left  # Go straight  # Turn right
            dtype=jnp.float32,
        )

        # Select v and omega based on the action index
        # Ensure action is treated as an integer index if it's not already
        action_idx = action.astype(jnp.int32)
        v = possible_vs[action_idx]
        omega = possible_omegas[action_idx]

        # --- Update robot state using Euler integration ---
        prev_x, prev_y = state.x, state.y
        new_theta = state.theta + omega * params.dt
        # Normalize angle to [-pi, pi]
        new_theta = jnp.arctan2(jnp.sin(new_theta), jnp.cos(new_theta))

        new_x = state.x + v * jnp.cos(state.theta) * params.dt
        new_y = state.y + v * jnp.sin(state.theta) * params.dt

        # --- Boundary conditions (clamp within arena) ---
        new_x = jnp.clip(new_x, 0.0, params.arena_size)
        new_y = jnp.clip(new_y, 0.0, params.arena_size)

        # --- Calculate reward ---
        prev_dist = jnp.sqrt((prev_x - state.goal_x) ** 2 + (prev_y - state.goal_y) ** 2)
        new_dist = jnp.sqrt((new_x - state.goal_x) ** 2 + (new_y - state.goal_y) ** 2)

        # Reward is improvement in distance minus step penalty
        reward = prev_dist - new_dist - params.step_penalty

        # --- Check for termination ---
        goal_reached = new_dist <= params.goal_tolerance
        time_exceeded = (state.time + 1) >= params.max_steps_in_episode
        done = jnp.logical_or(goal_reached, time_exceeded)

        # Add large bonus if goal reached
        reward = jax.lax.select(goal_reached, reward + params.goal_reward, reward)

        # --- Update state ---
        state = state.replace(
            x=new_x,
            y=new_y,
            theta=new_theta,
            time=state.time + 1,
            terminal=done,  # Store termination status (used by is_terminal)
        )

        # --- Get observation and info ---
        obs = self.get_obs(state)
        # The discount is often used in RL algorithms, gymnax includes it in info
        info = {"discount": self.discount(state, params)}

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    # Make sure discount method is defined (usually 1.0 unless goal reached)
    def discount(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Return discount rate."""
        return jax.lax.select(state.terminal, 0.0, 1.0)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment state."""
        # For simplicity, fixed start and goal. Can be randomized using key.
        start_x = 0.1 * params.arena_size
        start_y = 0.1 * params.arena_size
        start_theta = jnp.pi / 4  # Start facing towards goal approx
        goal_x = 0.9 * params.arena_size
        goal_y = 0.9 * params.arena_size

        state = EnvState(x=start_x, y=start_y, theta=start_theta, goal_x=goal_x, goal_y=goal_y, time=0, terminal=False)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = None) -> chex.Array:
        """Return observation from state."""
        # Observation: [x, y, cos(theta), sin(theta), goal_x, goal_y]
        return jnp.array(
            [state.x, state.y, jnp.cos(state.theta), jnp.sin(state.theta), state.goal_x, state.goal_y]
        ).astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(state.terminal, done_steps)

    @property
    def name(self) -> str:
        """Environment name."""
        return "DiffDrive-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3  # 0: Turn Left, 1: Move Forward, 2: Turn Right

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # obs: [x, y, cos(theta), sin(theta), goal_x, goal_y]
        low = jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=jnp.float32)
        high = jnp.array(
            [params.arena_size, params.arena_size, 1.0, 1.0, params.arena_size, params.arena_size], dtype=jnp.float32
        )
        return spaces.Box(low, high, (6,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "y": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "theta": spaces.Box(-jnp.pi, jnp.pi, (), jnp.float32),
                "goal_x": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "goal_y": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),  # Boolean flag
            }
        )

    # --- START: GIF Rendering Methods using Pillow ---
    # TODO: move this to utils/vis

    Frame = np.ndarray  # Type alias for frames (images)

    @staticmethod
    def render_frame(state: EnvState, params: EnvParams) -> Frame:
        """Renders the current state into a NumPy array image using Pillow."""

        # Create a blank white image
        img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=WHITE)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # --- Calculate scaling and transformation ---
        scale = min(IMG_WIDTH, IMG_HEIGHT) / params.arena_size

        # Function to transform world (x, y) to image pixel coordinates (px, py)
        def world_to_pixels(x, y):
            # Convert JAX arrays/scalars to standard Python floats first
            x_f, y_f = float(x), float(y)
            px = int(x_f * scale)
            py = int(IMG_HEIGHT - y_f * scale)  # Flip y-axis
            # Clamp to image bounds just in case
            px = max(0, min(IMG_WIDTH - 1, px))
            py = max(0, min(IMG_HEIGHT - 1, py))
            return px, py

        # --- Draw Arena Boundaries (optional) ---
        draw.rectangle([0, 0, IMG_WIDTH - 1, IMG_HEIGHT - 1], outline=BLACK, width=1)

        # --- Draw Goal ---
        goal_px, goal_py = world_to_pixels(state.goal_x, state.goal_y)
        goal_bbox = [
            goal_px - GOAL_RADIUS_PX,
            goal_py - GOAL_RADIUS_PX,
            goal_px + GOAL_RADIUS_PX,
            goal_py + GOAL_RADIUS_PX,
        ]
        draw.ellipse(goal_bbox, fill=GREEN, outline=BLACK, width=1)

        # --- Draw Robot ---
        robot_px, robot_py = world_to_pixels(state.x, state.y)
        robot_bbox = [
            robot_px - ROBOT_RADIUS_PX,
            robot_py - ROBOT_RADIUS_PX,
            robot_px + ROBOT_RADIUS_PX,
            robot_py + ROBOT_RADIUS_PX,
        ]
        draw.ellipse(robot_bbox, fill=BLUE, outline=BLACK, width=1)

        # --- Draw Orientation line ---
        line_len_world = (ROBOT_RADIUS_PX * 1.5) / scale  # Length in world units

        # Convert state elements to float for numpy trig functions
        state_x_f, state_y_f, state_theta_f = float(state.x), float(state.y), float(state.theta)
        end_x_world = state_x_f + line_len_world * np.cos(state_theta_f)
        end_y_world = state_y_f + line_len_world * np.sin(state_theta_f)
        end_px, end_py = world_to_pixels(end_x_world, end_y_world)
        draw.line([robot_px, robot_py, end_px, end_py], fill=RED, width=2)

        time_text = f"T: {int(state.time)}"  # Use int() just in case
        draw.text((5, 5), time_text, fill=BLACK, font=font)

        # Convert Pillow Image to NumPy array (HxWxRGB format)
        frame = np.array(img)

        return frame

    @staticmethod
    def save_gif(frames: List[Frame], filename: str, duration_per_frame: float = 100):
        # TODO: move this to utils
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
                frames,
                duration=duration_per_frame_sec,
                loop=0,
            )
            print(f"GIF saved to {filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Ensure imageio is installed correctly.")
