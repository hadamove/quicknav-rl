import math
import time

import jax
import jax.numpy as jnp
import numpy as np  # For visualization bridge
from flax import struct  # Or use typing.NamedTuple if flax isn't available
from jax import random

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    print("Pygame not found. Visualization will be disabled.")
    PYGAME_AVAILABLE = False


# --- Environment Configuration ---
@struct.dataclass
class EnvConfig:
    max_steps_in_episode: int = 200
    dt: float = 0.1  # Time step duration
    goal_threshold: float = 0.1  # Target radius around goal
    max_wheel_vel: float = 1.0  # Max velocity for each wheel
    dist_reward_scale: float = 1.0  # Scale for distance-based reward
    goal_bonus: float = 100.0  # Bonus for reaching the goal
    step_penalty: float = 0.1  # Penalty per step
    robot_radius: float = 0.15  # For viz
    wheel_base: float = 0.3  # Distance between wheels (L), for viz/realism scale

    # --- World bounds (optional, for randomization or viz) ---
    world_size: float = 5.0

    # --- Action/Observation space dims ---
    action_dim: int = 2
    # Obs: [x, y, cos(theta), sin(theta), gx, gy]
    observation_dim: int = 6


# --- Environment State ---
@struct.dataclass
class EnvState:
    robot_pos: jnp.ndarray  # Shape (2,) - x, y
    robot_theta: jnp.ndarray  # Shape (,) - angle in radians
    goal_pos: jnp.ndarray  # Shape (2,) - gx, gy
    timestep: int
    key: random.PRNGKey  # JAX random key


class DiffDriveEnv:
    """
    A JAX-based Differential Drive Robot Environment
    """

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.config = config

    @property
    def observation_space_shape(self):
        return (self.config.observation_dim,)

    @property
    def action_space_shape(self):
        return (self.config.action_dim,)

    def reset(self, key: random.PRNGKey) -> tuple[EnvState, jnp.ndarray]:
        """
        Resets the environment to a starting state.

        Args:
            key: JAX PRNG key.

        Returns:
            A tuple containing the initial EnvState and the initial observation.
        """
        key, subkey1, subkey2 = random.split(key, 3)

        # Example: Fixed start and goal. Can be randomized using subkeys.
        start_pos = jnp.array([0.5, 0.5])
        start_theta = jnp.pi / 4  # Start facing towards goal typical quadrant 1
        goal_pos = jnp.array([self.config.world_size - 0.5, self.config.world_size - 0.5])

        # Example: Random start/goal (uncomment to use)
        # world_margin = 0.5
        # start_pos = random.uniform(
        #     subkey1, shape=(2,), minval=world_margin, maxval=self.config.world_size - world_margin
        # )
        # goal_pos = random.uniform(
        #     subkey2, shape=(2,), minval=world_margin, maxval=self.config.world_size - world_margin
        # )
        # start_theta = random.uniform(subkey1, shape=(), minval=-jnp.pi, maxval=jnp.pi) # Random orientation

        initial_state = EnvState(
            robot_pos=start_pos, robot_theta=start_theta, goal_pos=goal_pos, timestep=0, key=key  # Store remaining key
        )
        observation = self._get_observation(initial_state)
        return initial_state, observation

    def step(self, state: EnvState, action: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, float, bool, bool, dict]:
        """
        Performs one step in the environment.

        Args:
            state: Current environment state.
            action: Action chosen by the agent [left_wheel_vel, right_wheel_vel].

        Returns:
            A tuple containing:
                - next_state: The state after the step.
                - observation: Observation corresponding to next_state.
                - reward: Scalar reward received.
                - done: Boolean indicating if the goal is reached.
                - truncated: Boolean indicating if the episode timed out.
                - info: Dictionary with auxiliary information (e.g., distance).
        """
        # Clip action
        action = jnp.clip(action, -self.config.max_wheel_vel, self.config.max_wheel_vel)
        vl, vr = action[0], action[1]

        # --- Differential Drive Kinematics ---
        # Assuming wheel radius r=1 for simplicity, action is directly velocity
        # If using r, v = r*(vl+vr)/2, omega = r*(vr-vl)/L
        # Here, assume actions *are* scaled wheel contributions to v and omega
        # More standard:
        # v = (vl + vr) / 2.0
        # omega = (vr - vl) / self.config.wheel_base # Use wheel_base L if actions are wheel velocities

        # Simplified interpretation: action controls linear and angular velocity directly for simplicity
        # v = action[0] * self.config.max_wheel_vel # Max linear vel
        # omega = action[1] * (self.config.max_wheel_vel / self.config.robot_radius) # Max angular vel (example scaling)
        # Let's stick to wheel velocities as it's more standard for diff drive

        v = (vl + vr) / 2.0
        omega = (vr - vl) / self.config.wheel_base  # Use actual wheel base

        # --- Update State (Euler integration) ---
        new_x = state.robot_pos[0] + v * jnp.cos(state.robot_theta) * self.config.dt
        new_y = state.robot_pos[1] + v * jnp.sin(state.robot_theta) * self.config.dt
        new_theta = state.robot_theta + omega * self.config.dt

        # Wrap angle to [-pi, pi]
        new_theta = jnp.arctan2(jnp.sin(new_theta), jnp.cos(new_theta))

        # --- Boundary collision (optional - simple stop) ---
        # new_x = jnp.clip(new_x, 0.0, self.config.world_size)
        # new_y = jnp.clip(new_y, 0.0, self.config.world_size)

        next_robot_pos = jnp.array([new_x, new_y])
        next_timestep = state.timestep + 1

        # --- Calculate Reward ---
        current_dist = jnp.linalg.norm(state.robot_pos - state.goal_pos)
        next_dist = jnp.linalg.norm(next_robot_pos - state.goal_pos)

        # Reward for getting closer + step penalty
        reward_dist = (current_dist - next_dist) * self.config.dist_reward_scale
        reward = reward_dist - self.config.step_penalty

        # --- Check Termination Conditions ---
        goal_reached = next_dist < self.config.goal_threshold
        timed_out = next_timestep >= self.config.max_steps_in_episode

        # Add goal bonus
        # Use jax.lax.cond for conditional logic within jit-compiled functions
        reward = jax.lax.cond(
            goal_reached,
            lambda r: r + self.config.goal_bonus,  # Function if true
            lambda r: r,  # Function if false
            reward,  # Operand
        )

        done = goal_reached
        truncated = timed_out
        # Note: In JAX-based libraries (like Brax, Gymnax), often only a single 'done'
        # is returned (done | truncated), and truncation is flagged in 'info'.
        # Here we return both explicitly for clarity, similar to Gymnasium v0.26+.

        next_state = EnvState(
            robot_pos=next_robot_pos,
            robot_theta=new_theta,
            goal_pos=state.goal_pos,  # Goal stays the same
            timestep=next_timestep,
            key=state.key,  # Pass key along, split if stochastic transitions needed
        )

        observation = self._get_observation(next_state)
        info = {"distance_to_goal": next_dist, "goal_reached": goal_reached}

        return next_state, observation, reward, done, truncated, info

    def _get_observation(self, state: EnvState) -> jnp.ndarray:
        """Extracts the observation from the state."""
        # Observation: [x, y, cos(theta), sin(theta), gx, gy]
        return jnp.concatenate(
            [state.robot_pos, jnp.array([jnp.cos(state.robot_theta), jnp.sin(state.robot_theta)]), state.goal_pos]
        )


# --- Pygame Visualization ---
class Renderer:
    def __init__(self, config: EnvConfig, screen_size=600):
        if not PYGAME_AVAILABLE:
            print("Pygame not available, renderer cannot be initialized.")
            self.screen = None
            return

        pygame.init()
        pygame.display.set_caption("Differential Drive Robot")
        self.config = config
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.font = pygame.font.SysFont(None, 24)

        # Colors
        self.colors = {
            "background": (240, 240, 240),
            "robot": (50, 100, 200),
            "goal": (50, 200, 100),
            "robot_dir": (255, 50, 50),
            "info_text": (0, 0, 0),
        }

        # Coordinate transformation
        self.world_to_screen_scale = self.screen_size / self.config.world_size
        # Flip y-axis because pygame's (0,0) is top-left
        self.y_flip = lambda y: self.screen_size - y

    def _world_to_screen(self, pos: np.ndarray) -> tuple[int, int]:
        """Converts world coordinates (JAX/NumPy array) to screen pixels."""
        screen_x = int(pos[0] * self.world_to_screen_scale)
        screen_y = int(self.y_flip(pos[1] * self.world_to_screen_scale))
        return screen_x, screen_y

    def draw(self, state: EnvState, reward: float = 0.0, done: bool = False, truncated: bool = False):
        """Renders the current environment state."""
        if self.screen is None:
            return

        # --- Convert JAX arrays to NumPy for Pygame ---
        # This is crucial as Pygame cannot handle JAX arrays directly.
        # Device retrieval happens here.
        robot_pos_np = np.array(state.robot_pos)
        robot_theta_np = float(state.robot_theta)  # Pygame needs float, not 0-dim array
        goal_pos_np = np.array(state.goal_pos)

        # --- Handle Pygame events (e.g., closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False  # Indicate closed

        # --- Drawing ---
        self.screen.fill(self.colors["background"])

        # Draw Goal
        goal_screen_pos = self._world_to_screen(goal_pos_np)
        goal_radius_screen = int(self.config.goal_threshold * self.world_to_screen_scale)
        pygame.draw.circle(self.screen, self.colors["goal"], goal_screen_pos, goal_radius_screen)
        pygame.draw.circle(self.screen, (0, 0, 0), goal_screen_pos, goal_radius_screen, 1)  # Outline

        # Draw Robot
        robot_screen_pos = self._world_to_screen(robot_pos_np)
        robot_radius_screen = int(self.config.robot_radius * self.world_to_screen_scale)
        pygame.draw.circle(self.screen, self.colors["robot"], robot_screen_pos, robot_radius_screen)
        pygame.draw.circle(self.screen, (0, 0, 0), robot_screen_pos, robot_radius_screen, 1)  # Outline

        # Draw Robot Orientation Line
        line_len = robot_radius_screen * 1.2
        end_x = robot_screen_pos[0] + line_len * math.cos(robot_theta_np)
        # Need to subtract sin component because screen y is flipped
        end_y = robot_screen_pos[1] - line_len * math.sin(robot_theta_np)
        pygame.draw.line(self.screen, self.colors["robot_dir"], robot_screen_pos, (int(end_x), int(end_y)), 3)

        # --- Display Info Text ---
        info_text = f"Step: {state.timestep} | Reward: {reward:.2f} | Done: {done} | Truncated: {truncated}"
        text_surface = self.font.render(info_text, True, self.colors["info_text"])
        self.screen.blit(text_surface, (10, 10))

        # --- Update Display ---
        pygame.display.flip()
        return True  # Indicate still open

    def close(self):
        """Closes the Pygame window."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# --- Example Usage ---
if __name__ == "__main__":
    config = EnvConfig(max_steps_in_episode=300)
    env = DiffDriveEnv(config)

    # JIT compile the step function for potential speedup
    # Note: reset might not benefit as much from jit if it involves significant
    #       random number generation logic that changes frequently, but step usually does.
    #       Also, first call will be slow due to compilation.
    step_jit = jax.jit(env.step)

    # Setup random key
    seed = 0
    key = random.PRNGKey(seed)

    # --- Simple Random Agent Loop ---
    print("Running random agent loop...")
    key, reset_key = random.split(key)
    state, obs = env.reset(reset_key)

    renderer = Renderer(config)  # Initialize visualization

    total_reward = 0.0
    running = True
    render_delay = 0.05  # Small delay for visualization

    for i in range(config.max_steps_in_episode * 2):  # Run a bit longer to see outcome
        if not running:
            break

        # Sample random action
        key, action_key = random.split(state.key)  # Use key from state for reproducibility
        # Generate action in range [-max_vel, max_vel]
        action = random.uniform(
            action_key, shape=(config.action_dim,), minval=-config.max_wheel_vel, maxval=config.max_wheel_vel
        )

        # Step the environment (using JIT compiled version)
        start_time = time.time()
        next_state, next_obs, reward, done, truncated, info = step_jit(state, action)
        jax.block_until_ready(next_state)  # Ensure computation finishes for timing
        step_time = time.time() - start_time

        # Update state
        state = next_state
        total_reward += reward

        # Render (outside JIT)
        if renderer.screen:
            running = renderer.draw(state, reward, done, truncated)
            time.sleep(render_delay)  # Control visualization speed

        # Print step info
        print(
            f"Step: {state.timestep}, Action: [{action[0]:.2f}, {action[1]:.2f}], Reward: {reward:.3f}, Dist: {info['distance_to_goal']:.3f}, Done: {done}, Trunc: {truncated}, StepTime: {step_time*1000:.2f}ms"
        )

        if done or truncated:
            print(f"Episode finished after {state.timestep} steps. Total Reward: {total_reward:.2f}")
            print(f"Reason: {'Goal Reached' if done else 'Time Limit'}")

            # Optional: Pause at the end or reset immediately
            if renderer.screen:
                print("Episode End. Pausing for 2 seconds...")
                time.sleep(2.0)

            # Reset for a new episode
            key, reset_key = random.split(key)
            state, obs = env.reset(reset_key)
            total_reward = 0.0
            print("\n--- New Episode ---")
            if not running and renderer.screen is None:  # If window closed during pause
                break

    renderer.close()  # Clean up pygame window
    print("Simulation finished.")
