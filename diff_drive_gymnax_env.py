from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import pygame
from flax import struct
from gymnax.environments import environment, spaces

# Define constants for visualization
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
ROBOT_RADIUS = 10
GOAL_RADIUS = 12
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
        # Pygame visualization variables (initialized lazily)
        self.screen = None
        self.clock = None
        self.font = None

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
        # Example: Random start/goal (ensure they are not too close)
        # key_start, key_goal = jax.random.split(key)
        # start_x = jax.random.uniform(key_start, (), minval=0.1*params.arena_size, maxval=0.4*params.arena_size)
        # start_y = jax.random.uniform(key_start, (), minval=0.1*params.arena_size, maxval=0.4*params.arena_size)
        # start_theta = jax.random.uniform(key_start, (), minval=-jnp.pi, maxval=jnp.pi)
        # goal_x = jax.random.uniform(key_goal, (), minval=0.6*params.arena_size, maxval=0.9*params.arena_size)
        # goal_y = jax.random.uniform(key_goal, (), minval=0.6*params.arena_size, maxval=0.9*params.arena_size)

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
        # Using cos/sin of theta is often better than raw angle for NNs
        return jnp.array(
            [state.x, state.y, jnp.cos(state.theta), jnp.sin(state.theta), state.goal_x, state.goal_y]
        ).astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Episode ends if agent is terminated OR exceeds max_steps
        # NOTE: 'state.terminal' already incorporates goal_reached logic from step_env
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(state.terminal, done_steps)

    @property
    def name(self) -> str:
        """Environment name."""
        return "DiffDrive-v0"

    # @property
    # def num_actions(self) -> int:
    #     """Number of actions possible in environment."""
    #     return 3  # 0: Turn Left, 1: Move Forward, 2: Turn Right

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

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

    # --- Visualization using Pygame (Not JIT-compatible) ---

    def _init_render(self):
        """Initialize Pygame."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption(self.name)
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(None, 24)  # Basic font for text

    def render(self, state: EnvState, params: EnvParams):
        """Render the current state using Pygame."""
        self._init_render()

        # --- Calculate scaling from world coords to screen coords ---
        scale = min(SCREEN_WIDTH, SCREEN_HEIGHT) / params.arena_size

        # Function to transform world (x, y) to screen (sx, sy)
        # Pygame origin (0,0) is top-left
        def world_to_screen(x, y):
            sx = int(x * scale)
            sy = int(SCREEN_HEIGHT - y * scale)  # Flip y-axis
            return sx, sy

        # --- Clear screen ---
        self.screen.fill(WHITE)

        # --- Draw Arena Boundaries (optional) ---
        pygame.draw.rect(self.screen, BLACK, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 1)

        # --- Draw Goal ---
        goal_sx, goal_sy = world_to_screen(state.goal_x, state.goal_y)
        pygame.draw.circle(self.screen, GREEN, (goal_sx, goal_sy), GOAL_RADIUS)
        pygame.draw.circle(self.screen, BLACK, (goal_sx, goal_sy), GOAL_RADIUS, 1)  # Outline

        # --- Draw Robot ---
        robot_sx, robot_sy = world_to_screen(state.x, state.y)
        # Body
        pygame.draw.circle(self.screen, BLUE, (robot_sx, robot_sy), ROBOT_RADIUS)
        pygame.draw.circle(self.screen, BLACK, (robot_sx, robot_sy), ROBOT_RADIUS, 1)  # Outline
        # Orientation line
        line_len = ROBOT_RADIUS * 1.5
        end_x = state.x + line_len / scale * jnp.cos(state.theta)  # Back to world scale for calculation
        end_y = state.y + line_len / scale * jnp.sin(state.theta)
        end_sx, end_sy = world_to_screen(end_x, end_y)
        pygame.draw.line(self.screen, RED, (robot_sx, robot_sy), (end_sx, end_sy), 3)

        # --- Display Timestep (optional) ---
        time_text = self.font.render(f"Time: {state.time}", True, BLACK)
        self.screen.blit(time_text, (10, 10))

        # --- Update display ---
        pygame.display.flip()

        # --- Handle Pygame events (e.g., closing window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False  # Indicate rendering should stop

        # --- Control frame rate ---
        self.clock.tick(30)  # Limit to 30 FPS
        return True  # Indicate rendering continues

    def close(self):
        """Close Pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None


# --- Example Usage ---
if __name__ == "__main__":
    env = DiffDriveEnv()
    params = env.default_params
    key = jax.random.PRNGKey(0)

    # Reset environment
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params)

    print("Initial State:", state)
    print("Initial Obs:", obs)
    print("Observation Space:", env.observation_space(params))
    print("Action Space:", env.action_space(params))

    # Simple interaction loop with rendering
    ep_reward = 0
    rendering_active = True
    try:
        for step in range(params.max_steps_in_episode * 2):  # Run a bit longer to see termination
            if not rendering_active:
                break

            # Render current state
            rendering_active = env.render(state, params)
            if not rendering_active:
                print("Rendering window closed by user.")
                break

            # Sample a random action (replace with agent policy)
            key, action_key = jax.random.split(key)
            # action = env.action_space(params).sample(action_key)
            # Simple policy: turn towards goal, then move forward
            # dx = state.goal_x - state.x
            # dy = state.goal_y - state.y
            # goal_angle = jnp.arctan2(dy, dx)
            # angle_diff = goal_angle - state.theta
            # # Normalize angle diff to [-pi, pi]
            # angle_diff = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff))

            # if abs(angle_diff) > 0.2:  # If not aligned, turn
            #     action = 0 if angle_diff > 0 else 2  # 0=Left, 2=Right
            # else:  # If aligned, move forward
            #     action = 1

            # Do completely random action
            action = jax.random.randint(action_key, (), 0, env.num_actions)

            # Step environment
            key, step_key = jax.random.split(key)
            next_obs, next_state, reward, done, info = env.step(step_key, state, action, params)

            print(f"Step: {state.time}, Action: {action}, Reward: {reward:.3f}, Done: {done}")
            # print(f"  State: x={next_state.x:.2f}, y={next_state.y:.2f}, th={next_state.theta:.2f}")

            ep_reward += reward
            state = next_state  # Prepare for next step

            if done:
                print(f"Episode finished after {state.time} steps. Total Reward: {ep_reward:.3f}")
                # Render the final state
                rendering_active = env.render(state, params)
                # Optional pause at the end
                if rendering_active:
                    pygame.time.wait(2000)
                break

    finally:
        env.close()  # Ensure Pygame resources are released
