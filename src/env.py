from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


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
    accumulated_reward: float = 0.0  # Accumulated reward during the episode


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters for the differential drive robot."""

    max_steps_in_episode: int = 200
    goal_tolerance: float = 0.1  # meters
    arena_size: float = 5.0  # meters (square arena from 0 to arena_size)
    dt: float = 0.1  # simulation timestep

    # Action parameters
    linear_speed: float = 1.0  # m/s for 'forward' action
    angular_speed: float = jnp.pi / 2  # rad/s for 'turn' actions

    # Reward parameters
    step_penalty: float = 0.01  # Small penalty for each step
    goal_reward: float = 10.0  # Large reward for reaching the goal


class NavigationEnv(environment.Environment[EnvState, EnvParams]):
    """
    Gymnax compatible environment for a differential drive robot navigating to a goal.

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

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Numeric,
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
                params.linear_speed * 0.5,  # Small forward speed while turning left
                params.linear_speed,  # Full forward speed
                params.linear_speed * 0.5,  # Small forward speed while turning right
            ],
            dtype=jnp.float32,
        )

        # Define possible angular speeds based on action
        possible_omegas = jnp.array(
            [
                params.angular_speed,  # Turn left
                0.0,  # Go straight
                -params.angular_speed,  # Turn right
            ],
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
            terminal=done,
            accumulated_reward=state.accumulated_reward + reward,
        )

        # --- Get observation and info ---
        obs = self.get_obs(state)
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
        """Reset environment state with random start and goal positions."""
        # Split key for generating start and goal positions
        key, start_key, goal_key = jax.random.split(key, 3)

        # Generate random start position (x, y, theta) uniformly within the arena
        start_x, start_y = jax.random.uniform(
            start_key, shape=(2,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size
        )

        # Generate random start orientation
        key, theta_key = jax.random.split(key)
        start_theta = jax.random.uniform(theta_key, shape=(), dtype=jnp.float32, minval=-jnp.pi, maxval=jnp.pi)

        # Generate random goal position uniformly within the arena
        goal_x, goal_y = jax.random.uniform(
            goal_key, shape=(2,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size
        )

        # Create the initial state with the random start and goal
        state = EnvState(
            x=start_x,  # Random start x
            y=start_y,  # Random start y
            theta=start_theta,  # Random start theta
            goal_x=goal_x,  # Random goal x
            goal_y=goal_y,  # Random goal y
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = None) -> chex.Array:
        """Return observation from state."""
        # Observation: [x, y, cos(theta), sin(theta), goal_x, goal_y]
        return jnp.array(
            [state.x, state.y, jnp.cos(state.theta), jnp.sin(state.theta), state.goal_x, state.goal_y]
        ).astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check if goal was reached in the previous step OR if max steps exceeded
        time_exceeded = state.time >= params.max_steps_in_episode
        # The state.terminal flag is set to True if goal_reached or time_exceeded was True in the *previous* step.
        done = jnp.logical_or(state.terminal, time_exceeded)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "NavigationEnv-v0"

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
