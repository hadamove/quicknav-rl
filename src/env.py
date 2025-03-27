from typing import Any, Dict, Tuple

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
    theta: float  # Robot orientation (radians)
    goal_x: float  # Goal x position
    goal_y: float  # Goal y position
    time: int  # Current timestep
    terminal: bool  # Episode termination flag
    accumulated_reward: float = 0.0  # Total reward accumulated


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters for the differential drive robot."""

    max_steps_in_episode: int = 200
    goal_tolerance: float = 0.1  # meters
    arena_size: float = 5.0  # meters (arena is [0, arena_size] square)
    dt: float = 0.1  # simulation timestep

    # Differential drive parameters
    wheel_base: float = 0.5  # distance between wheels (meters)
    max_wheel_speed: float = 1.0  # maximum wheel speed (m/s)

    # Reward parameters
    step_penalty: float = 0.01  # penalty per step
    goal_reward: float = 10.0  # bonus for reaching the goal


class NavigationEnv(environment.Environment[EnvState, EnvParams]):
    """
    Gymnax environment for a differential drive robot navigating to a goal.

    ENVIRONMENT DESCRIPTION:
    - A differential drive robot operates in a square 2D arena.
    - The robot starts at a random position and must navigate to a random goal.
    - Observations include the robot's pose (x, y, cos(theta), sin(theta)) and the goal (gx, gy).
    - Actions are continuous: [v_left, v_right] (wheel speeds in m/s).
    - Dynamics follow:
          v     = (v_left + v_right) / 2
          omega = (v_right - v_left) / wheel_base
      updated via Euler integration.
    - Reward is the improvement in distance minus a step penalty, with a bonus upon reaching the goal.
    - Termination occurs when the goal is reached or max steps are exceeded.
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
        action: chex.Array,  # Continuous: [v_left, v_right]
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform a single timestep state transition."""
        # --- Extract and clip wheel speeds ---
        v_left = jnp.clip(action[0], -params.max_wheel_speed, params.max_wheel_speed)
        v_right = jnp.clip(action[1], -params.max_wheel_speed, params.max_wheel_speed)

        # --- Compute differential drive kinematics ---
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / params.wheel_base

        # --- Euler integration ---
        new_theta = state.theta + omega * params.dt
        new_theta = jnp.arctan2(jnp.sin(new_theta), jnp.cos(new_theta))  # Normalize to [-pi, pi]
        new_x = state.x + v * jnp.cos(state.theta) * params.dt
        new_y = state.y + v * jnp.sin(state.theta) * params.dt

        # --- Enforce arena boundaries ---
        new_x = jnp.clip(new_x, 0.0, params.arena_size)
        new_y = jnp.clip(new_y, 0.0, params.arena_size)

        # --- Reward calculation ---
        prev_dist = jnp.sqrt((state.x - state.goal_x) ** 2 + (state.y - state.goal_y) ** 2)
        new_dist = jnp.sqrt((new_x - state.goal_x) ** 2 + (new_y - state.goal_y) ** 2)
        reward = prev_dist - new_dist - params.step_penalty

        # --- Check termination ---
        goal_reached = new_dist <= params.goal_tolerance
        time_exceeded = (state.time + 1) >= params.max_steps_in_episode
        done = jnp.logical_or(goal_reached, time_exceeded)
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

        # --- Observation and additional info ---
        obs = self.get_obs(state)
        info = {"discount": self.discount(state, params)}

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def discount(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Return discount factor."""
        return jax.lax.select(state.terminal, 0.0, 1.0)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment with random start and goal positions."""
        key, start_key, goal_key = jax.random.split(key, 3)
        start_x, start_y = jax.random.uniform(
            start_key, shape=(2,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size
        )
        key, theta_key = jax.random.split(key)
        start_theta = jax.random.uniform(theta_key, shape=(), dtype=jnp.float32, minval=-jnp.pi, maxval=jnp.pi)
        goal_x, goal_y = jax.random.uniform(
            goal_key, shape=(2,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size
        )
        state = EnvState(
            x=start_x,
            y=start_y,
            theta=start_theta,
            goal_x=goal_x,
            goal_y=goal_y,
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation: [x, y, cos(theta), sin(theta), goal_x, goal_y]."""
        return jnp.array(
            [state.x, state.y, jnp.cos(state.theta), jnp.sin(state.theta), state.goal_x, state.goal_y],
            dtype=jnp.float32,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Determine if the state is terminal."""
        return jnp.logical_or(state.terminal, state.time >= params.max_steps_in_episode)

    @property
    def name(self) -> str:
        """Environment name."""
        return "DifferentialDriveEnv-v0"

    @property
    def num_actions(self) -> int:
        """Number of action dimensions (left and right wheel speeds)."""
        return 2

    def action_space(self, params: EnvParams) -> spaces.Box:
        """Action space: continuous wheel speeds for left and right wheels."""
        low = jnp.array([-params.max_wheel_speed, -params.max_wheel_speed], dtype=jnp.float32)
        high = jnp.array([params.max_wheel_speed, params.max_wheel_speed], dtype=jnp.float32)
        return spaces.Box(low, high, (2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: [x, y, cos(theta), sin(theta), goal_x, goal_y]."""
        low = jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=jnp.float32)
        high = jnp.array(
            [params.arena_size, params.arena_size, 1.0, 1.0, params.arena_size, params.arena_size],
            dtype=jnp.float32,
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
