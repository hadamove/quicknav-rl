from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from lidar import simulate_lidar
from utils import generate_obstacles, sample_valid_positions


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters for configuring the navigation environment.

    Defines all configurable aspects of the environment, including physical dimensions,
    robot characteristics, sensor properties, rewards, and episode parameters.
    """

    # Robot parameters
    wheel_base: float = 0.5  # Distance between wheels (meters)
    max_wheel_speed: float = 1.0  # Maximum speed of each wheel (m/s)
    robot_radius: float = 0.15  # Radius of the robot for collision detection (meters)
    dt: float = 0.1  # Simulation timestep (seconds)

    # Environment parameters
    arena_size: float = 5.0  # Width and height of the square arena (meters)
    num_obstacles: int = struct.field(pytree_node=False, default=5)  # Number of obstacles in the environment
    min_obstacle_size: float = 0.3  # Minimum width/height of obstacles (meters)
    max_obstacle_size: float = 1.0  # Maximum width/height of obstacles (meters)

    # Sensor parameters
    lidar_num_beams: int = struct.field(pytree_node=False, default=32)  # Number of lidar beams
    lidar_fov: float = 120.0  # Lidar field of view (degrees)
    lidar_max_distance: float = 5.0  # Maximum detection range of lidar (meters)

    # Reward parameters
    goal_tolerance: float = 0.1  # Distance threshold for reaching the goal (meters)
    step_penalty: float = 0.01  # Small penalty applied at each timestep to encourage efficiency
    collision_penalty: float = 5.0  # Penalty for colliding with obstacles
    goal_reward: float = 100.0  # Reward for reaching the goal

    # Episode parameters
    max_steps_in_episode: int = 200  # Maximum number of steps before episode terminates


@struct.dataclass
class EnvState(struct.PyTreeNode):
    """Environment state for the differential drive robot navigation task.

    Contains all relevant information about the current state of the environment,
    including robot pose, goal location, obstacles, and sensor readings.
    """

    # Robot state
    x: jnp.ndarray  # Robot x position (meters)
    y: jnp.ndarray  # Robot y position (meters)
    theta: jnp.ndarray  # Robot orientation (radians)

    # Goal state
    goal_x: jnp.ndarray  # Goal x position (meters)
    goal_y: jnp.ndarray  # Goal y position (meters)

    # Environment elements
    obstacles: jnp.ndarray  # Obstacle coordinates as [x, y, width, height] array

    # Sensor readings
    lidar_distances: jnp.ndarray = jnp.array([])  # Distance readings from lidar beams (meters)
    lidar_collision_types: jnp.ndarray = jnp.array([])  # Type of object each beam hit (0=none, 1=obstacle, 2=goal)

    # Episode state
    time: int = 0  # Current timestep in the episode
    terminal: jnp.ndarray = jnp.array(False)  # Whether the episode has terminated
    accumulated_reward: jnp.ndarray = jnp.array(0.0)  # Total reward collected so far


class NavigationEnv(environment.Environment):
    """Differential drive robot navigating to a goal with obstacles and lidar."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def default_params(self) -> environment.EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[chex.Array, int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Perform a single timestep state transition."""
        # 1. Physics update
        action = jnp.asarray(action, dtype=jnp.float32)
        v_left, v_right = jnp.clip(action, -params.max_wheel_speed, params.max_wheel_speed)
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / params.wheel_base

        # Update position and orientation
        new_theta = jnp.arctan2(jnp.sin(state.theta + omega * params.dt), jnp.cos(state.theta + omega * params.dt))
        dx, dy = v * jnp.cos(state.theta) * params.dt, v * jnp.sin(state.theta) * params.dt
        new_x = jnp.clip(state.x + dx, params.robot_radius, params.arena_size - params.robot_radius)
        new_y = jnp.clip(state.y + dy, params.robot_radius, params.arena_size - params.robot_radius)

        # 2. Collision detection
        # Compute distances to obstacle centers
        obs_center_x = state.obstacles[:, 0] + state.obstacles[:, 2] / 2
        obs_center_y = state.obstacles[:, 1] + state.obstacles[:, 3] / 2
        min_dim = jnp.minimum(state.obstacles[:, 2], state.obstacles[:, 3]) / 2

        # Check if any obstacle is too close
        dists = jnp.sqrt((new_x - obs_center_x) ** 2 + (new_y - obs_center_y) ** 2)
        collision = jnp.any(dists < (params.robot_radius + min_dim))

        # On collision: stay in place and get penalty
        new_x = jnp.where(collision, state.x, new_x)
        new_y = jnp.where(collision, state.y, new_y)

        # 3. Reward calculation
        prev_dist = jnp.sqrt((state.x - state.goal_x) ** 2 + (state.y - state.goal_y) ** 2)
        new_dist = jnp.sqrt((new_x - state.goal_x) ** 2 + (new_y - state.goal_y) ** 2)
        goal_reached = new_dist <= params.goal_tolerance

        # Compute reward components
        progress_reward = prev_dist - new_dist
        collision_reward = jnp.where(collision, -params.collision_penalty, 0.0)
        goal_reward = jnp.where(goal_reached, params.goal_reward, 0.0)
        step_penalty = -params.step_penalty

        reward = progress_reward + collision_reward + goal_reward + step_penalty

        # 4. Terminal state check
        out_of_time = state.time + 1 >= params.max_steps_in_episode
        done = jnp.logical_or(goal_reached, out_of_time)

        # 5. Lidar simulation
        lidar_distances, collision_types = simulate_lidar(
            new_x, new_y, new_theta, state.obstacles, state.goal_x, state.goal_y, params
        )

        # 6. Update state
        new_state = EnvState(
            x=new_x,
            y=new_y,
            theta=new_theta,
            goal_x=state.goal_x,
            goal_y=state.goal_y,
            obstacles=state.obstacles,
            time=state.time + 1,
            terminal=done,
            accumulated_reward=state.accumulated_reward + reward,
            lidar_distances=lidar_distances,
            lidar_collision_types=collision_types,
        )

        # Return observations, state, reward, done, info
        obs = self._get_obs(new_state, params)
        info = {"discount": jnp.where(done, 0.0, 1.0)}

        return obs, new_state, reward, done, info

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment with random obstacles and valid start/goal positions."""
        # 1. Generate random obstacles
        key, obs_key = jax.random.split(key)
        obstacles = generate_obstacles(
            obs_key, params.arena_size, params.num_obstacles, params.min_obstacle_size, params.max_obstacle_size
        )

        # 2. Sample valid positions for robot and goal
        key, pos_key = jax.random.split(key)
        clearance = 2.0 * params.robot_radius
        robot_pos, goal_pos = sample_valid_positions(pos_key, obstacles, params.arena_size, clearance)

        # 3. Random orientation
        start_theta = jax.random.uniform(key, shape=(), minval=-jnp.pi, maxval=jnp.pi)

        # 4. Create initial state
        state = EnvState(
            x=robot_pos[0],
            y=robot_pos[1],
            theta=start_theta,
            goal_x=goal_pos[0],
            goal_y=goal_pos[1],
            obstacles=obstacles,
            time=0,
            terminal=jnp.array(False),
            accumulated_reward=jnp.array(0.0),
            lidar_distances=jnp.zeros(params.lidar_num_beams),
            lidar_collision_types=jnp.zeros(params.lidar_num_beams, dtype=jnp.int32),
        )

        # 5. Initialize lidar readings
        lidar_distances, collision_types = simulate_lidar(
            state.x, state.y, state.theta, obstacles, goal_pos[0], goal_pos[1], params
        )

        state = state.replace(
            lidar_distances=lidar_distances,
            lidar_collision_types=collision_types,
        )

        return self._get_obs(state, params), state

    def _get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Convert state to observation vector [x, y, sin(θ), cos(θ), goal_x, goal_y, dist, angle, lidar...]."""
        # Robot pose (x, y, sin, cos)
        pose = jnp.array([state.x, state.y, jnp.sin(state.theta), jnp.cos(state.theta)])

        # Goal position
        goal = jnp.array([state.goal_x, state.goal_y])

        # Goal in robot frame (distance and angle)
        dx, dy = state.goal_x - state.x, state.goal_y - state.y
        goal_distance = jnp.sqrt(dx**2 + dy**2)
        goal_angle = jnp.arctan2(dy, dx) - state.theta
        goal_angle = jnp.arctan2(jnp.sin(goal_angle), jnp.cos(goal_angle))  # Normalize to [-pi, pi]
        goal_relative = jnp.array([goal_distance, goal_angle])

        # Combine with lidar
        return jnp.concatenate([pose, goal, goal_relative, state.lidar_distances])

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space for the agent.

        Vector containing: robot position (x,y), orientation (sin,cos), goal position,
        goal relative coordinates (distance, angle), and lidar readings.
        """
        n_dims = 8 + params.lidar_num_beams

        # Lower bounds
        low = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -jnp.pi]),  # Robot, goal, relative
                jnp.zeros(params.lidar_num_beams),  # Lidar
            ]
        )

        # Upper bounds
        high = jnp.concatenate(
            [
                jnp.array(
                    [
                        params.arena_size,
                        params.arena_size,  # Robot position
                        1.0,
                        1.0,  # Sin/cos
                        params.arena_size,
                        params.arena_size,  # Goal position
                        jnp.sqrt(2) * params.arena_size,
                        jnp.pi,  # Goal distance/angle
                    ]
                ),
                jnp.ones(params.lidar_num_beams) * params.lidar_max_distance,  # Lidar
            ]
        )

        return spaces.Box(low, high, (n_dims,), jnp.float32)

    def action_space(self, params: EnvParams) -> spaces.Box:
        """Action space: [left_wheel_speed, right_wheel_speed].

        Controls differential drive via wheel speeds. Equal speeds move straight,
        different speeds turn, opposite speeds rotate in place.
        """
        low = jnp.array([-params.max_wheel_speed, -params.max_wheel_speed])
        high = jnp.array([params.max_wheel_speed, params.max_wheel_speed])
        return spaces.Box(low, high, (2,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """Internal state space of the environment.

        Contains robot position/orientation, goal position, and episode tracking.
        Does not include obstacles, lidar readings, or rewards (handled internally).
        """
        return spaces.Dict(
            {
                "x": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "y": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "theta": spaces.Box(-jnp.pi, jnp.pi, (), jnp.float32),
                "goal_x": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "goal_y": spaces.Box(0.0, params.arena_size, (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Terminal when goal is reached or max steps exceeded."""
        return jnp.logical_or(state.terminal, state.time >= params.max_steps_in_episode)

    @property
    def name(self) -> str:
        return "DifferentialDriveEnv-v0"

    @property
    def num_actions(self) -> int:
        return 2
