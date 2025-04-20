"""JAX environment for differential drive robot navigation with lidar sensing."""

from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from geometry import point_to_rectangle_distance
from lidar import Collision, simulate_lidar
from rooms import RoomParams, sample_position


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters for configuring the navigation environment.

    Defines all configurable aspects of the environment, including physical dimensions,
    robot characteristics, sensor properties, rewards, and episode parameters.
    """

    # Robot parameters
    wheel_base: float = 0.3
    """Distance between wheels (meters)"""
    max_wheel_speed: float = 1.0
    """Maximum speed of each wheel (m/s)"""
    robot_radius: float = 0.1
    """Radius of the robot for collision detection (meters)"""
    dt: float = 0.1
    """Simulation timestep (seconds)"""

    # Environment parameters
    rooms: RoomParams = struct.field(pytree_node=False, default=RoomParams())
    """Parameters for pre-generated rooms"""

    # Pre-generated rooms
    obstacles: jnp.ndarray = jnp.zeros((0, 0, 4))
    """Obstacle arrays [num_rooms, max_obstacles, 4]"""
    free_positions: jnp.ndarray = jnp.zeros((0, 0, 2))
    """Free positions [num_rooms, max_positions, 2]"""

    # Sensor parameters
    lidar_num_beams: int = struct.field(pytree_node=False, default=16)
    """Number of lidar beams"""
    lidar_fov: float = 120.0
    """Lidar field of view (degrees)"""
    lidar_max_distance: float = 3.0
    """Maximum detection range of lidar (meters)"""

    # Reward parameters
    goal_tolerance: float = 0.1
    """Distance threshold for reaching the goal (meters)"""
    step_penalty: float = 0.01
    """Small penalty applied at each timestep to encourage efficiency"""
    collision_penalty: float = 1.0
    """Penalty for colliding with obstacles"""
    goal_reward: float = 100.0
    """Reward for reaching the goal"""

    # Episode parameters
    max_steps_in_episode: int = 300
    """Maximum number of steps before episode terminates"""


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
    room_idx: jnp.ndarray  # Index of the currently used pre-generated room

    # Sensor readings
    lidar_distances: jnp.ndarray  # Distance readings from lidar beams (meters)
    lidar_collision_types: jnp.ndarray  # Type of object each beam hit (0=none, 1=obstacle, 2=goal)

    # Episode state
    steps: int  # Current timestep in the episode
    episode_done: jnp.ndarray  # Whether the episode has terminated
    accumulated_reward: jnp.ndarray  # Total reward collected so far


class NavigationEnv(environment.Environment):
    """Differential drive robot navigating to a goal with obstacles and lidar."""

    def __init__(self) -> None:
        super().__init__()
        # No need to store cached rooms in the instance anymore

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
        new_x = jnp.clip(state.x + dx, params.robot_radius, params.rooms.size - params.robot_radius)
        new_y = jnp.clip(state.y + dy, params.robot_radius, params.rooms.size - params.robot_radius)

        # 2. Collision detection
        # Calculate distance from robot to each obstacle
        robot_pos = jnp.array([new_x, new_y])
        distances = jax.vmap(point_to_rectangle_distance, in_axes=(None, 0))(robot_pos, state.obstacles)
        collision = jnp.any(distances < params.robot_radius)

        # On collision: stay in place and get penalty
        new_x = jnp.where(collision, state.x, new_x)
        new_y = jnp.where(collision, state.y, new_y)

        # 3. Reward calculation
        reward, goal_reached = self._calculate_reward(state, new_x, new_y, collision, params)

        # 4. Terminal state check
        out_of_time = state.steps + 1 >= params.max_steps_in_episode
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
            room_idx=state.room_idx,
            steps=state.steps + 1,
            episode_done=done,
            accumulated_reward=state.accumulated_reward + reward,
            lidar_distances=lidar_distances,
            lidar_collision_types=collision_types,
        )

        # Return observations, state, reward, done, info
        obs = self._get_obs(new_state, params)
        info = {"discount": jnp.where(done, 0.0, 1.0)}

        return obs, new_state, reward, done, info

    def _calculate_reward(
        self,
        state: EnvState,
        new_x: jnp.ndarray,
        new_y: jnp.ndarray,
        collision: jnp.ndarray,
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate reward and check if goal is reached."""
        # Calculate distance to goal before and after movement
        prev_dist = jnp.sqrt((state.x - state.goal_x) ** 2 + (state.y - state.goal_y) ** 2)
        new_dist = jnp.sqrt((new_x - state.goal_x) ** 2 + (new_y - state.goal_y) ** 2)
        goal_reached = new_dist <= params.goal_tolerance

        # Compute reward components
        progress_reward = prev_dist - new_dist
        collision_reward = jnp.where(collision, -params.collision_penalty, 0.0)
        goal_reward = jnp.where(goal_reached, params.goal_reward, 0.0)
        step_penalty = -params.step_penalty

        total_reward = progress_reward + collision_reward + goal_reward + step_penalty

        return total_reward, goal_reached

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling a pre-generated room layout."""
        key, room_key, pos_key, angle_key = jax.random.split(key, 4)

        # Sample a random room index
        room_idx = jax.random.randint(room_key, (), 0, params.rooms.num_rooms)

        # Get the obstacles and free positions for this room from params
        obstacles = params.obstacles[room_idx]
        free_positions = params.free_positions[room_idx]

        # Sample positions for robot and goal separately
        key_start, key_goal = jax.random.split(pos_key)
        robot_pos = sample_position(key_start, free_positions)
        goal_pos = sample_position(key_goal, free_positions)

        # Randomly initialize robot orientation
        robot_angle = jax.random.uniform(angle_key, minval=0, maxval=2 * jnp.pi)

        # Create initial state
        state = EnvState(
            x=robot_pos[0],
            y=robot_pos[1],
            theta=robot_angle,
            goal_x=goal_pos[0],
            goal_y=goal_pos[1],
            steps=0,
            episode_done=jnp.array(False),
            room_idx=room_idx,
            obstacles=obstacles,
            lidar_distances=jnp.zeros(params.lidar_num_beams),
            lidar_collision_types=jnp.zeros(params.lidar_num_beams, dtype=jnp.int32),
            accumulated_reward=jnp.array(0.0),
        )

        # Get initial observation
        obs = self._get_obs(state, params)

        return obs, state

    def _get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Convert state to observation vector."""
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

        # Convert collision types to goal flag (1 for goal, 0 for wall/obstacle)
        lidar_goal = (state.lidar_collision_types == Collision.Goal).astype(jnp.float32)

        # Combine robot state, goal, and sensor readings
        return jnp.concatenate([pose, goal, goal_relative, state.lidar_distances, lidar_goal])

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space for the agent.

        Vector containing: robot position (x,y), orientation (sin,cos), goal position,
        goal relative coordinates (distance, angle), lidar distances, and goal flags.
        """
        # Total dimensions: 8 base + lidar distances + goal flags
        n_dims = 8 + params.lidar_num_beams * 2

        # Lower bounds
        low = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -jnp.pi]),  # Robot, goal, relative
                jnp.zeros(params.lidar_num_beams),  # Lidar distances
                jnp.zeros(params.lidar_num_beams),  # Goal flags
            ]
        )

        # Upper bounds
        high = jnp.concatenate(
            [
                jnp.array(
                    [
                        params.rooms.size,
                        params.rooms.size,  # Robot position
                        1.0,
                        1.0,  # Sin/cos
                        params.rooms.size,
                        params.rooms.size,  # Goal position
                        jnp.sqrt(2) * params.rooms.size,
                        jnp.pi,  # Goal distance/angle
                    ]
                ),
                jnp.ones(params.lidar_num_beams) * params.lidar_max_distance,  # Lidar distances
                jnp.ones(params.lidar_num_beams),  # Goal flags
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
                "x": spaces.Box(0.0, params.rooms.size, (), jnp.float32),
                "y": spaces.Box(0.0, params.rooms.size, (), jnp.float32),
                "theta": spaces.Box(-jnp.pi, jnp.pi, (), jnp.float32),
                "goal_x": spaces.Box(0.0, params.rooms.size, (), jnp.float32),
                "goal_y": spaces.Box(0.0, params.rooms.size, (), jnp.float32),
                "room_idx": spaces.Discrete(params.rooms.num_rooms),
                "steps": spaces.Discrete(params.max_steps_in_episode),
                "episode_done": spaces.Discrete(2),
            }
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Terminal when goal is reached or max steps exceeded."""
        return jnp.logical_or(state.episode_done, state.steps >= params.max_steps_in_episode)

    @property
    def name(self) -> str:
        return "DifferentialDriveEnv-v1"

    @property
    def num_actions(self) -> int:
        return 2
