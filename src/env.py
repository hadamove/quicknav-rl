from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from lidar import simulate_lidar


@struct.dataclass
class EnvState(struct.PyTreeNode):
    """Environment state for the differential drive robot."""

    x: jnp.ndarray  # Robot x position
    y: jnp.ndarray  # Robot y position
    theta: jnp.ndarray  # Robot orientation (radians)
    goal_x: jnp.ndarray  # Goal x position
    goal_y: jnp.ndarray  # Goal y position
    obstacles: jnp.ndarray  # Obstacles [x, y, w, h]
    time: int = 0  # Current timestep
    terminal: jnp.ndarray = jnp.array(False)  # Episode termination flag
    accumulated_reward: jnp.ndarray = jnp.array(0.0)  # Total reward
    lidar_distances: jnp.ndarray = jnp.array([])  # Lidar readings
    lidar_collision_types: jnp.ndarray = jnp.array([])  # Beam collision types


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters."""

    arena_size: float = 5.0  # Arena size (meters)
    dt: float = 0.1  # Simulation timestep
    wheel_base: float = 0.5  # Distance between wheels
    max_wheel_speed: float = 1.0  # Max wheel speed (m/s)
    robot_radius: float = 0.15  # Robot radius
    goal_tolerance: float = 0.1  # Goal reach tolerance
    lidar_num_beams: int = struct.field(pytree_node=False, default=32)  # Number of lidar beams
    lidar_fov: float = 120.0  # Lidar field of view (degrees)
    lidar_max_distance: float = 5.0  # Maximum lidar range
    num_obstacles: int = struct.field(pytree_node=False, default=5)  # Number of obstacles
    min_obstacle_size: float = 0.3  # Minimum obstacle size
    max_obstacle_size: float = 1.0  # Maximum obstacle size
    step_penalty: float = 0.01  # Penalty per step
    collision_penalty: float = 5.0  # Collision penalty
    goal_reward: float = 100.0  # Goal reach reward
    max_steps_in_episode: int = 200  # Maximum episode length


class NavigationEnv(environment.Environment):
    """Differential drive robot navigating to a goal with obstacles and lidar."""

    def __init__(self):
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
        done = jnp.logical_or(goal_reached, state.time + 1 >= params.max_steps_in_episode)

        # 5. Lidar simulation - using imported function
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
        key, obs_key, start_key, goal_key = jax.random.split(key, 4)

        # Sample obstacle sizes and positions
        wh = jax.random.uniform(
            obs_key, shape=(params.num_obstacles, 2), minval=params.min_obstacle_size, maxval=params.max_obstacle_size
        )

        # Make sure obstacles are within arena
        xy = jax.random.uniform(
            jax.random.split(obs_key)[0],
            shape=(params.num_obstacles, 2),
            minval=0.0,
            maxval=params.arena_size - jnp.max(wh),
        )

        # Stack as [x, y, width, height]
        obstacles = jnp.concatenate([xy, wh], axis=1)

        # 2. Sample start position with safety check
        buffer = params.robot_radius * 1.5
        valid_range = params.arena_size - 2 * buffer

        def is_valid_position(pos, obstacles, radius):
            """Check if position is clear of obstacles."""
            obs_center_x = obstacles[:, 0] + obstacles[:, 2] / 2
            obs_center_y = obstacles[:, 1] + obstacles[:, 3] / 2
            min_dim = jnp.minimum(obstacles[:, 2], obstacles[:, 3]) / 2

            dists = jnp.sqrt((pos[0] - obs_center_x) ** 2 + (pos[1] - obs_center_y) ** 2)
            return jnp.all(dists >= (radius + min_dim))

        # Function to find valid robot position
        def find_valid_position(i, val):
            key, pos, valid = val
            new_key, subkey = jax.random.split(key)
            new_pos = jax.random.uniform(subkey, shape=(2,), minval=buffer, maxval=buffer + valid_range)
            new_valid = is_valid_position(new_pos, obstacles, params.robot_radius)

            # Update position if we found a valid one and didn't have one before
            should_update = new_valid & ~valid
            pos = pos.at[0].set(jnp.where(should_update, new_pos[0], pos[0]))
            pos = pos.at[1].set(jnp.where(should_update, new_pos[1], pos[1]))

            # Update validity flag
            valid = jnp.logical_or(valid, new_valid)
            return (new_key, pos, valid)

        # Try to find valid positions for robot and goal
        max_attempts = 10
        init_robot_key, init_goal_key = jax.random.split(start_key)

        # Initial random position for robot
        robot_pos_init = jax.random.uniform(init_robot_key, shape=(2,), minval=buffer, maxval=buffer + valid_range)
        robot_valid_init = is_valid_position(robot_pos_init, obstacles, params.robot_radius)

        # Try to find valid robot position
        robot_key, robot_pos, robot_valid = jax.lax.fori_loop(
            0, max_attempts, find_valid_position, (init_robot_key, robot_pos_init, robot_valid_init)
        )

        # Initial random position for goal (with minimum separation from robot)
        def is_valid_goal_position(pos, robot_pos, obstacles, radius):
            """Check if goal position is clear of obstacles and not too close to robot."""
            # Check obstacle clearance
            obs_center_x = obstacles[:, 0] + obstacles[:, 2] / 2
            obs_center_y = obstacles[:, 1] + obstacles[:, 3] / 2
            min_dim = jnp.minimum(obstacles[:, 2], obstacles[:, 3]) / 2

            obs_dists = jnp.sqrt((pos[0] - obs_center_x) ** 2 + (pos[1] - obs_center_y) ** 2)
            clear_of_obstacles = jnp.all(obs_dists >= (radius + min_dim))

            # Check minimum separation from robot (at least 4x robot radius)
            robot_dist = jnp.sqrt((pos[0] - robot_pos[0]) ** 2 + (pos[1] - robot_pos[1]) ** 2)
            min_separation = 4.0 * params.robot_radius  # Minimum distance between robot and goal
            away_from_robot = robot_dist >= min_separation

            return clear_of_obstacles & away_from_robot

        # Find a valid goal position
        def find_valid_goal(i, val):
            key, pos, valid = val
            new_key, subkey = jax.random.split(key)
            new_pos = jax.random.uniform(subkey, shape=(2,), minval=buffer, maxval=buffer + valid_range)
            new_valid = is_valid_goal_position(new_pos, robot_pos, obstacles, params.goal_tolerance)

            # Update position if we found a valid one and didn't have one before
            should_update = new_valid & ~valid
            pos = pos.at[0].set(jnp.where(should_update, new_pos[0], pos[0]))
            pos = pos.at[1].set(jnp.where(should_update, new_pos[1], pos[1]))

            # Update validity flag
            valid = jnp.logical_or(valid, new_valid)
            return (new_key, pos, valid)

        # Initial random position for goal
        goal_pos_init = jax.random.uniform(init_goal_key, shape=(2,), minval=buffer, maxval=buffer + valid_range)
        goal_valid_init = is_valid_goal_position(goal_pos_init, robot_pos, obstacles, params.goal_tolerance)

        # Try to find valid goal position
        goal_key, goal_pos, goal_valid = jax.lax.fori_loop(
            0, max_attempts, find_valid_goal, (init_goal_key, goal_pos_init, goal_valid_init)
        )

        # Random orientation
        start_theta = jax.random.uniform(jax.random.split(robot_key)[0], shape=(), minval=-jnp.pi, maxval=jnp.pi)

        # 3. Create initial state
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

        # 4. Initialize lidar readings - using imported function
        lidar_distances, collision_types = simulate_lidar(
            state.x, state.y, state.theta, obstacles, goal_pos[0], goal_pos[1], params
        )

        state = state.replace(
            lidar_distances=lidar_distances,
            lidar_collision_types=collision_types,
        )

        return self._get_obs(state, params), state

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

        # Combine with lidar
        return jnp.concatenate([pose, goal, goal_relative, state.lidar_distances])

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: [x, y, sin, cos, goal_x, goal_y, dist, angle, lidar...]."""
        n_dims = 8 + params.lidar_num_beams

        low = jnp.concatenate(
            [jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -jnp.pi]), jnp.zeros(params.lidar_num_beams)]
        )

        high = jnp.concatenate(
            [
                jnp.array(
                    [
                        params.arena_size,
                        params.arena_size,
                        1.0,
                        1.0,
                        params.arena_size,
                        params.arena_size,
                        jnp.sqrt(2) * params.arena_size,
                        jnp.pi,
                    ]
                ),
                jnp.ones(params.lidar_num_beams) * params.lidar_max_distance,
            ]
        )

        return spaces.Box(low, high, (n_dims,), jnp.float32)

    def action_space(self, params: EnvParams) -> spaces.Box:
        """Action space: [left_wheel_speed, right_wheel_speed]."""
        low = jnp.array([-params.max_wheel_speed, -params.max_wheel_speed])
        high = jnp.array([params.max_wheel_speed, params.max_wheel_speed])
        return spaces.Box(low, high, (2,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space definition."""
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
        """Terminal state check."""
        return jnp.logical_or(state.terminal, state.time >= params.max_steps_in_episode)

    @property
    def name(self) -> str:
        return "DifferentialDriveEnv-v0"

    @property
    def num_actions(self) -> int:
        return 2
