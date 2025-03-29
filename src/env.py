from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from constants import GOAL, MAX_DIST, OBSTACLE, WALL


def is_in_obstacle(x: jnp.ndarray, y: jnp.ndarray, obstacles: jnp.ndarray) -> jnp.ndarray:
    """Returns True if (x,y) is inside any obstacle (axis-aligned check)."""
    return jnp.any(
        (x >= obstacles[:, 0])
        & (x <= obstacles[:, 0] + obstacles[:, 2])
        & (y >= obstacles[:, 1])
        & (y <= obstacles[:, 1] + obstacles[:, 3])
    )


def sample_valid_point(
    key: chex.PRNGKey, arena_size: float, obstacles: jnp.ndarray, robot_radius: float
) -> Tuple[jnp.ndarray, chex.PRNGKey]:
    """
    Samples a point uniformly from [robot_radius, arena_size - robot_radius]
    until it is not inside any obstacle.
    """

    def cond_fn(carry):
        key, point = carry
        return is_in_obstacle(point[0], point[1], obstacles)

    def body_fn(carry):
        key, _ = carry
        key, new_key = jax.random.split(key)
        new_point = jax.random.uniform(
            new_key, shape=(2,), dtype=jnp.float32, minval=robot_radius, maxval=arena_size - robot_radius
        )
        return key, new_point

    point = jax.random.uniform(
        key, shape=(2,), dtype=jnp.float32, minval=robot_radius, maxval=arena_size - robot_radius
    )
    key, point = jax.lax.while_loop(cond_fn, body_fn, (key, point))
    return point, key


def sphere_collision(x: jnp.ndarray, y: jnp.ndarray, obstacles: jnp.ndarray, robot_radius: float) -> jnp.ndarray:
    """
    Returns True if the distance from (x,y) to any obstacle (using a sphere collider)
    is less than robot_radius.
    """
    # For each obstacle, compute horizontal and vertical distances to its edges.
    dx = jnp.maximum(obstacles[:, 0] - x, x - (obstacles[:, 0] + obstacles[:, 2]))
    dx = jnp.maximum(dx, 0.0)
    dy = jnp.maximum(obstacles[:, 1] - y, y - (obstacles[:, 1] + obstacles[:, 3]))
    dy = jnp.maximum(dy, 0.0)
    dist = jnp.sqrt(dx**2 + dy**2)
    return jnp.any(dist < robot_radius)


def compute_wall_distance(
    x: jnp.ndarray, y: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray, arena_size: float
) -> jnp.ndarray:
    # Compute t for each wall; if direction component is near zero, use a large number.
    t_x = jnp.where(jnp.abs(dx) > 1e-6, jnp.where(dx > 0, (arena_size - x) / dx, -x / dx), jnp.inf)
    t_y = jnp.where(jnp.abs(dy) > 1e-6, jnp.where(dy > 0, (arena_size - y) / dy, -y / dy), jnp.inf)
    # Only positive intersections matter.
    t_candidates = jnp.array([t_x, t_y])
    t_candidates = jnp.where(t_candidates > 0, t_candidates, jnp.inf)
    return jnp.min(t_candidates)


def compute_obstacle_distance(
    x: jnp.ndarray, y: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray, obstacles: jnp.ndarray
) -> jnp.ndarray:
    """
    For each obstacle, compute intersection distance via the slab method.
    obstacles: shape (n, 4) with rows [ox, oy, w, h]
    Returns the minimal positive distance if any intersection; otherwise, returns jnp.inf.
    """
    # For numerical stability, add small epsilon to denominator.
    eps = 1e-6
    # Extract obstacle coordinates
    ox, oy, ow, oh = obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], obstacles[:, 3]

    # Compute t for x and y slabs
    t_min_x = jnp.minimum((ox - x) / (dx + eps), ((ox + ow) - x) / (dx + eps))
    t_max_x = jnp.maximum((ox - x) / (dx + eps), ((ox + ow) - x) / (dx + eps))
    t_min_y = jnp.minimum((oy - y) / (dy + eps), ((oy + oh) - y) / (dy + eps))
    t_max_y = jnp.maximum((oy - y) / (dy + eps), ((oy + oh) - y) / (dy + eps))

    t_entry = jnp.maximum(t_min_x, t_min_y)
    t_exit = jnp.minimum(t_max_x, t_max_y)

    valid = (t_exit >= t_entry) & (t_exit > 0)
    # For valid intersections, use t_entry (if negative, set to inf).
    t_entry = jnp.where(t_entry > 0, t_entry, jnp.inf)
    distances = jnp.where(valid, t_entry, jnp.inf)
    return jnp.min(distances)


def compute_goal_intersection(
    x: jnp.ndarray,
    y: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    goal_x: jnp.ndarray,
    goal_y: jnp.ndarray,
    goal_tolerance: float,
) -> jnp.ndarray:
    """Compute the distance to intersection with the goal (circle)"""
    # Vector from start to goal center
    ocx, ocy = x - goal_x, y - goal_y

    # Quadratic equation coefficients for circle intersection
    b = 2 * (ocx * dx + ocy * dy)
    c = ocx**2 + ocy**2 - goal_tolerance**2
    disc = b**2 - 4 * c

    # If discriminant is negative, no intersection
    t_candidates = jnp.array([jnp.inf, jnp.inf])

    # For valid intersections, compute both t values
    t1 = (-b - jnp.sqrt(jnp.maximum(0.0, disc))) / 2
    t2 = (-b + jnp.sqrt(jnp.maximum(0.0, disc))) / 2

    # Only positive t values matter
    t_candidates = jnp.where(disc >= 0, jnp.where(t1 >= 0, t1, jnp.where(t2 >= 0, t2, jnp.inf)), jnp.inf)

    return t_candidates


def simulate_lidar(
    x: jnp.ndarray,
    y: jnp.ndarray,
    theta: jnp.ndarray,
    obstacles: jnp.ndarray,
    goal_x: jnp.ndarray,
    goal_y: jnp.ndarray,
    params: "EnvParams",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate a lidar sensor.
    Returns:
        - distances: array of shape (lidar_num_beams,) with min distances
        - collision_types: array of shape (lidar_num_beams,) with collision types
          (0=max_distance, 1=wall, 2=obstacle, 3=goal)
    """
    num_beams = params.lidar_num_beams
    fov_rad = params.lidar_fov * jnp.pi / 180.0
    beam_offsets = jnp.linspace(-fov_rad / 2, fov_rad / 2, num_beams)
    beam_angles = theta + beam_offsets

    def beam_distance(angle):
        dx, dy = jnp.cos(angle), jnp.sin(angle)
        wall_dist = compute_wall_distance(x, y, dx, dy, params.arena_size)
        obs_dist = compute_obstacle_distance(x, y, dx, dy, obstacles)
        goal_dist = compute_goal_intersection(x, y, dx, dy, goal_x, goal_y, params.goal_tolerance)

        # Compute minimum distance and identify what was hit
        distances = jnp.array([wall_dist, obs_dist, goal_dist, params.lidar_max_distance])
        min_idx = jnp.argmin(distances)
        dist = distances[min_idx]

        # Map index to collision type (wall=1, obstacle=2, goal=3, max_dist=0)
        collision_type = jnp.where(
            min_idx == 0, WALL, jnp.where(min_idx == 1, OBSTACLE, jnp.where(min_idx == 2, GOAL, MAX_DIST))
        )

        return dist, collision_type

    # Apply beam_distance to all angles
    distances_and_types = jax.vmap(beam_distance)(beam_angles)

    # Unpack the results
    distances = distances_and_types[0]
    collision_types = distances_and_types[1]

    return distances, collision_types


class EnvState(struct.PyTreeNode):
    """Environment state for the differential drive robot."""

    x: jnp.ndarray  # Robot x position
    y: jnp.ndarray  # Robot y position
    theta: jnp.ndarray  # Robot orientation (radians)
    goal_x: jnp.ndarray  # Goal x position
    goal_y: jnp.ndarray  # Goal y position
    time: int  # Current timestep
    terminal: jnp.ndarray  # Episode termination flag (as bool ndarray)
    obstacles: jnp.ndarray  # Obstacles as rows [x, y, w, h]
    accumulated_reward: jnp.ndarray  # Total reward (as float ndarray)
    # Updated lidar fields
    lidar_distances: jnp.ndarray = jnp.array([])  # Lidar distances
    lidar_collision_types: jnp.ndarray = jnp.array([])  # Collision types (0=max_dist, 1=wall, 2=obstacle, 3=goal)


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters for the differential drive robot."""

    max_steps_in_episode: int = 200
    goal_tolerance: float = 0.1  # meters
    arena_size: float = 5.0  # arena size (meters)
    dt: float = 0.1  # simulation timestep

    wheel_base: float = 0.5  # distance between wheels
    max_wheel_speed: float = 1.0  # max wheel speed (m/s)
    collision_penalty: float = 5.0  # penalty for collision with an obstacle
    robot_radius: float = 0.15  # robot collider radius

    # Lidar parameters:
    lidar_num_beams: int = struct.field(pytree_node=False, default=32)
    lidar_fov: float = 120.0  # in degrees
    lidar_max_distance: float = 5.0

    num_obstacles: int = struct.field(pytree_node=False, default=5)
    min_obstacle_size: float = 0.3
    max_obstacle_size: float = 1.0

    step_penalty: float = 0.01  # penalty per step
    goal_reward: float = 100.0  # bonus for reaching the goal


class NavigationEnv(environment.Environment):
    """
    Differential drive robot navigating to a goal with obstacles and a simulated lidar.

    ENVIRONMENT DESCRIPTION:
    - The robot operates in a square arena with random rectangular obstacles.
    - Observations: [x, y, cos(theta), sin(theta), goal_x, goal_y, lidar...].
    - Actions: Continuous [v_left, v_right] wheel speeds.
    - Kinematics: v = (v_left+v_right)/2, omega = (v_right-v_left)/wheel_base.
    - Collisions (using a sphere collider) yield negative rewards but do not terminate the episode.
    - Start and goal positions are resampled to be outside obstacles and away from walls.
    - Lidar beams are simulated in a configurable FOV.
    """

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
        # Ensure action is a 2D array
        action = jnp.asarray(action, dtype=jnp.float32)
        assert action.shape == (2,), "Action must be a 2D array."

        # --- Compute kinematics ---
        v_left = jnp.clip(action[0], -params.max_wheel_speed, params.max_wheel_speed)
        v_right = jnp.clip(action[1], -params.max_wheel_speed, params.max_wheel_speed)

        # Linear and angular velocities
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / params.wheel_base

        # Update orientation and position
        new_theta = state.theta + omega * params.dt
        new_theta = jnp.arctan2(jnp.sin(new_theta), jnp.cos(new_theta))  # Normalize to [-pi, pi]

        dx = v * jnp.cos(state.theta) * params.dt
        dy = v * jnp.sin(state.theta) * params.dt
        new_x = state.x + dx
        new_y = state.y + dy

        # Enforce arena boundaries
        new_x = jnp.clip(new_x, params.robot_radius, params.arena_size - params.robot_radius)
        new_y = jnp.clip(new_y, params.robot_radius, params.arena_size - params.robot_radius)

        # --- Reward calculation ---
        prev_dist = jnp.sqrt((state.x - state.goal_x) ** 2 + (state.y - state.goal_y) ** 2)
        new_dist = jnp.sqrt((new_x - state.goal_x) ** 2 + (new_y - state.goal_y) ** 2)
        reward = prev_dist - new_dist - params.step_penalty

        # --- Check termination conditions ---
        goal_reached = new_dist <= params.goal_tolerance
        time_exceeded = (state.time + 1) >= params.max_steps_in_episode
        done = jnp.logical_or(goal_reached, time_exceeded)

        # Apply reward for goal reached
        reward = jnp.where(goal_reached, reward + params.goal_reward, reward)

        # --- Collision detection ---
        obs_centers_x = state.obstacles[:, 0] + state.obstacles[:, 2] / 2
        obs_centers_y = state.obstacles[:, 1] + state.obstacles[:, 3] / 2
        min_dim = jnp.minimum(state.obstacles[:, 2], state.obstacles[:, 3])

        collision = jnp.any(
            jnp.sqrt((new_x - obs_centers_x) ** 2 + (new_y - obs_centers_y) ** 2)
            < (params.robot_radius + 0.5 * min_dim)
        )

        # If collision, reset position and apply penalty
        new_x = jnp.where(collision, state.x, new_x)
        new_y = jnp.where(collision, state.y, new_y)
        reward = jnp.where(collision, -params.collision_penalty, reward)

        # Simulate lidar for the new state
        lidar_distances, lidar_collision_types = simulate_lidar(
            new_x, new_y, new_theta, state.obstacles, state.goal_x, state.goal_y, params
        )

        # Update state
        new_state = EnvState(
            x=new_x,
            y=new_y,
            theta=new_theta,
            goal_x=state.goal_x,
            goal_y=state.goal_y,
            time=state.time + 1,
            terminal=done,
            obstacles=state.obstacles,
            accumulated_reward=state.accumulated_reward + reward,
            lidar_distances=lidar_distances,
            lidar_collision_types=lidar_collision_types,
        )

        # Get observation and return results
        obs = self._get_obs(new_state, params)
        info = {"discount": jnp.where(new_state.terminal, jnp.array(0.0), jnp.array(1.0))}

        return obs, new_state, reward, done, info

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment with obstacles and valid start/goal positions."""
        # --- Generate obstacles ---
        num_obs = params.num_obstacles
        key, wh_key, xy_key = jax.random.split(key, 3)

        # Generate obstacle sizes
        wh = jax.random.uniform(wh_key, shape=(num_obs, 2), dtype=jnp.float32)
        wh = params.min_obstacle_size + (params.max_obstacle_size - params.min_obstacle_size) * wh

        # Generate obstacle positions
        x_key, y_key = jax.random.split(xy_key)
        x_obs = jax.random.uniform(
            x_key, shape=(num_obs,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size - wh[:, 0]
        )
        y_obs = jax.random.uniform(
            y_key, shape=(num_obs,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size - wh[:, 1]
        )

        # Stack obstacles [x, y, w, h]
        obstacles = jnp.stack([x_obs, y_obs, wh[:, 0], wh[:, 1]], axis=1)

        # --- Sample valid start and goal positions ---
        start, key = sample_valid_point(key, params.arena_size, obstacles, params.robot_radius)
        key, theta_key = jax.random.split(key)
        start_theta = jax.random.uniform(theta_key, shape=(), dtype=jnp.float32, minval=-jnp.pi, maxval=jnp.pi)
        goal, key = sample_valid_point(key, params.arena_size, obstacles, params.robot_radius)

        # Initialize state with empty arrays for lidar arrays, which will be updated
        state = EnvState(
            x=start[0],
            y=start[1],
            theta=start_theta,
            goal_x=goal[0],
            goal_y=goal[1],
            time=0,
            terminal=jnp.array(False),
            obstacles=obstacles,
            accumulated_reward=jnp.array(0.0),
            lidar_distances=jnp.zeros(params.lidar_num_beams),
            lidar_collision_types=jnp.zeros(params.lidar_num_beams, dtype=jnp.int32),
        )

        # Update lidar readings for initial state
        lidar_distances, lidar_collision_types = simulate_lidar(
            state.x, state.y, state.theta, obstacles, goal[0], goal[1], params
        )

        state = state.replace(
            lidar_distances=lidar_distances,
            lidar_collision_types=lidar_collision_types,
        )

        return self._get_obs(state, params), state

    def _get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return observation: [x, y, cos(theta), sin(theta), goal_x, goal_y, goal_distance, goal_angle, lidar beams]."""
        # Base observations
        cos_theta, sin_theta = jnp.cos(state.theta), jnp.sin(state.theta)
        base_obs = jnp.array([state.x, state.y, cos_theta, sin_theta, state.goal_x, state.goal_y], dtype=jnp.float32)

        # Goal sensor readings
        goal_dx = state.goal_x - state.x
        goal_dy = state.goal_y - state.y
        goal_distance = jnp.sqrt(goal_dx**2 + goal_dy**2)
        goal_angle = jnp.arctan2(goal_dy, goal_dx) - state.theta
        goal_angle = jnp.arctan2(jnp.sin(goal_angle), jnp.cos(goal_angle))  # Normalize to [-pi, pi]

        goal_obs = jnp.array([goal_distance, goal_angle], dtype=jnp.float32)

        # Use precomputed lidar distances
        lidar_obs = state.lidar_distances

        return jnp.concatenate([base_obs, goal_obs, lidar_obs], axis=0)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: [x, y, cos(theta), sin(theta), goal_x, goal_y, goal_distance, goal_angle, lidar beams]."""
        n_dims = 6 + 2 + params.lidar_num_beams

        # Construct low and high bounds
        low = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -jnp.pi], dtype=jnp.float32),
                jnp.zeros(params.lidar_num_beams, dtype=jnp.float32),
            ]
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
                        params.arena_size * jnp.sqrt(2),
                        jnp.pi,
                    ],
                    dtype=jnp.float32,
                ),
                jnp.ones(params.lidar_num_beams, dtype=jnp.float32) * params.lidar_max_distance,
            ]
        )

        return spaces.Box(low, high, (n_dims,), dtype=jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        return jnp.logical_or(state.terminal, state.time >= params.max_steps_in_episode)

    @property
    def name(self) -> str:
        return "DifferentialDriveEnv-v0"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: EnvParams) -> spaces.Box:
        low = jnp.array([-params.max_wheel_speed, -params.max_wheel_speed], dtype=jnp.float32)
        high = jnp.array([params.max_wheel_speed, params.max_wheel_speed], dtype=jnp.float32)
        return spaces.Box(low, high, (2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
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
