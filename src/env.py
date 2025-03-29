from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


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
    t_x = jnp.where(dx > 1e-6, (arena_size - x) / dx, jnp.where(dx < -1e-6, (0.0 - x) / dx, jnp.inf))
    t_y = jnp.where(dy > 1e-6, (arena_size - y) / dy, jnp.where(dy < -1e-6, (0.0 - y) / dy, jnp.inf))
    # Only positive intersections matter.
    t_candidates = jnp.array([t_x, t_y])
    t_candidates = jnp.where(t_candidates > 0, t_candidates, jnp.inf)
    return jnp.minimum(t_candidates[0], t_candidates[1])


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
    # Expand x,y and direction to shape (num_obstacles,)
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    ow = obstacles[:, 2]
    oh = obstacles[:, 3]

    # Compute t for x slabs.
    t1 = (ox - x) / (dx + eps)
    t2 = ((ox + ow) - x) / (dx + eps)
    t_min_x = jnp.minimum(t1, t2)
    t_max_x = jnp.maximum(t1, t2)

    # Compute t for y slabs.
    t3 = (oy - y) / (dy + eps)
    t4 = ((oy + oh) - y) / (dy + eps)
    t_min_y = jnp.minimum(t3, t4)
    t_max_y = jnp.maximum(t3, t4)

    t_entry = jnp.maximum(t_min_x, t_min_y)
    t_exit = jnp.minimum(t_max_x, t_max_y)

    valid = (t_exit >= t_entry) & (t_exit > 0)
    # For valid intersections, use t_entry (if negative, set to inf).
    t_entry = jnp.where(t_entry > 0, t_entry, jnp.inf)
    distances = jnp.where(valid, t_entry, jnp.inf)
    return jnp.min(distances)


def simulate_lidar(
    x: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, obstacles: jnp.ndarray, params: "EnvParams"
) -> jnp.ndarray:
    """
    Simulate a lidar sensor.
    Returns an array of distances of shape (lidar_num_beams,).
    """
    num_beams = params.lidar_num_beams
    fov_rad = params.lidar_fov * jnp.pi / 180.0
    beam_offsets = jnp.linspace(-fov_rad / 2, fov_rad / 2, num_beams)
    beam_angles = theta + beam_offsets

    def beam_distance(angle):
        dx = jnp.cos(angle)
        dy = jnp.sin(angle)
        wall_dist = compute_wall_distance(x, y, dx, dy, params.arena_size)
        obs_dist = compute_obstacle_distance(x, y, dx, dy, obstacles)
        d = jnp.minimum(wall_dist, obs_dist)
        return jnp.minimum(d, params.lidar_max_distance)

    lidar_dists = jax.vmap(beam_distance)(beam_angles)
    return lidar_dists


@struct.dataclass
class EnvState(environment.EnvState):
    """Environment state for the differential drive robot."""

    x: chex.Array  # Robot x position
    y: chex.Array  # Robot y position
    theta: chex.Array  # Robot orientation (radians)
    goal_x: chex.Array  # Goal x position
    goal_y: chex.Array  # Goal y position
    time: int  # Current timestep
    terminal: bool  # Episode termination flag
    obstacles: jnp.ndarray  # Obstacles as rows [x, y, w, h]
    accumulated_reward: float = 0.0  # Total reward


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


class NavigationEnv(environment.Environment[EnvState, EnvParams]):
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
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int | float | chex.Array,  # [v_left, v_right]
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform a single timestep state transition."""

        # Ensure that action is a 2D array
        assert isinstance(action, jnp.ndarray) and action.shape == (2,), "Action must be a 2D array."

        # --- Compute kinematics ---
        v_left = jnp.clip(action[0], -params.max_wheel_speed, params.max_wheel_speed)
        v_right = jnp.clip(action[1], -params.max_wheel_speed, params.max_wheel_speed)

        # Compute linear and angular velocities
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / params.wheel_base

        # Compute new position and orientation
        new_theta = jnp.add(state.theta, omega * params.dt)
        new_theta = jnp.arctan2(jnp.sin(new_theta), jnp.cos(new_theta))  # Normalize to [-pi, pi]

        # Update position (use jnp.add, jnp.multiply for arithmetic)
        new_x = jnp.add(state.x, jnp.multiply(v, jnp.multiply(jnp.cos(state.theta), params.dt)))
        new_y = jnp.add(state.y, jnp.multiply(v, jnp.multiply(jnp.sin(state.theta), params.dt)))

        # --- Enforce arena boundaries with a sphere collider ---
        new_x = jnp.clip(new_x, params.robot_radius, params.arena_size - params.robot_radius)
        new_y = jnp.clip(new_y, params.robot_radius, params.arena_size - params.robot_radius)

        # --- Reward calculation ---
        prev_dist = jnp.sqrt(
            jnp.add(
                jnp.power(jnp.subtract(state.x, state.goal_x), 2), jnp.power(jnp.subtract(state.y, state.goal_y), 2)
            )
        )
        new_dist = jnp.sqrt(
            jnp.add(jnp.power(jnp.subtract(new_x, state.goal_x), 2), jnp.power(jnp.subtract(new_y, state.goal_y), 2))
        )
        reward = jnp.subtract(prev_dist, new_dist) - params.step_penalty

        # --- Check termination (goal reached or time exceeded) ---
        goal_reached = new_dist <= params.goal_tolerance
        time_exceeded = (state.time + 1) >= params.max_steps_in_episode
        done = jnp.logical_or(goal_reached, time_exceeded)

        # Apply reward for goal reached
        reward = jax.lax.select(goal_reached, reward + params.goal_reward, reward)

        # --- Collision detection (sphere collider) with obstacles ---
        collision = jnp.any(
            jnp.sqrt(
                jnp.add(
                    jnp.power(jnp.subtract(new_x, (state.obstacles[:, 0] + state.obstacles[:, 2] / 2)), 2),
                    jnp.power(jnp.subtract(new_y, (state.obstacles[:, 1] + state.obstacles[:, 3] / 2)), 2),
                )
            )
            < (params.robot_radius + 0.5 * jnp.minimum(state.obstacles[:, 2], state.obstacles[:, 3]))
        )

        # If collision, reset position and apply collision penalty
        new_x = jax.lax.select(collision, state.x, new_x)
        new_y = jax.lax.select(collision, state.y, new_y)
        reward = jax.lax.select(collision, -params.collision_penalty, reward)
        # Note: collision does not terminate the episode.

        # Update state
        state = state.replace(
            x=new_x,
            y=new_y,
            theta=new_theta,
            time=state.time + 1,
            terminal=done,
            accumulated_reward=state.accumulated_reward + reward,
        )

        # Get observation and return results
        obs = self.get_obs(state, params)
        info = {"discount": self.discount(state, params)}

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def discount(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        return jax.lax.select(state.terminal, 0.0, 1.0)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment with obstacles and valid start/goal positions."""
        # --- Generate obstacles ---
        num_obs = params.num_obstacles
        key, wh_key, xy_key = jax.random.split(key, 3)
        wh = jax.random.uniform(wh_key, shape=(num_obs, 2), dtype=jnp.float32)
        wh = params.min_obstacle_size + (params.max_obstacle_size - params.min_obstacle_size) * wh
        w_obs = wh[:, 0]
        h_obs = wh[:, 1]
        x_key, y_key = jax.random.split(xy_key)
        x_obs = jax.random.uniform(
            x_key, shape=(num_obs,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size - w_obs
        )
        y_obs = jax.random.uniform(
            y_key, shape=(num_obs,), dtype=jnp.float32, minval=0.0, maxval=params.arena_size - h_obs
        )
        obstacles = jnp.stack([x_obs, y_obs, w_obs, h_obs], axis=1)

        # --- Sample valid start and goal positions ---
        start, key = sample_valid_point(key, params.arena_size, obstacles, params.robot_radius)
        key, theta_key = jax.random.split(key)
        start_theta = jax.random.uniform(theta_key, shape=(), dtype=jnp.float32, minval=-jnp.pi, maxval=jnp.pi)
        goal, key = sample_valid_point(key, params.arena_size, obstacles, params.robot_radius)

        state = EnvState(
            x=start[0],
            y=start[1],
            theta=start_theta,
            goal_x=goal[0],
            goal_y=goal[1],
            time=0,
            terminal=False,
            obstacles=obstacles,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return observation: [x, y, cos(theta), sin(theta), goal_x, goal_y, goal_distance, goal_angle, lidar beams]."""
        base_obs = jnp.array(
            [state.x, state.y, jnp.cos(state.theta), jnp.sin(state.theta), state.goal_x, state.goal_y],
            dtype=jnp.float32,
        )
        # Compute dedicated goal sensor readings.
        goal_dx = jnp.subtract(state.goal_x, state.x)
        goal_dy = jnp.subtract(state.goal_y, state.y)
        goal_distance = jnp.sqrt(goal_dx**2 + goal_dy**2)
        goal_angle = jnp.arctan2(goal_dy, goal_dx) - state.theta
        goal_angle = jnp.arctan2(jnp.sin(goal_angle), jnp.cos(goal_angle))  # Normalize to [-pi, pi]
        goal_obs = jnp.array([goal_distance, goal_angle], dtype=jnp.float32)
        lidar_obs = simulate_lidar(state.x, state.y, state.theta, state.obstacles, params)
        return jnp.concatenate([base_obs, goal_obs, lidar_obs], axis=0)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: [x, y, cos(theta), sin(theta), goal_x, goal_y, goal_distance, goal_angle, lidar beams]."""
        base_low = jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=jnp.float32)
        base_high = jnp.array(
            [params.arena_size, params.arena_size, 1.0, 1.0, params.arena_size, params.arena_size], dtype=jnp.float32
        )
        goal_low = jnp.array([0.0, -jnp.pi], dtype=jnp.float32)
        # Maximum goal distance can be diagonal of arena.
        goal_high = jnp.array([params.arena_size * jnp.sqrt(2), jnp.pi], dtype=jnp.float32)
        lidar_low = jnp.zeros((params.lidar_num_beams,), dtype=jnp.float32)
        lidar_high = params.lidar_max_distance * jnp.ones((params.lidar_num_beams,), dtype=jnp.float32)
        low = jnp.concatenate([base_low, goal_low, lidar_low], axis=0)
        high = jnp.concatenate([base_high, goal_high, lidar_high], axis=0)
        return spaces.Box(low, high, (6 + 2 + params.lidar_num_beams,), dtype=jnp.float32)

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
