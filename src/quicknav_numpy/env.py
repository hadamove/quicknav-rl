"""Numpy environment for differential drive robot navigation with lidar sensing."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .geometry import handle_collision_with_sliding, point_to_rectangle_distance
from .lidar import Collision, simulate_lidar
from .rooms import RoomParams, sample_position


class NavigationEnvParams:
    """Parameters for configuring the navigation environment.

    Defines all configurable aspects of the environment, including physical dimensions,
    robot characteristics, sensor properties, rewards, and episode parameters.
    """

    def __init__(
        self,
        # Robot parameters
        wheel_base: float = 0.3,
        max_wheel_speed: float = 1.0,
        robot_radius: float = 0.15,
        dt: float = 0.1,
        # Environment parameters
        rooms: Optional[RoomParams] = None,
        # Pre-generated rooms
        obstacles: Optional[np.ndarray] = None,
        free_positions: Optional[np.ndarray] = None,
        # Sensor parameters
        lidar_num_beams: int = 16,
        lidar_fov: float = 120.0,
        lidar_max_distance: float = 3.0,
        # Reward parameters
        goal_tolerance: float = 0.1,
        step_penalty: float = 0.1,
        collision_penalty: float = 10.0,
        goal_reward: float = 100.0,
        # Episode parameters
        max_steps_in_episode: int = 300,
    ):
        """Initialize environment parameters."""
        # Robot parameters
        self.wheel_base = wheel_base
        self.max_wheel_speed = max_wheel_speed
        self.robot_radius = robot_radius
        self.dt = dt

        # Environment parameters
        self.rooms = rooms if rooms is not None else RoomParams()

        # Pre-generated rooms (should be provided before use)
        if obstacles is None:
            self.obstacles = np.zeros((0, 0, 4))
        else:
            self.obstacles = obstacles

        if free_positions is None:
            self.free_positions = np.zeros((0, 0, 2))
        else:
            self.free_positions = free_positions

        # Sensor parameters
        self.lidar_num_beams = lidar_num_beams
        self.lidar_fov = lidar_fov
        self.lidar_max_distance = lidar_max_distance

        # Reward parameters
        self.goal_tolerance = goal_tolerance
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward

        # Episode parameters
        self.max_steps_in_episode = max_steps_in_episode


class EnvState:
    """Environment state for the differential drive robot navigation task.

    Contains all relevant information about the current state of the environment,
    including robot pose, goal location, obstacles, and sensor readings.
    """

    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        goal_x: float,
        goal_y: float,
        obstacles: np.ndarray,
        room_idx: int,
        steps: int,
        episode_done: bool,
        accumulated_reward: float,
        lidar_distances: np.ndarray,
        lidar_collision_types: np.ndarray,
        position_history: np.ndarray,
    ):
        """Initialize environment state."""
        # Robot state
        self.x = x  # Robot x position (meters)
        self.y = y  # Robot y position (meters)
        self.theta = theta  # Robot orientation (radians)

        # Goal state
        self.goal_x = goal_x  # Goal x position (meters)
        self.goal_y = goal_y  # Goal y position (meters)

        # Environment elements
        self.obstacles = obstacles  # Obstacle coordinates as [x, y, width, height] array
        self.room_idx = room_idx  # Index of the currently used pre-generated room

        # Sensor readings
        self.lidar_distances = lidar_distances  # Distance readings from lidar beams (meters)
        self.lidar_collision_types = lidar_collision_types  # Type of object each beam hit

        # Episode state
        self.steps = steps  # Current timestep in the episode
        self.episode_done = episode_done  # Whether the episode has terminated
        self.accumulated_reward = accumulated_reward  # Total reward collected so far

        # Position history
        self.position_history = position_history  # Buffer of past positions


class NavigationEnv(gym.Env):
    """Differential drive robot navigating to a goal with obstacles and lidar."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        params: Optional[NavigationEnvParams] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the environment."""
        self.params = params if params is not None else NavigationEnvParams()
        self.render_mode = render_mode
        self.state = None
        self.random_state = np.random.RandomState(seed)

        # Lower bounds for observation
        low = np.concatenate(
            [
                np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -np.pi]),  # Robot, goal, relative
                np.zeros(self.params.lidar_num_beams),  # Lidar distances
                np.zeros(self.params.lidar_num_beams),  # Goal flags
            ]
        )

        # Upper bounds for observation
        high = np.concatenate(
            [
                np.array(
                    [
                        self.params.rooms.size,
                        self.params.rooms.size,  # Robot position
                        1.0,
                        1.0,  # Sin/cos
                        self.params.rooms.size,
                        self.params.rooms.size,  # Goal position
                        np.sqrt(2) * self.params.rooms.size,
                        np.pi,  # Goal distance/angle
                    ]
                ),
                np.ones(self.params.lidar_num_beams) * self.params.lidar_max_distance,  # Lidar distances
                np.ones(self.params.lidar_num_beams),  # Goal flags
            ]
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: [left_wheel_speed, right_wheel_speed]
        self.action_space = spaces.Box(
            low=np.array([-self.params.max_wheel_speed, -self.params.max_wheel_speed]),
            high=np.array([self.params.max_wheel_speed, self.params.max_wheel_speed]),
            dtype=np.float32,
        )

        # For rendering
        self.viewer = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Perform a single timestep state transition.

        Args:
            action: [left_wheel_speed, right_wheel_speed]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert self.state is not None

        # 1. Physics update
        action = np.clip(action, -self.params.max_wheel_speed, self.params.max_wheel_speed)
        v_left, v_right = action
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.params.wheel_base

        # Update position and orientation
        new_theta = np.arctan2(
            np.sin(self.state.theta + omega * self.params.dt), np.cos(self.state.theta + omega * self.params.dt)
        )
        dx, dy = v * np.cos(self.state.theta) * self.params.dt, v * np.sin(self.state.theta) * self.params.dt
        new_x = np.clip(self.state.x + dx, self.params.robot_radius, self.params.rooms.size - self.params.robot_radius)
        new_y = np.clip(self.state.y + dy, self.params.robot_radius, self.params.rooms.size - self.params.robot_radius)

        # 2. Collision detection
        # Calculate distance from robot to each obstacle
        robot_pos = np.array([new_x, new_y])
        distances = np.array([point_to_rectangle_distance(robot_pos, obstacle) for obstacle in self.state.obstacles])
        collision = np.any(distances < self.params.robot_radius)

        # Handle collision with sliding behavior from geometry module
        slide_x, slide_y = handle_collision_with_sliding(
            self.state.x, self.state.y, new_x, new_y, self.state.obstacles, self.params.robot_radius
        )

        # Apply sliding only if there's a collision
        if collision:
            new_x, new_y = slide_x, slide_y

        # 3. Reward calculation
        reward, goal_reached = self._calculate_reward(new_x, new_y, bool(collision))

        # 4. Terminal state check
        out_of_time = self.state.steps + 1 >= self.params.max_steps_in_episode
        done = goal_reached or out_of_time

        # 5. Lidar simulation
        lidar_distances, collision_types = simulate_lidar(
            new_x, new_y, new_theta, self.state.obstacles, self.state.goal_x, self.state.goal_y, self.params
        )

        # 6. Update position history buffer
        position_history = self.state.position_history.copy()
        position_history[self.state.steps] = np.array([new_x, new_y])

        # 7. Update state
        self.state = EnvState(
            x=new_x,
            y=new_y,
            theta=new_theta,
            goal_x=self.state.goal_x,
            goal_y=self.state.goal_y,
            obstacles=self.state.obstacles,
            room_idx=self.state.room_idx,
            steps=self.state.steps + 1,
            episode_done=done,
            accumulated_reward=self.state.accumulated_reward + reward,
            lidar_distances=lidar_distances,
            lidar_collision_types=collision_types,
            position_history=position_history,
        )

        # 8. Return observations, reward, done, info
        obs = self._get_obs()
        info = {"discount": 0.0 if done else 1.0}
        terminated = goal_reached  # Task completion
        truncated = out_of_time  # Episode time limit

        return obs, reward, terminated, truncated, info

    def _calculate_reward(
        self,
        new_x: float,
        new_y: float,
        collision: bool,
    ) -> Tuple[float, bool]:
        """Calculate reward and check if goal is reached.

        Args:
            new_x: New x position after movement
            new_y: New y position after movement
            collision: Whether a collision occurred

        Returns:
            Tuple of (reward, goal_reached)
        """
        assert self.state is not None

        # Calculate distance to goal before and after movement
        prev_dist = np.sqrt((self.state.x - self.state.goal_x) ** 2 + (self.state.y - self.state.goal_y) ** 2)
        new_dist = np.sqrt((new_x - self.state.goal_x) ** 2 + (new_y - self.state.goal_y) ** 2)
        goal_reached = new_dist <= self.params.goal_tolerance

        # Compute reward components
        progress_reward = prev_dist - new_dist
        collision_reward = -self.params.collision_penalty if collision else 0.0
        goal_reward = self.params.goal_reward if goal_reached else 0.0
        step_penalty = -self.params.step_penalty

        total_reward = progress_reward + collision_reward + goal_reward + step_penalty

        return total_reward, goal_reached

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment state by sampling a pre-generated room layout.

        Args:
            seed: Optional seed to use for RNG
            options: Optional dictionary with configuration:
                - test: If True, use second half of rooms for testing, else use first half for training

        Returns:
            Tuple of (observation, info)
        """
        # Initialize random state if seed is provided
        if seed is not None:
            self.random_state = np.random.RandomState(seed)

        # Set default options
        if options is None:
            options = {}

        test = options.get("test", False)

        # Sample a random room index from either the first or second half of rooms based on test flag
        num_rooms = self.params.rooms.num_rooms
        half_rooms = num_rooms // 2

        min_room = half_rooms if test else 0  # If test=True, use second half; if test=False, use first half
        max_room = num_rooms if test else half_rooms
        room_idx = self.random_state.randint(min_room, max_room)

        # Get the obstacles and free positions for this room from params
        obstacles = self.params.obstacles[room_idx]
        free_positions = self.params.free_positions[room_idx]

        # Sample positions for robot and goal separately
        robot_pos = sample_position(self.random_state, free_positions)
        goal_pos = sample_position(self.random_state, free_positions)

        # Randomly initialize robot orientation
        robot_angle = self.random_state.uniform(0, 2 * np.pi)

        # Initialize position history with robot's starting position in first slot, zeros elsewhere
        position_history = np.zeros((self.params.max_steps_in_episode, 2))
        position_history[0] = robot_pos

        # Create initial state
        self.state = EnvState(
            x=robot_pos[0],
            y=robot_pos[1],
            theta=robot_angle,
            goal_x=goal_pos[0],
            goal_y=goal_pos[1],
            steps=0,
            episode_done=False,
            room_idx=room_idx,
            obstacles=obstacles,
            lidar_distances=np.zeros(self.params.lidar_num_beams),
            lidar_collision_types=np.zeros(self.params.lidar_num_beams, dtype=np.int32),
            accumulated_reward=0.0,
            position_history=position_history,
        )

        # Update lidar readings for initial state
        lidar_distances, collision_types = simulate_lidar(
            self.state.x,
            self.state.y,
            self.state.theta,
            self.state.obstacles,
            self.state.goal_x,
            self.state.goal_y,
            self.params,
        )
        self.state.lidar_distances = lidar_distances
        self.state.lidar_collision_types = collision_types

        # Get initial observation
        obs = self._get_obs()
        info = {}

        return obs, info

    def _get_obs(self) -> np.ndarray:
        """Convert state to observation vector."""
        assert self.state is not None

        # Robot pose (x, y, sin, cos)
        pose = np.array([self.state.x, self.state.y, np.sin(self.state.theta), np.cos(self.state.theta)])

        # Goal position
        goal = np.array([self.state.goal_x, self.state.goal_y])

        # Goal in robot frame (distance and angle)
        dx, dy = self.state.goal_x - self.state.x, self.state.goal_y - self.state.y
        goal_distance = np.sqrt(dx**2 + dy**2)
        goal_angle = np.arctan2(dy, dx) - self.state.theta
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))  # Normalize to [-pi, pi]
        goal_relative = np.array([goal_distance, goal_angle])

        # Convert collision types to goal flag (1 for goal, 0 for wall/obstacle)
        lidar_goal = (self.state.lidar_collision_types == Collision.Goal).astype(np.float32)

        # Combine robot state, goal, and sensor readings
        return np.concatenate([pose, goal, goal_relative, self.state.lidar_distances, lidar_goal])

    def render(self):
        """Render the environment."""
        # This is just a placeholder - rendering would need to be implemented
        # with your preferred library (e.g., Pygame, matplotlib)
        if self.render_mode == "human":
            pass  # Implement human rendering
        elif self.render_mode == "rgb_array":
            return np.zeros((300, 300, 3))  # Placeholder array

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
