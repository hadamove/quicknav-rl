"""Example script for using the QuickNav environment with NumPy backend."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from quicknav_numpy import NavigationEnv, NavigationEnvParams, RoomParams, generate_rooms
from quicknav_utils import Theme, render_frame, save_gif


def generate_and_setup_environment(
    seed: int = 42,
    num_rooms: int = 16,  # Smaller number for quicker example setup
    render_mode: Optional[str] = None,
) -> Tuple[NavigationEnv, NavigationEnvParams]:
    """
    Generate rooms and initialize the environment.

    Args:
        seed: Random seed for reproducibility
        num_rooms: Number of room layouts to generate
        render_mode: Rendering mode (None, 'human', 'rgb_array')

    Returns:
        Tuple of (environment, environment parameters)
    """
    # Create RNG
    rng = np.random.default_rng(seed)

    # Generate room layouts
    room_params = RoomParams(num_rooms=num_rooms)
    obstacles_batch, free_positions_batch = generate_rooms(rng, room_params)

    # Create environment parameters with the generated rooms
    env_params = NavigationEnvParams(
        rooms=room_params,
        obstacles=obstacles_batch,
        free_positions=free_positions_batch,
    )

    # Create environment
    env = NavigationEnv(params=env_params, render_mode=render_mode, seed=seed)

    return env, env_params


def run_random_agent(
    env: NavigationEnv,
    num_episodes: int = 3,
    max_steps: int = 100,
    save_visualization: bool = True,
    output_path: Optional[Path] = None,
    img_width: int = 400,
    img_height: int = 400,
    fps: float = 15.0,
) -> List[np.ndarray]:
    """
    Run a random agent in the environment for a few episodes.

    Args:
        env: Navigation environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        save_visualization: Whether to save frames for visualization
        output_path: Path to save the visualization GIF
        img_width: Width of output frames in pixels
        img_height: Height of output frames in pixels
        fps: Frames per second for the output GIF

    Returns:
        List of rendered frames if save_visualization is True, otherwise empty list
    """
    # Create output directory if specified
    if output_path is not None:
        os.makedirs(output_path.parent, exist_ok=True)

    # Custom theme for visualization
    theme = Theme(
        background=(240, 240, 245),
        obstacle=(100, 100, 110),
        goal=(63, 176, 0),
        path=(100, 149, 237),  # Cornflower blue
    )

    all_frames = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_frames = []

        print(f"Episode {episode + 1}/{num_episodes}")

        steps_taken = 0
        for step in range(max_steps):
            steps_taken = step + 1

            # Save the current state as a frame
            if save_visualization and env.state is not None:
                frame = render_frame(env.state, env.params, img_width=img_width, img_height=img_height, theme=theme)
                episode_frames.append(frame)

            # Random action
            action = env.action_space.sample()

            # Take a step
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Check if episode is done
            if terminated or truncated:
                # Add one more frame for the final state
                if save_visualization and env.state is not None:
                    frame = render_frame(env.state, env.params, img_width=img_width, img_height=img_height, theme=theme)
                    episode_frames.append(frame)
                break

        print(f"  Steps: {steps_taken}, Total reward: {episode_reward:.2f}")
        print()

        all_frames.extend(episode_frames)

    # Save all frames as a GIF if requested
    if save_visualization and output_path is not None and all_frames:
        save_gif(all_frames, output_path, duration_per_frame=1.0 / fps)
        print(f"Visualization saved to {output_path}")

    return all_frames


if __name__ == "__main__":
    # Set up the environment
    env, _ = generate_and_setup_environment(seed=42)

    # Output path for visualization
    output_dir = Path("./output")
    output_path = output_dir / "quicknav_numpy_demo.gif"

    # Run random agent with visualization
    run_random_agent(
        env,
        num_episodes=2,  # Smaller number of episodes for demo
        max_steps=100,
        save_visualization=True,
        output_path=output_path,
        img_width=400,
        img_height=400,
        fps=15.0,
    )

    # Clean up
    env.close()
