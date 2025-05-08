"""Example script for using the QuickNav environment with NumPy backend."""

from typing import Optional, Tuple

import numpy as np

from quicknav_numpy import NavigationEnv, NavigationEnvParams, RoomParams, generate_rooms


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


def run_random_agent(env: NavigationEnv, num_episodes: int = 3, max_steps: int = 100) -> None:
    """
    Run a random agent in the environment for a few episodes.

    Args:
        env: Navigation environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0

        print(f"Episode {episode + 1}/{num_episodes}")

        steps_taken = 0
        for step in range(max_steps):
            steps_taken = step + 1
            # Random action
            action = env.action_space.sample()

            # Take a step
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Check if episode is done
            if terminated or truncated:
                break

        print(f"  Steps: {steps_taken}, Total reward: {episode_reward:.2f}")

        # Small separation between episodes
        print()


if __name__ == "__main__":
    # Set up the environment
    env, _ = generate_and_setup_environment(seed=42)

    # Run random agent
    run_random_agent(env)

    # Clean up
    env.close()
