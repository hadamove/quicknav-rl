from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from rejax import PPO, SAC, TD3

import quicknav_utils.env_vis as env_vis

Agent = PPO | SAC | TD3


@dataclass
class EvaluationResult:
    returns: jnp.ndarray
    """Returns for each episode."""
    rendered_frames: List[List[env_vis.Frame]] | None = None
    """Rendered frames for each episode if `render=True`."""


def evaluate_model(
    agent: Agent,
    train_state: TrainState,
    seed: int,
    n_eval_episodes: int = 20,
    render: bool = False,
) -> EvaluationResult:
    """
    Evaluate the trained model on the test environment.

    Args:
        agent: The trained model to evaluate.
        train_state: The train state of the model, returned from `Algorithm.train`.
        seed: The seed for the test environment.
        n_eval_episodes: The number of episodes to evaluate the model on.
        render: Whether to render the environment.

    Returns:
        EvaluationResult containing the returns and rendered frames.
    """

    act = jax.jit(agent.make_act(train_state))
    key = jax.random.PRNGKey(seed)
    reset_key, action_key = jax.random.split(key)

    frames = []
    returns = jnp.zeros(n_eval_episodes)

    for i in range(n_eval_episodes):
        # Split the reset key to get a new reset key for this episode and the next reset key
        reset_key, next_reset_key = jax.random.split(reset_key)

        # Reset the environment - choose a new random room, start, and goal
        obs, state = agent.env.reset(reset_key, agent.env_params)

        done = False
        episode_return = 0
        episode_frames = []

        while not done:
            # Render current state to frame
            if render:
                frame = env_vis.render_frame(state, agent.env_params)
                episode_frames.append(frame)

            # Choose action and step the environment
            action_key, act_subkey, step_subkey = jax.random.split(action_key, 3)
            action = act(obs, act_subkey)
            obs, state, reward, done, _ = agent.env.step(step_subkey, state, action, agent.env_params)
            episode_return += reward

        returns = returns.at[i].set(episode_return)
        reset_key = next_reset_key
        frames.append(episode_frames)

    print(f"Evaluation finished, mean return: {returns.mean()}")

    return EvaluationResult(
        returns=returns,
        rendered_frames=frames if render else None,
    )
