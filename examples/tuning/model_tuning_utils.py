"""Common utilities for hyperparameter tuning of RL Models."""


from typing import Callable
from models.critic import Critic
from criteria.ppo import PPO
from criteria.gaussian_policy import GaussianPolicy
import numpy as np
import flax.linen as nn
import optuna
from typing import Any, Callable, Dict
from pathlib import Path

from quicknav_jax.eval import evaluate_model

import json


import jax
from quicknav_jax import (
  RoomParams,
  generate_rooms,
  NavigationEnvParams,
  NavigationEnv
)



# Set up constants
ROOM_SEED = 42
EVAL_SEED = 77
N_TIMESTEPS = 2_000_000
N_ENVS = 512
N_EVAL_EPISODES = 10

BEST_PPO_PARAMS = {
    "learning_rate": 0.0004203802421088965,
    "num_minibatches": 128, # adjusted to fit model
    "num_steps": 512,
    "gae_lambda": 0.9265010993996222,
    "ent_coef": 0.00951169140356109,
    "clip_eps": 0.14550640064594372,
    "gamma": 0.9601333552614683
}

# Output directory
OUTPUT_DIR = Path("examples/temp/tuning_results")

# Type aliases
SearchSpaceFunc = Callable[[optuna.Trial], Dict[str, Any]]





def create_env() -> dict:
    """Create environment with fixed rooms for consistent evaluation"""
    room_key = jax.random.PRNGKey(ROOM_SEED)
    room_params = RoomParams(size=8.0, grid_size=16)
    obstacles, free_positions = generate_rooms(room_key, room_params)

    env_params = {
      "rooms": room_params,
      "obstacles": obstacles,
      "free_positions": free_positions,
      "lidar_fov": 90
    }

    return env_params



def create_config() -> dict:
    return {
        "env": NavigationEnv(),
        "total_timesteps": N_TIMESTEPS,
        "normalize_observations": True,
        "num_envs": N_ENVS,
        **BEST_PPO_PARAMS
    } 


def create_actor(model: nn.Module) -> GaussianPolicy:
    return GaussianPolicy(2, (np.array([-1., -1.]), np.array([1., 1.])), model) # type: ignore
    

def create_agent(model: Callable[[], nn.Module], mem_init, env_params: dict, config: dict) -> PPO:
    # Initialize the training algorithm parameters
    config = {
        # Pass our environment to the agent
        "env_params": NavigationEnvParams(
          memory_init=lambda: mem_init,
          **env_params,
        ),
        **config,
    }

    # Create the training algorithm agent from `rejax` library
    agent = PPO.create(**config)
    agent = agent.replace(
      actor=create_actor(model()),
      critic=Critic(model())
    )

    return agent




def create_objective(
    model_param_keys: list[str],
    model: Callable[[dict], nn.Module],
    mem_init: Callable[[dict], Any],
    get_search_space: SearchSpaceFunc
) -> Callable[[optuna.Trial], float]:
    """Create an objective function for the specified algorithm"""

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter tuning"""

        # Get hyperparameter search space based on algorithm
        params = get_search_space(trial)

        # Extract model parameters from the search space
        model_params = {
            k: v for k, v in params.items() if k in model_param_keys
        }

        agent = create_agent(
            lambda: model(model_params),
            mem_init=mem_init(model_params),
            env_params={
                **create_env(),
                **{k: v for k, v in params.items() if k not in model_param_keys},  # Exclude model params
            },
            config=create_config(),
        )

        print(f"\nTrial {trial.number} - Testing parameters: {trial.params}")

        train_fn = jax.jit(agent.train)

        # Set the seed for reproducibility
        rng = jax.random.PRNGKey(trial.number)  # Use trial number as seed for diversity

        # Train the model
        try:
            train_state, _ = train_fn(rng)

            # Evaluate the trained model
            evaluation = evaluate_model(
                agent=agent,
                train_state=train_state,
                seed=EVAL_SEED,
                n_eval_episodes=N_EVAL_EPISODES,
                render=False,
            )

            mean_steps = float(evaluation.steps.mean())
            mean_return = float(evaluation.returns.mean())
            print(f"Trial {trial.number} finished with mean return: {mean_return:.4f} in {mean_steps} mean steps")
            return mean_steps

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            raise optuna.exceptions.TrialPruned()

    return objective


def save_best_params(study: optuna.Study, model: str) -> None:
    """Save the best parameters from the study"""
    best_params = study.best_params

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    json_path = OUTPUT_DIR / f"best_{model}_params.json"
    txt_path = OUTPUT_DIR / f"best_{model}_params.txt"

    # Save best parameters as JSON file
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=4)

    # Also save as text file for easy reading
    with open(txt_path, "w") as f:
        f.write(f"Best {model.upper()} parameters (mean return: {study.best_value:.4f}):\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")

    print(f"\nBest parameters saved to {txt_path}")



def run_optimization(
    model_name: str,
    model_param_keys: list[str],
    model: Callable[[dict], nn.Module],
    mem_init: Callable[[dict], Any],
    search_space: SearchSpaceFunc,
    n_trials: int
) -> None:
    """Run the hyperparameter optimization"""
    

    # Create study name and storage path based on algorithm
    study_name = f"{model_name}_optimization"

    # Create database directory if it doesn't exist
    db_dir = OUTPUT_DIR / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    # Use absolute path for SQLite database
    db_path = db_dir / f"{model_name}_study.db"
    storage_name = f"sqlite:///{db_path.absolute()}"

    # Create or load a study
    sampler = optuna.samplers.TPESampler(seed=42)  # TPE sampler implements Bayesian optimization

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    # Run optimization
    print(f"Starting optimization of {model_name.upper()} with {n_trials} trials")
    print(f"Run `optuna-dashboard {storage_name}` to open the dashboard")

    # Create and optimize with algorithm-specific objective
    objective = create_objective(model_param_keys, model, mem_init, search_space)
    study.optimize(objective, n_trials=n_trials)

    # Print and save best parameters
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best mean return: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    save_best_params(study, model_name)
