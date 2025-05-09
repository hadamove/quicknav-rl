"""Common utilities for hyperparameter tuning of RL algorithms."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Type, Union

import jax
import optuna
from rejax import PPO, SAC

from quicknav_jax import NavigationEnv, NavigationEnvParams, RoomParams, generate_rooms
from quicknav_jax.eval import evaluate_model

# Set up constants
ROOM_SEED = 42
EVAL_SEED = 77
N_TIMESTEPS = 500_000
N_EVAL_EPISODES = 10

# Output directory
OUTPUT_DIR = Path("examples/temp/tuning_results")

# Type aliases
AgentClass = Union[Type[PPO], Type[SAC]]
SearchSpaceFunc = Callable[[optuna.Trial], Dict[str, Any]]


def create_env() -> tuple[NavigationEnv, NavigationEnvParams]:
    """Create environment with fixed rooms for consistent evaluation"""
    room_key = jax.random.PRNGKey(ROOM_SEED)
    room_params = RoomParams(size=8.0, grid_size=16)
    obstacles, free_positions = generate_rooms(room_key, room_params)
    env_params = NavigationEnvParams(
        rooms=room_params, obstacles=obstacles, free_positions=free_positions, lidar_fov=90
    )
    return NavigationEnv(), env_params


def create_objective(agent_class: AgentClass, get_search_space: SearchSpaceFunc) -> Callable[[optuna.Trial], float]:
    """Create an objective function for the specified algorithm"""

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter tuning"""
        # Create environment (same for all trials for consistency)
        env, env_params = create_env()

        # Get hyperparameter search space based on algorithm
        params = get_search_space(trial)

        # Create full config
        config = {
            "env": env,
            "env_params": env_params,
            "total_timesteps": N_TIMESTEPS,
            "normalize_observations": True,
            "num_envs": 512,
            "agent_kwargs": {
                "hidden_layer_sizes": (128, 128),
            },
            **params,
        }

        print(f"\nTrial {trial.number} - Testing parameters: {trial.params}")

        # Create and train the agent
        agent = agent_class.create(**config)
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

            mean_return = float(evaluation.returns.mean())
            print(f"Trial {trial.number} finished with mean return: {mean_return:.4f}")
            return mean_return

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            raise optuna.exceptions.TrialPruned()

    return objective


def save_best_params(study: optuna.Study, algorithm: str) -> None:
    """Save the best parameters from the study"""
    best_params = study.best_params

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    json_path = OUTPUT_DIR / f"best_{algorithm}_params.json"
    txt_path = OUTPUT_DIR / f"best_{algorithm}_params.txt"

    # Save best parameters as JSON file
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=4)

    # Also save as text file for easy reading
    with open(txt_path, "w") as f:
        f.write(f"Best {algorithm.upper()} parameters (mean return: {study.best_value:.4f}):\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")

    print(f"\nBest parameters saved to {txt_path}")


def run_optimization(algorithm: str, agent_class: AgentClass, search_space: SearchSpaceFunc, n_trials: int) -> None:
    """Run the hyperparameter optimization"""
    # Create study name and storage path based on algorithm
    study_name = f"{algorithm}_optimization"

    # Create database directory if it doesn't exist
    db_dir = OUTPUT_DIR / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    # Use absolute path for SQLite database
    db_path = db_dir / f"{algorithm}_study.db"
    storage_name = f"sqlite:///{db_path.absolute()}"

    # Create or load a study
    sampler = optuna.samplers.TPESampler(seed=42)  # TPE sampler implements Bayesian optimization

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    # Run optimization
    print(f"Starting optimization of {algorithm.upper()} with {n_trials} trials")
    print(f"Run `optuna-dashboard {storage_name}` to open the dashboard")

    # Create and optimize with algorithm-specific objective
    objective = create_objective(agent_class, search_space)
    study.optimize(objective, n_trials=n_trials)

    # Print and save best parameters
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best mean return: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    save_best_params(study, algorithm)
