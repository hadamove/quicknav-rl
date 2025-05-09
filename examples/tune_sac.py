#!/usr/bin/env python3
import json
import os

import jax
import optuna
from rejax import SAC

from quicknav_jax import NavigationEnv, NavigationEnvParams, RoomParams, generate_rooms
from quicknav_jax.eval import evaluate_model

# Set up constants
ROOM_SEED = 42
EVAL_SEED = 77
N_TRIALS = 100
MAX_TIMESTEPS = 500_000
N_EVAL_EPISODES = 10
STUDY_NAME = "sac_optimization"
STORAGE_NAME = "sqlite:///sac_study.db"


def create_env():
    """Create environment with fixed rooms for consistent evaluation"""
    room_key = jax.random.PRNGKey(ROOM_SEED)
    room_params = RoomParams(size=8.0, grid_size=16)
    obstacles, free_positions = generate_rooms(room_key, room_params)
    env_params = NavigationEnvParams(
        rooms=room_params, obstacles=obstacles, free_positions=free_positions, lidar_fov=90
    )
    return NavigationEnv(), env_params


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for SAC hyperparameter tuning"""
    # Create environment (same for all trials for consistency)
    env, env_params = create_env()

    # Sample hyperparameters - focus on the most impactful ones for SAC
    config = {
        "env": env,
        "env_params": env_params,
        "total_timesteps": MAX_TIMESTEPS,
        # Key hyperparameters to tune
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000, 500000]),
        "polyak": trial.suggest_float("polyak", 0.9, 0.999),
        "target_entropy_ratio": trial.suggest_float("target_entropy_ratio", 0.5, 1.0),
        # Fixed parameters
        "normalize_observations": True,
        "num_envs": 512,
        "agent_kwargs": {
            "hidden_layer_sizes": (128, 128),
        },
    }

    print(f"\nTrial {trial.number} - Testing parameters: {trial.params}")

    # Create and train the agent
    agent = SAC.create(**config)
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


def save_best_params(study: optuna.Study) -> None:
    """Save the best parameters from the study"""
    best_params = study.best_params

    # Create 'results' directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Save best parameters as JSON file
    with open("results/best_sac_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Also save as text file for easy reading
    with open("results/best_sac_params.txt", "w") as f:
        f.write(f"Best SAC parameters (mean return: {study.best_value:.4f}):\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")

    print("\nBest parameters saved to results/best_sac_params.txt")


def main():
    """Main function to run the hyperparameter optimization"""
    # Create or load a study
    sampler = optuna.samplers.TPESampler(seed=42)  # TPE sampler implements Bayesian optimization

    os.makedirs("results", exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    # Run optimization
    print(f"Starting optimization with {N_TRIALS} trials")
    print(f"Run `optuna-dashboard {STORAGE_NAME}` to open the dashboard")
    study.optimize(objective, n_trials=N_TRIALS)

    # Print and save best parameters
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best mean return: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    save_best_params(study)


if __name__ == "__main__":
    main()
