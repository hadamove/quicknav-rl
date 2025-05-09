"""Hyperparameter tuning for the TD3 algorithm."""

import argparse
from typing import Any, Dict

import optuna
from rejax import TD3
from utils import run_optimization


def td3_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for TD3."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "exploration_noise": trial.suggest_float("exploration_noise", 0.1, 0.5),
        "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000, 500_000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "polyak": trial.suggest_float("polyak", 0.9, 0.999),
        "target_noise": trial.suggest_float("target_noise", 0.1, 0.4),
        "policy_delay": trial.suggest_categorical("policy_delay", [1, 2, 3, 4]),
    }


def main() -> None:
    """Main function to run the hyperparameter optimization for TD3."""
    parser = argparse.ArgumentParser(description="Tune TD3 hyperparameters")
    parser.add_argument("--trials", "-t", type=int, default=200, help="Number of trials to run")
    args = parser.parse_args()

    # Run the optimization
    run_optimization(algorithm="td3", agent_class=TD3, search_space=td3_search_space, n_trials=args.trials)


if __name__ == "__main__":
    # To run: `python examples/tuning/tune_td3.py`
    main() 