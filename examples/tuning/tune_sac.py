"""Hyperparameter tuning for the SAC algorithm."""

import argparse
from typing import Any, Dict

import optuna
from rejax import SAC
from utils import run_optimization


def sac_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for SAC."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000, 500_000]),
        "polyak": trial.suggest_float("polyak", 0.9, 0.999),
        "target_entropy_ratio": trial.suggest_float("target_entropy_ratio", 0.5, 1.0),
    }


def main() -> None:
    """Main function to run the hyperparameter optimization for SAC."""
    parser = argparse.ArgumentParser(description="Tune SAC hyperparameters")
    parser.add_argument("--trials", "-t", type=int, default=200, help="Number of trials to run")
    args = parser.parse_args()

    # Run the optimization
    run_optimization(algorithm="sac", agent_class=SAC, search_space=sac_search_space, n_trials=args.trials)


if __name__ == "__main__":
    # To run: `uv examples/tuning/tune_sac.py`
    main()
