"""Hyperparameter tuning for the PPO algorithm."""

import argparse
from typing import Any, Dict

import optuna
from rejax import PPO
from utils import N_TRIALS, run_optimization


def ppo_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for PPO."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "num_minibatches": trial.suggest_categorical("num_minibatches", [16, 32, 64, 128, 256]),
        "num_steps": trial.suggest_categorical("num_steps", [64, 128, 256, 512]),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "clip_eps": trial.suggest_float("clip_eps", 0.1, 0.3),
    }


def main() -> None:
    """Main function to run the hyperparameter optimization for PPO."""
    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters")
    parser.add_argument("--trials", "-t", type=int, default=N_TRIALS, help="Number of trials to run")
    args = parser.parse_args()

    # Run the optimization
    run_optimization(algorithm="ppo", agent_class=PPO, search_space=ppo_search_space, n_trials=args.trials)


if __name__ == "__main__":
    # To run: `uv examples/tuning/tune_ppo.py`
    main()
