"""Hyperparameter tuning for the PPO algorithm."""

import argparse
from typing import Any, Dict

import optuna
from models.mlp import MLP
from model_tuning_utils import run_optimization


def mlp_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for PPO."""
    return {
        # Model params
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 3),
        "hidden_layer_size": trial.suggest_categorical("hidden_layer_size", [64, 128, 256]),
        # Rewards
        "step_penalty": trial.suggest_float("step_penalty", 0.01, 0.5),
        "progress_reward": trial.suggest_float("progress_reward", 0.1, 1.0),
        "cycling_penalty": trial.suggest_float("cycling_penalty", 0.001, 0.5),
    }


def main() -> None:
    """Main function to run the hyperparameter optimization for PPO."""
    parser = argparse.ArgumentParser(description="Tune MLP hyperparameters")
    parser.add_argument("--trials", "-t", type=int, default=100, help="Number of trials to run")
    args = parser.parse_args()

    # Run the optimization
    run_optimization(
        model_name="mlp",
        model_param_keys=["num_hidden_layers", "hidden_layer_size"],
        model=lambda params: MLP(
            hidden_size=[params["hidden_layer_size"]] * params["num_hidden_layers"],
        ),
        mem_init=lambda params: MLP.initialize_state(mem_len=32),
        search_space=mlp_search_space, n_trials=args.trials
    )


if __name__ == "__main__":
    # To run: `uv run examples/tuning/tune_mlp.py`
    main()
