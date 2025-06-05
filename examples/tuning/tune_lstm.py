"""Hyperparameter tuning for the PPO algorithm."""

import argparse
from typing import Any, Dict

import optuna
from models.lstm import LSTMMultiLayer
from model_tuning_utils import run_optimization


def lstm_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for PPO."""
    return {
        # Model params
        "layers": trial.suggest_int("layers", 2, 10),
        "depth": trial.suggest_int("depth", 5, 16),
        # Rewards
        "step_penalty": trial.suggest_float("step_penalty", 0.01, 0.5),
        "progress_reward": trial.suggest_float("progress_reward", 0.1, 1.0),
        "cycling_penalty": trial.suggest_float("cycling_penalty", 0.001, 0.5),
    }


def main() -> None:
    """Main function to run the hyperparameter optimization for PPO."""
    parser = argparse.ArgumentParser(description="Tune LSTM hyperparameters")
    parser.add_argument("--trials", "-t", type=int, default=100, help="Number of trials to run")
    args = parser.parse_args()

    # Run the optimization
    run_optimization(
        model_name="lstm",
        model_param_keys=["depth", "layers"],
        model=lambda params: LSTMMultiLayer(
            d_model=params["depth"], n_layers=params["layers"]
        ),
        mem_init=lambda params: LSTMMultiLayer.initialize_state(d_model=params["depth"], n_layers=params["layers"]),
        search_space=lstm_search_space, n_trials=args.trials
    )


if __name__ == "__main__":
    # To run: `uv run examples/tuning/tune_lstm.py`
    main()
