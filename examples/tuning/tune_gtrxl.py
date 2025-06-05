"""Hyperparameter tuning for the PPO algorithm."""

import argparse
from typing import Any, Dict

import optuna
from models.gtrxl import GTrXL
from model_tuning_utils import run_optimization


def gtrxl_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for PPO."""
    return {
        # Model params
        "embed_dim": trial.suggest_categorical("embed_dim", [8, 16]),
        "head_dim": trial.suggest_int("head_dim", 1, 8),
        "layer_num": trial.suggest_int("layer_num", 2, 5),
        # Rewards
        "step_penalty": trial.suggest_float("step_penalty", 0.01, 0.5),
        "progress_reward": trial.suggest_float("progress_reward", 0.1, 1.0),
        "cycling_penalty": trial.suggest_float("cycling_penalty", 0.001, 0.5),
    }


def main() -> None:
    """Main function to run the hyperparameter optimization for PPO."""
    parser = argparse.ArgumentParser(description="Tune GTrXL hyperparameters")
    parser.add_argument("--trials", "-t", type=int, default=100, help="Number of trials to run")
    args = parser.parse_args()

    # Run the optimization
    run_optimization(
        model_name="gtrxl",
        model_param_keys=["layer_num", "head_dim", "embed_dim"],
        model=lambda params: GTrXL(
            head_dim=params["head_dim"],
            embedding_dim=params["embed_dim"],
            head_num=2,
            mlp_num=2,
            layer_num=params["layer_num"],
            memory_len=32
        ),
        mem_init=lambda params: GTrXL.init_memory(
            memory_len=32,
            embedding_dim=params["embed_dim"],
            layer_num=params["layer_num"]
        ),
        search_space=gtrxl_search_space, n_trials=args.trials
    )


if __name__ == "__main__":
    # To run: `uv run examples/tuning/tune_gtrxl.py`
    main()
