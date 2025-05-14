## Hyperparameter tuning results

To inspect the results saved in db files in this directory, run:

```bash
optuna-dashboard examples/tuning/results/PPO_study.db
```

(or replace `PPO_study.db` with the name of the db file you want to inspect)

`optuna-dashboard` should be installed within the virtual environment you used to run the tuning.

## Figures

### PPO

|      |      |
| ---- | ---- |
| ![PPO history](./img/ppo_history.png) | ![PPO parameter imjportance](./img/ppo_importance.png) |
| ![PPO parallel coordinate](./img/ppo_parallel.png) | |

### TD3

|      |      |
| ---- | ---- |
| ![TD3 history](./img/td3_history.png) | ![TD3 parameter importance](./img/td3_importance.png) |
| ![TD3 parallel coordinate](./img/td3_parallel.png) | |

### SAC

|      |      |
| ---- | ---- |
| ![SAC history](./img/sac_history.png) | ![SAC parameter importance](./img/sac_importance.png) |
| ![SAC parallel coordinate](./img/sac_parallel.png) | |
