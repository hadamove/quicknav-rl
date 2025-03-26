# Robot Navigation in JAX

This repository contains a JAX implementation of the robot navigation problem. The goal is to navigate a robot in a 2D grid world with obstacles, using reinforcement learning techniques.

## Installation

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. Afterwards, create a new virtual environment and install the dependencies using `uv`:
```bash
uv sync
```

>Alternatively, you can use `poetry` or anything that works with `pyproject.toml` (but `uv` is ultra superior!)

## Running

To get started, checkout `src/notebooks/intro.ipynb` for a quick introduction to training the agent, evaluation and visualization of the environment.
