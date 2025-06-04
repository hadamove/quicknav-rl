# QuickNav - Robot Navigation in JAX

This repository contains a JAX implementation of the robot navigation problem. The goal is to navigate a robot in a 2D grid world with obstacles, using reinforcement learning techniques.

## Demo

![Demo](./media/demo.gif)

## Installation

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. Afterwards, create a new virtual environment and install the dependencies using `uv`:
```bash
uv sync
```

> Alternatively, you can use `poetry` or anything that works with `pyproject.toml` (but `uv` is superior!)

If your device has a GPU with CUDA support, you can enable it during the environment setup by running:
```bash
uv sync --extra cuda
```


To format the code, you can run:
```bash
uvx ruff format
```

## Pre-commit hooks

Pre-commit hook is set-up automatically strip output cells from Jupyter notebooks before committing so that the 10 MB gifs from notebook are no longer accidentally uploaded to git 💀

To install the pre-commit hooks (already included in dev dependencies):
```bash
pre-commit install
```

## Running

To get started, checkout `examples/ppo.ipynb` for a quick introduction to training the agent, evaluation and visualization of the environment.
Choose python from `.venv` as interpreter and it should work out of the box.
