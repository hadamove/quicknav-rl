# QuickNav RL Environment

This repository contains a robot navigation environment for reinforcement learning, with two implementations:

1. **JAX Implementation** (`quicknav_jax`): Highly optimized for parallel simulations and compatible with the gymnax interface.
2. **NumPy Implementation** (`quicknav_numpy`): Sequential implementation compatible with the gymnasium interface.

The environment simulates a differential drive robot navigating in a room with obstacles, using lidar for sensing. The goal is to reach a target position while avoiding obstacles.

## Features

- Differential drive robot physics
- Lidar-based obstacle sensing
- Procedurally generated room layouts
- Customizable environment parameters
- Support for both JAX and NumPy backends

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quicknav-rl.git
cd quicknav-rl

# Install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Running

To get started, checkout `examples/intro.ipynb` for a quick introduction to training the agent, evaluation and visualization of the environment.
Choose python from `.venv` as interpreter and it should work out of the box.

## TODO

- [x] Randomized start and goal positions
- [x] Randomized obstacles (walls) generation
    - [x] Simulate lidar sensor to detect walls and avoid them
    - [x] Add negative reward for hitting walls
    - [x] Add walls to visualization
- [x] "Better" walls - something that actually resembles a real world room not just random rectangles 😆
- [x] Better observations for the policy (currently position & rotation of the robot + position of the goal)
    - [x] Maybe include rotation and/or distance of the goal
- [x] High priority: Fix lidar beams, currently they originate from the center of the robot, but they should originate from the perimeter of the "circle" collider representing the robot
    - [x] This is important for the agent to learn to avoid walls, without it detects "free" space although it's close to a wall
- [ ] Try different reward weights
    - [ ] Penalization for going back (i.e. backtracking)
- [ ] W&B integration -- see https://github.com/keraJLi/rejax/blob/main/examples/wandb_integration.py
    - [ ] Essentially just copy/reimplement a `wandb_callback` like in that file (might be outdated)
- [ ] Add support for decimation/frame skipping in the environment (policy predicts 1 action, but environment takes N steps with that action)
    - Making decisions too frequently makes it harder to relate which actions led to which outcome. Longer actions or actions that persist allow the agent to plan from a higher level at a cost of delayed response.
    - This will allow us to lower `env_params.dt` to 0.01 or even lower, which will make the simulation look smoother
- [ ] Experiment with different algorithms (currently we use PPO, good alternatives are PQN, SAC, TD3)
- [ ] Experiment with a different NN architecture (currently we use default 2 hidden layers (64, 64), see `ppo.config` in `intro.ipynb`)