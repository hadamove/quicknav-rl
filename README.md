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

## Quick Start

### Using the NumPy Implementation

```python
import numpy as np
from quicknav_numpy import NavigationEnv, NavigationEnvParams, RoomParams, generate_rooms

# Generate room layouts
rng = np.random.default_rng(42)
room_params = RoomParams(num_rooms=16)
obstacles, free_positions = generate_rooms(rng, room_params)

# Create environment parameters with the generated rooms
env_params = NavigationEnvParams(
    rooms=room_params,
    obstacles=obstacles,
    free_positions=free_positions,
)

# Create the environment
env = NavigationEnv(params=env_params, seed=42)

# Run an episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished with reward: {total_reward}")
```

### Using the JAX Implementation

```python
import jax
import jax.numpy as jnp
from quicknav_jax import NavigationEnv, NavigationEnvParams

# Set up environment
env = NavigationEnv()
params = env.default_params

# Reset the environment
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
obs, state = env.reset(subkey, params)

# Run an episode
done = False
total_reward = 0.0

while not done:
    key, subkey = jax.random.split(key)
    action = jax.random.uniform(subkey, shape=(2,), minval=-1.0, maxval=1.0)
    
    obs, state, reward, done, info = env.step(subkey, state, action, params)
    total_reward += reward

print(f"Episode finished with reward: {total_reward}")
```

## Environment Parameters

The environment can be customized using the `NavigationEnvParams` class:

- **Robot parameters**: wheel_base, max_wheel_speed, robot_radius, dt
- **Sensor parameters**: lidar_num_beams, lidar_fov, lidar_max_distance
- **Reward parameters**: goal_tolerance, step_penalty, collision_penalty, goal_reward
- **Episode parameters**: max_steps_in_episode

## License

MIT