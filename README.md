# GridWorld RL Project

A reinforcement learning project that trains a DQN (Deep Q-Network) agent to navigate and solve tasks in a GridWorld environment.

## Overview

The agent learns to navigate a grid world with walls and markers, performing actions like moving, turning, picking/placing markers, and finishing tasks. The environment loads different scenarios from JSON files containing grid layouts, wall positions, and start/goal configurations.

## Key Components

- `main.py` - Training loop and DQN implementation
- `model.py` - Neural network architecture and replay memory
- `envs/env_project.py` - GridWorld environment implementation
- `utils.py` - Plotting utilities for training metrics
- `datasets/` - Training and validation data (easy, medium difficulty levels)

## Actions

The agent can perform 6 actions:
- Move forward
- Turn left/right
- Pick/put markers
- Finish task

## Usage

Run training:
```bash
python main.py
```

The script will:
1. Load random task configurations from the dataset
2. Train the DQN agent using experience replay
3. Save the trained model as `model.pth`
4. Generate training plots (episode duration, loss, rewards)

## Requirements

- PyTorch
- Gym
- NumPy
- Matplotlib
- Pygame (for rendering)

## Dataset

Contains JSON files with grid configurations including:
- Grid size and walls
- Agent start/goal positions and orientations  
- Marker locations (pre/post states) 