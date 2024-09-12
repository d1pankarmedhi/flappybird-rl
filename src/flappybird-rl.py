from collections import deque

import flappy_bird_gymnasium
import gym
import gym_pygame
import gymnasium
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import RLModel, reinforce


def main():
    # game environment
    env = gymnasium.make("FlappyBird-v0", use_lidar=False)
    s_size = env.observation_space.shape[0]  # state spaces
    a_size = env.action_space.n  # action spaces

    # hyperparameters
    flappybird_hyperparameters = {
        "h_size": 64,
        "n_training_episodes": 30000,
        "n_evaluation_episodes": 10,
        "max_t": 10000,
        "gamma": 0.99,
        "lr": 1e-4,
        "env_id": "FlappyBird-v0",
        "state_space": s_size,
        "action_space": a_size,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flappybird_policy = RLModel(
        flappybird_hyperparameters["state_space"],
        flappybird_hyperparameters["action_space"],
        flappybird_hyperparameters["h_size"],
    ).to(device)
    flappybird_optimizer = optim.Adam(
        flappybird_policy.parameters(), lr=flappybird_hyperparameters["lr"]
    )

    # training model
    scores = reinforce(
        flappybird_policy,
        flappybird_optimizer,
        flappybird_hyperparameters["n_training_episodes"],
        flappybird_hyperparameters["max_t"],
        flappybird_hyperparameters["gamma"],
        1000,
        env,
    )

    # Save Model
    torch.save(flappybird_policy, "model.pt")


if __name__ == "__main__":
    main()
