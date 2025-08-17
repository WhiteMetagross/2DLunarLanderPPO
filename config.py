# This file contains the configuration for the PPO algorithm and the 2D Lunar Lander Continuous environment.

import os
import torch.nn as nn

class Config:
    #The configuration for the 2D Lunar Lander Continuous environment.
    ENV_NAME = "LunarLander-v3"
    ENV_KWARGS = {"continuous": True}
    N_ENVS = 16
    
    TOTAL_TIMESTEPS = 3_200_000
    EVAL_EPISODES = 10
    
    #The tuned hyperparameters for the PPO algorithm.
    PPO_PARAMS = {
        "learning_rate": 0.00020008809615992198,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 11,
        "gamma": 0.997151183012025,
        "gae_lambda": 0.9803222253501476,
        "clip_range": 0.2433693624803069,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.0013291011108410845,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "tensorboard_log": "./logs/",
        "policy_kwargs": {
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
            "activation_fn": nn.Tanh
        },
        "verbose": 1,
        "seed": 17,
        "device": "cpu"
    }
    
    #The paths for saving the model and the logs.
    MODEL_SAVE_PATH = "models/2DLunarLanderContinuousPPO"
    LOG_DIR = "logs/"
    
    @staticmethod
    def create_directories():
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)