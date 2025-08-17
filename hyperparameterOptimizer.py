#This program is used to optimize the hyperparameters of the PPO algorithm.
#This program uses the optuna library to optimize the hyperparameters.

import optuna
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from config import Config

#This function is used to optimize the hyperparameters.
def objective(trial):
    #The hyperparameters to optimize.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 3, 15)
    gamma = trial.suggest_float("gamma", 0.99, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    
    net_arch_size = trial.suggest_categorical("net_arch", [128, 256, 512])
    
    #Create the vectorized environment.
    env = make_vec_env(
        Config.ENV_NAME,
        n_envs=8,
        env_kwargs=Config.ENV_KWARGS,
        vec_env_cls=SubprocVecEnv
    )
    
    #Create the model.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs={
            "net_arch": dict(pi=[net_arch_size, net_arch_size], vf=[net_arch_size, net_arch_size]),
            "activation_fn": nn.Tanh
        },
        device="cpu",
        verbose=0
    )
    
    #Train the model with 200,000 timesteps.
    model.learn(total_timesteps=200_000)
    
    #Evaluate the model.
    eval_env = gym.make(Config.ENV_NAME, **Config.ENV_KWARGS)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    
    env.close()
    eval_env.close()
    
    return mean_reward

if __name__ == "__main__":
    #Create the study.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    print("Best hyperparameters:")
    print(study.best_params)
    print(f"Best mean reward: {study.best_value:.2f}")