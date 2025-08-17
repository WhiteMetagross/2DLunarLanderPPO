#This program is used to train the PPO model for the 2D Lunar Lander Continuous environment.
#The model is trained for a total of 10 million timesteps.

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import numpy as np
from config import Config

def main():
    Config.create_directories()
    
    print(f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Training with {Config.N_ENVS} parallel environments")
    
    #Create the vectorized environment.
    env = make_vec_env(
        Config.ENV_NAME, 
        n_envs=Config.N_ENVS, 
        seed=Config.PPO_PARAMS["seed"],
        env_kwargs=Config.ENV_KWARGS,
        vec_env_cls=SubprocVecEnv
    )
    
    #Create the evaluation environment.
    eval_env = make_vec_env(
        Config.ENV_NAME,
        n_envs=1,
        env_kwargs=Config.ENV_KWARGS,
        vec_env_cls=SubprocVecEnv
    )
    
    #Create the evaluation callback.
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=50000, #Evaluate every 50,000 timesteps.
        best_model_save_path=Config.MODEL_SAVE_PATH,
        log_path=Config.LOG_DIR,
        verbose=1,
        n_eval_episodes=5,
        deterministic=True
    )
    
    model = PPO("MlpPolicy", env, **Config.PPO_PARAMS)
    
    print(f"Starting training for {Config.TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Model will be saved to: {Config.MODEL_SAVE_PATH}")
    
    #Train the model.
    try:
        model.learn(
            total_timesteps=Config.TOTAL_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True
        )
        
        model.save(f"{Config.MODEL_SAVE_PATH}_final")
        print(f"Training completed! Model saved to {Config.MODEL_SAVE_PATH}_final.zip")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        model.save(f"{Config.MODEL_SAVE_PATH}_interrupted")
        print(f"Model saved to {Config.MODEL_SAVE_PATH}_interrupted.zip")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()