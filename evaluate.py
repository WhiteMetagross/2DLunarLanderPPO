#This program is used to evaluate the performance of the trained model on the 2DLunarLanderContinuous-v3  environment.*
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import argparse
import os
from config import Config
import matplotlib.pyplot as plt

#This function is used to evaluate the performance of the trained model on the 2DLunarLanderContinuous-v3  environment.*
def evaluate_model(model_path, n_episodes=10, render=True, plot=False):
    if not os.path.exists(model_path):
        available_models = []
        if os.path.exists("models/"):
            for file in os.listdir("models/"):
                if file.endswith(".zip"):
                    available_models.append(os.path.join("models", file))
       
        if available_models:
            print(f"Model not found at {model_path}")
            print("Available models:")
            for model in available_models:
                print (f"  {model}")
            return
        else:
            print("No trained models found. Please run train.py first.")
            return
   
    env = gym.make(Config.ENV_NAME, render_mode="human" if render else None, **Config.ENV_KWARGS)
   
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")
   
    episode_rewards = []
    episode_lengths = []
   
    print(f"Evaluating for {n_episodes} episodes...")
   
    #Evaluate the model for n_episodes episodes.*
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
       
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
           
            if render:
                env.render()
       
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
       
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
   
    env.close()
   
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
   
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
   
    success_rate = np.sum(np.array(episode_rewards) >= 200) / len(episode_rewards)
    print(f"Success Rate (Reward >= 200): {success_rate:.1%}")
    
    #Plot the evaluation rewards over episodes.
    if plot:
        plt.figure(figsize=(10, 6))
        episodes = range(1, len(episode_rewards) + 1)
        plt.plot(episodes, episode_rewards, 'b-', marker='o', linewidth=2, markersize=5)
        plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean Reward: {mean_reward:.2f}')
        plt.xlabel('Episode Number')
        plt.ylabel('Episode Reward')
        plt.title('Evaluation Rewards Over Episodes')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        #Save the plot.
        plot_filename = f"evaluation_rewards_{os.path.basename(model_path).replace('.zip', '')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {plot_filename}")
        plt.show()

def main():
    #This function is used to parse the command line arguments.*
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model on LunarLander")
    parser.add_argument("--model", type=str, default=r"models\2DLunarLanderContinuousPPO_final.zip",
                        help="Path to the trained model.")
    parser.add_argument("--episodes", type=int, default=Config.EVAL_EPISODES,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering.")
    parser.add_argument("--plot", action="store_true",
                        help="Plot and save evaluation rewards over episodes.")
   
    args = parser.parse_args()
   
    evaluate_model(args.model, args.episodes, not args.no_render, args.plot)

if __name__ == "__main__":
    main()