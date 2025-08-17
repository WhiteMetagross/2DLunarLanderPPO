#This program is used to visualise the training data from tensorboard.
#It plots the training metrics and policy metrics from the tensorboard data.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import numpy as np
from datetime import datetime

#This class is used to visualise the training data from tensorboard.
class TrainingVisualizer:
    def __init__(self, log_dir="logs/"):
        self.log_dir = log_dir
        self.data = {}
    
    #This function is used to load the tensorboard data from the event file.
    def load_tensorboard_data(self, event_file_path):
        event_acc = EventAccumulator(event_file_path)
        event_acc.Reload()
        
        data = {}
        
        scalar_tags = event_acc.Tags()['scalars']
        for tag in scalar_tags:
            scalar_events = event_acc.Scalars(tag)
            timestamps = [event.wall_time for event in scalar_events]
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            
            data[tag] = {
                'timestamps': timestamps,
                'steps': steps,
                'values': values
            }
        
        return data
    
    #This function is used to find all the event files in the log directory.
    def find_event_files(self):
        event_files = []
        for root, dirs, files in os.walk(self.log_dir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_files.append(os.path.join(root, file))
        return event_files
    
    #This function is used to load all the data from the event files.
    def load_all_data(self):
        event_files = self.find_event_files()
        if not event_files:
            print(f"No tensorboard event files found in {self.log_dir}")
            return False
        
        print(f"Found {len(event_files)} event files")
        
        all_data = {}
        for event_file in event_files:
            try:
                data = self.load_tensorboard_data(event_file)
                run_name = os.path.basename(os.path.dirname(event_file))
                all_data[run_name] = data
                print(f"Loaded data from {run_name}")
            except Exception as e:
                print(f"Error loading {event_file}: {e}")
        
        self.data = all_data
        return len(all_data) > 0
    
    #This function is used to plot the training metrics.
    def plot_training_metrics(self, save_plots=True, show_plots=True):
        if not self.data:
            print("No data loaded. Run load_all_data() first.")
            return
        
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Training Metrics', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.data)))
        
        for i, (run_name, run_data) in enumerate(self.data.items()):
            color = colors[i]
            
            if 'rollout/ep_rew_mean' in run_data:
                steps = run_data['rollout/ep_rew_mean']['steps']
                values = run_data['rollout/ep_rew_mean']['values']
                axes[0, 0].plot(steps, values, label=run_name, color=color, linewidth=2)
            
            if 'rollout/ep_len_mean' in run_data:
                steps = run_data['rollout/ep_len_mean']['steps']
                values = run_data['rollout/ep_len_mean']['values']
                axes[0, 1].plot(steps, values, label=run_name, color=color, linewidth=2)
            
            if 'train/learning_rate' in run_data:
                steps = run_data['train/learning_rate']['steps']
                values = run_data['train/learning_rate']['values']
                axes[1, 0].plot(steps, values, label=run_name, color=color, linewidth=2)
            
            if 'train/loss' in run_data:
                steps = run_data['train/loss']['steps']
                values = run_data['train/loss']['values']
                axes[1, 1].plot(steps, values, label=run_name, color=color, linewidth=2)
        
        axes[0, 0].set_title('Episode Reward Mean', fontweight='bold')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Episode Length Mean', fontweight='bold')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Learning Rate', fontweight='bold')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Training Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    #This function is used to plot the policy metrics.
    def plot_policy_metrics(self, save_plots=True, show_plots=True):
        if not self.data:
            print("No data loaded.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Policy Metrics', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.data)))
        
        for i, (run_name, run_data) in enumerate(self.data.items()):
            color = colors[i]
            
            if 'train/entropy_loss' in run_data:
                steps = run_data['train/entropy_loss']['steps']
                values = run_data['train/entropy_loss']['values']
                axes[0, 0].plot(steps, values, label=run_name, color=color, linewidth=2)
            
            if 'train/policy_gradient_loss' in run_data:
                steps = run_data['train/policy_gradient_loss']['steps']
                values = run_data['train/policy_gradient_loss']['values']
                axes[0, 1].plot(steps, values, label=run_name, color=color, linewidth=2)
            
            if 'train/value_loss' in run_data:
                steps = run_data['train/value_loss']['steps']
                values = run_data['train/value_loss']['values']
                axes[1, 0].plot(steps, values, label=run_name, color=color, linewidth=2)
            
            if 'train/clip_fraction' in run_data:
                steps = run_data['train/clip_fraction']['steps']
                values = run_data['train/clip_fraction']['values']
                axes[1, 1].plot(steps, values, label=run_name, color=color, linewidth=2)
        
        axes[0, 0].set_title('Entropy Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Policy Gradient Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Value Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Clip Fraction', fontweight='bold')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Fraction')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"policy_metrics_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    #This function is used to plot the reward distribution.
    def plot_reward_distribution(self, save_plots=True, show_plots=True):
        if not self.data:
            print("No data loaded.")
            return
        
        plt.figure(figsize=(12, 6))
        
        for run_name, run_data in self.data.items():
            if 'rollout/ep_rew_mean' in run_data:
                rewards = run_data['rollout/ep_rew_mean']['values']
                plt.hist(rewards, bins=50, alpha=0.7, label=f'{run_name} (μ={np.mean(rewards):.1f})', density=True)
        
        plt.title('Episode Reward Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Reward')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reward_distribution_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    #This function is used to plot the smoothed rewards.
    def plot_smoothed_rewards(self, window_size=100, save_plots=True, show_plots=True):
        if not self.data:
            print("No data loaded.")
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.data)))
        
        for i, (run_name, run_data) in enumerate(self.data.items()):
            if 'rollout/ep_rew_mean' in run_data:
                steps = np.array(run_data['rollout/ep_rew_mean']['steps'])
                rewards = np.array(run_data['rollout/ep_rew_mean']['values'])
                
                smoothed_rewards = pd.Series(rewards).rolling(window=min(window_size, len(rewards))).mean()
                
                plt.plot(steps, rewards, alpha=0.3, color=colors[i])
                plt.plot(steps, smoothed_rewards, label=f'{run_name} (smoothed)', 
                        color=colors[i], linewidth=2)
        
        plt.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Success Threshold (200)')
        plt.title(f'Episode Rewards (Smoothed with window={window_size})', fontsize=14, fontweight='bold')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"smoothed_rewards_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    #This function is used to generate a summary report.
    def generate_summary_report(self):
        if not self.data:
            print("No data loaded.")
            return
        
        print("=" * 80)
        print("TRAINING SUMMARY REPORT")
        print("=" * 80)
        
        for run_name, run_data in self.data.items():
            print(f"\nRun: {run_name}")
            print("-" * 40)
            
            if 'rollout/ep_rew_mean' in run_data:
                rewards = run_data['rollout/ep_rew_mean']['values']
                steps = run_data['rollout/ep_rew_mean']['steps']
                
                print(f"Total Steps: {max(steps):,}")
                print(f"Final Reward: {rewards[-1]:.2f}")
                print(f"Best Reward: {max(rewards):.2f}")
                print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
                
                success_rewards = [r for r in rewards if r >= 200]
                success_rate = len(success_rewards) / len(rewards) * 100
                print(f"Success Rate (≥200): {success_rate:.1f}%")
                
                if success_rewards:
                    first_success_idx = next(i for i, r in enumerate(rewards) if r >= 200)
                    print(f"First Success at Step: {steps[first_success_idx]:,}")
            
            if 'rollout/ep_len_mean' in run_data:
                lengths = run_data['rollout/ep_len_mean']['values']
                print(f"Final Episode Length: {lengths[-1]:.1f}")
                print(f"Mean Episode Length: {np.mean(lengths):.1f}")

def main():
    #This is the main function that is used to run the visualizer.
    parser = argparse.ArgumentParser(description="Visualize PPO training logs")
    parser.add_argument("--log-dir", type=str, default="logs/", help="Directory containing tensorboard logs.")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots, only save them.")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots, only display them.")
    parser.add_argument("--smooth-window", type=int, default=100, help="Window size for smoothing rewards.")
    
    args = parser.parse_args()
    
    visualizer = TrainingVisualizer(args.log_dir)
    
    if not visualizer.load_all_data():
        print("Failed to load any data. Exiting...")
        return
    
    show_plots = not args.no_show
    save_plots = not args.no_save
    
    print("Generating training metrics plot...")
    visualizer.plot_training_metrics(save_plots, show_plots)
    
    print("Generating policy metrics plot...")
    visualizer.plot_policy_metrics(save_plots, show_plots)
    
    print("Generating reward distribution plot...")
    visualizer.plot_reward_distribution(save_plots, show_plots)
    
    print("Generating smoothed rewards plot...")
    visualizer.plot_smoothed_rewards(args.smooth_window, save_plots, show_plots)
    
    print("Generating summary report...")
    visualizer.generate_summary_report()

if __name__ == "__main__":
    main()