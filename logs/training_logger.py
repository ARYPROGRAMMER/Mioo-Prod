import logging
import os
import json
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure file handler
        self.log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.logger = logging.getLogger("PersonaTrainer")
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Metrics tracking
        self.rewards_history = []
        self.loss_history = []
        self.persona_performance = {}
        
    def log_training_step(self, episode: int, reward: float, loss: float, 
                         persona_name: str, action_taken: str):
        """Log training step information"""
        self.rewards_history.append(reward)
        self.loss_history.append(loss)
        
        if persona_name not in self.persona_performance:
            self.persona_performance[persona_name] = []
        self.persona_performance[persona_name].append(reward)
        
        self.logger.info(
            f"Episode {episode} - Reward: {reward:.4f}, Loss: {loss:.4f}, "
            f"Persona: {persona_name}, Action: {action_taken}"
        )
        
    def save_training_metrics(self):
        """Save training metrics to JSON"""
        metrics = {
            "average_reward": float(np.mean(self.rewards_history)),
            "average_loss": float(np.mean(self.loss_history)),
            "persona_performance": {
                name: {
                    "avg_reward": float(np.mean(rewards)),
                    "num_interactions": len(rewards)
                }
                for name, rewards in self.persona_performance.items()
            }
        }
        
        metrics_file = os.path.join(self.log_dir, "training_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
    def plot_training_curves(self):
        """Generate training visualization plots"""
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(self.rewards_history, alpha=0.6, label="Raw rewards")
        # Add smoothed curve
        window = 100
        smoothed = np.convolve(self.rewards_history, 
                             np.ones(window)/window, 
                             mode='valid')
        plt.plot(smoothed, label="Smoothed rewards")
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 1, 2)
        plt.plot(self.loss_history)
        plt.title("Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_curves.png"))
        plt.close()
