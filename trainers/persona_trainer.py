import os
import json
import random
import numpy as np
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from collections import Counter  # Add this missing import

from personas.student_personas import PersonaManager, StudentPersona 
from models.rl_model import PersonaRLAgent, PersonaFeatureExtractor

class ContentAction:
    """Represents a content adaptation action based on personas."""
    EXPLANATION_STYLES = [
        "Detailed step-by-step", 
        "Visual aid focused", 
        "Real-world application", 
        "Simplified language", 
        "Bullet point summary",
        "Interactive questioning",
        "Analogy-based",
        "Story context"
    ]
    
    DIFFICULTY_LEVELS = [
        "Basic",
        "Standard",
        "Advanced",
        "Challenge"
    ]
    
    CONTENT_THEMES = [
        "Sports",
        "Video Games",
        "Technology",
        "Food",
        "Music",
        "Fashion",
        "Entertainment",
        "Transportation",
        "Social Media",
        "Theme Parks" 
    ]
    
    def __init__(self, 
                 style_idx: int, 
                 difficulty_idx: int, 
                 theme_idx: int):
        self.style = self.EXPLANATION_STYLES[style_idx % len(self.EXPLANATION_STYLES)]
        self.difficulty = self.DIFFICULTY_LEVELS[difficulty_idx % len(self.DIFFICULTY_LEVELS)]
        self.theme = self.CONTENT_THEMES[theme_idx % len(self.CONTENT_THEMES)]
    
    @classmethod
    def from_action_id(cls, action_id: int) -> 'ContentAction':
        """Create a ContentAction from an integer action ID."""
        n_styles = len(cls.EXPLANATION_STYLES)
        n_difficulties = len(cls.DIFFICULTY_LEVELS)
        
        style_idx = action_id % n_styles
        difficulty_idx = (action_id // n_styles) % n_difficulties
        theme_idx = action_id // (n_styles * n_difficulties)
        
        return cls(style_idx, difficulty_idx, theme_idx)
    
    @classmethod
    def action_space_size(cls) -> int:
        """Get the total number of possible actions."""
        return (len(cls.EXPLANATION_STYLES) * 
                len(cls.DIFFICULTY_LEVELS) * 
                len(cls.CONTENT_THEMES))
    
    def __repr__(self) -> str:
        return (f"ContentAction(style='{self.style}', "
                f"difficulty='{self.difficulty}', "
                f"theme='{self.theme}')")


class PersonaEnvironment:
    """Simulated environment for training the RL model."""
    
    def __init__(self, personas: List[StudentPersona]):
        self.personas = personas
        self.feature_extractor = PersonaFeatureExtractor()
        self.current_persona = None
        self.reset()
        
    def reset(self):
        """Reset the environment with a random persona."""
        self.current_persona = random.choice(self.personas)
        return self.feature_extractor.extract_features(self.current_persona.to_dict())
    
    def step(self, action_id: int) -> tuple:
        """Take an action and return (next_state, reward, done)."""
        action = ContentAction.from_action_id(action_id)
        reward = self._calculate_reward(action)
        done = True  # One-step episode for simplicity
        next_state = self.feature_extractor.extract_features(self.current_persona.to_dict())
        return next_state, reward, done
    
    def _calculate_reward(self, action: ContentAction) -> float:
        """Calculate reward based on how well the action matches the persona."""
        persona_dict = self.current_persona.to_dict()
        reward = 0.0
        
        # Reward for matching learning style preference
        learning_style = persona_dict.get("learning_style", "").lower()
        # Enhanced learning style matching with more nuanced rewards
        if "example" in learning_style and "step-by-step" in action.style.lower():
            reward += 1.0
        elif "example" in learning_style and "real-world application" in action.style.lower():
            reward += 0.8
        elif "visual" in learning_style and "visual" in action.style.lower():
            reward += 1.0
        elif "step-by-step" in learning_style and "step-by-step" in action.style.lower():
            reward += 1.0
        elif "simple language" in learning_style and "simplified" in action.style.lower():
            reward += 1.0
        elif "concise" in learning_style and "simplified" in action.style.lower():
            reward += 0.8
        elif "analogi" in learning_style and "analogy" in action.style.lower():
            reward += 1.0
        elif "stories" in learning_style and "story" in action.style.lower():
            reward += 1.0
        elif "bullet point" in learning_style and "bullet" in action.style.lower():
            reward += 1.0
        elif "interactive" in learning_style and "interactive" in action.style.lower():
            reward += 1.0
        elif "practice problem" in learning_style and ("step-by-step" in action.style.lower() or 
                                                     "detailed" in action.style.lower()):
            reward += 0.9
        
        # Reward for matching theme preference
        likes_theme = persona_dict.get("likes_theme", "").lower()
        # Enhanced theme matching with weighted rewards for exact vs. related matches
        theme_match_found = False
        for theme_word in action.theme.lower().split():
            if theme_word in likes_theme:
                reward += 1.0
                theme_match_found = True
                break
        
        # Partial credit for related themes
        if not theme_match_found:
            related_themes = {
                "sports": ["video games", "entertainment"],
                "video games": ["technology", "entertainment"],
                "technology": ["video games", "social media"],
                "food": ["social media"],
                "music": ["entertainment", "technology"],
                "streaming": ["technology", "entertainment"],
                "fashion": ["social media", "design"],
                "theme park": ["entertainment"],
                "cars": ["technology", "transportation"],
                "photography": ["social media", "technology"]
            }
            
            # Check if user theme is related to action theme
            for user_theme, related in related_themes.items():
                if user_theme in likes_theme and action.theme.lower() in related:
                    reward += 0.5
                    break
        
        # Reward for appropriate difficulty based on math mastery
        math_mastery = persona_dict.get("math_mastery", {})
        avg_grade = 0
        if math_mastery:
            grades = []
            for subject, grade in math_mastery.items():
                if grade.startswith("Grade "):
                    try:
                        grades.append(int(grade.split(" ")[1]))
                    except ValueError:
                        pass
            if grades:
                avg_grade = sum(grades) / len(grades)
        
        # Map grade to appropriate difficulty
        appropriate_difficulty = ""
        if avg_grade <= 8:
            appropriate_difficulty = "Basic"
        elif avg_grade <= 9:
            appropriate_difficulty = "Standard"
        elif avg_grade <= 10:
            appropriate_difficulty = "Advanced" 
        else:
            appropriate_difficulty = "Challenge"
            
        # Enhanced difficulty matching with graduated rewards
        if action.difficulty == appropriate_difficulty:
            reward += 1.0
        elif abs(self._difficulty_level(action.difficulty) - self._difficulty_level(appropriate_difficulty)) == 1:
            # One level off - partial reward
            reward += 0.5
        elif abs(self._difficulty_level(action.difficulty) - self._difficulty_level(appropriate_difficulty)) == 2:
            # Two levels off - small reward
            reward += 0.2
        else:
            # Completely wrong difficulty - slight penalty
            reward -= 0.3
        
        return reward

    def _difficulty_level(self, difficulty: str) -> int:
        """Convert difficulty string to numeric level."""
        difficulty_map = {
            "Basic": 1,
            "Standard": 2,
            "Advanced": 3,
            "Challenge": 4
        }
        return difficulty_map.get(difficulty, 2)  # Default to Standard


class PersonaTrainer:
    """Trains the RL model on student personas."""
    
    def __init__(self, persona_data: str, model_dir: str = None):
        self.persona_manager = PersonaManager()
        self.persona_manager.load_from_json(persona_data)
        
        action_size = ContentAction.action_space_size()
        self.agent = PersonaRLAgent(
            action_size=action_size, 
            learning_rate=0.001,
            batch_size=32,
            epsilon=1.0,         # Start with full exploration
            epsilon_decay=0.997  # Slower decay for more thorough exploration
        )
        
        self.environment = PersonaEnvironment(self.persona_manager.personas)
        self.model_dir = model_dir or "."
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self, episodes: int = 1000, update_target_every: int = 10, progress_callback=None):
        """Train the RL agent on persona data."""
        best_avg_reward = -float('inf')
        rewards_history = []
        
        # Training metrics tracking
        action_distribution = {}
        persona_rewards = {persona.name: [] for persona in self.persona_manager.personas}
        
        # Add training log file
        log_file = os.path.join(self.model_dir, "training_log.txt")
        with open(log_file, "w") as f:
            f.write("Episode,Reward,AvgReward,Epsilon,Loss\n")
        
        for episode in tqdm(range(episodes)):
            state = self.environment.reset()
            current_persona = self.environment.current_persona
            
            action = self.agent.choose_action(current_persona.to_dict())
            next_state, reward, done = self.environment.step(action)
            
            # Track metrics
            action_key = str(ContentAction.from_action_id(action))
            action_distribution[action_key] = action_distribution.get(action_key, 0) + 1
            persona_rewards[current_persona.name].append(reward)
            
            self.agent.remember(state, action, reward, next_state, done)
            loss = self.agent.replay()
            
            rewards_history.append(reward)
            
            # Update target network periodically
            if episode % update_target_every == 0:
                self.agent.update_target_model()
            
            # Save the best model
            if episode > 100:  # Wait for some training to occur
                avg_reward = np.mean(rewards_history[-100:])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.agent.save_model(os.path.join(self.model_dir, "best_model.pth"))
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if rewards_history else 0
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.4f}, Epsilon: {self.agent.epsilon:.4f}")
            
            # Call progress callback if provided
            if progress_callback and episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if rewards_history else 0
                progress_callback(
                    episode=episode,
                    reward=reward,
                    avg_reward=avg_reward
                )
            
            # Enhanced logging
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if rewards_history else 0
                log_line = f"{episode},{reward:.4f},{avg_reward:.4f},{self.agent.epsilon:.4f},{loss:.6f}\n"
                with open(log_file, "a") as f:
                    f.write(log_line)
                    
                print(f"Episode: {episode}")
                print(f"Reward: {reward:.4f}")
                print(f"Avg Reward: {avg_reward:.4f}") 
                print(f"Epsilon: {self.agent.epsilon:.4f}")
                print(f"Loss: {loss:.6f}")
                print("-" * 40)
        
        # Save final model
        self.agent.save_model(os.path.join(self.model_dir, "final_model.pth"))
        
        # Save training metrics
        self._save_training_metrics(rewards_history, action_distribution, persona_rewards)
        
        return rewards_history
    
    def _save_training_metrics(self, rewards_history, action_distribution, persona_rewards):
        """Save training metrics to file."""
        metrics = {
            "average_reward": float(np.mean(rewards_history)),
            "max_reward": float(np.max(rewards_history)),
            "min_reward": float(np.min(rewards_history)),
            "final_epsilon": float(self.agent.epsilon),
            "action_distribution": {k: v for k, v in sorted(
                action_distribution.items(), key=lambda item: item[1], reverse=True)},
            "persona_performance": {
                name: {
                    "avg_reward": float(np.mean(rewards)) if rewards else 0,
                    "count": len(rewards)
                } for name, rewards in persona_rewards.items()
            }
        }
        
        with open(os.path.join(self.model_dir, "training_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def evaluate(self, num_evaluations: int = 100):
        """Evaluate the trained model."""
        self.agent.epsilon = 0  # Disable exploration
        total_reward = 0
        action_distribution = {}
        persona_matches = {}
        
        print("Evaluating model on personas...")
        for _ in range(num_evaluations):
            state = self.environment.reset()
            persona = self.environment.current_persona
            
            action = self.agent.choose_action(persona.to_dict())
            content_action = ContentAction.from_action_id(action)
            
            _, reward, _ = self.environment.step(action)
            total_reward += reward
            
            # Track chosen actions
            action_key = str(content_action)
            action_distribution[action_key] = action_distribution.get(action_key, 0) + 1
            
            # Track persona-specific performance
            if persona.name not in persona_matches:
                persona_matches[persona.name] = {"rewards": [], "actions": []}
            persona_matches[persona.name]["rewards"].append(reward)
            persona_matches[persona.name]["actions"].append(action_key)
            
            print(f"Persona: {persona.name} | Action: {content_action} | Reward: {reward:.2f}")
        
        print(f"\nAverage Reward: {total_reward / num_evaluations:.4f}")
        print("\nAction Distribution:")
        for action, count in sorted(action_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"{action}: {count} times ({count/num_evaluations*100:.1f}%)")
        
        print("\nPersona-Specific Performance:")
        for name, data in persona_matches.items():
            avg_reward = sum(data["rewards"]) / len(data["rewards"]) if data["rewards"] else 0
            most_common_action = Counter(data["actions"]).most_common(1)[0][0] if data["actions"] else "None"
            print(f"{name}: Avg Reward: {avg_reward:.2f}, Most Common Action: {most_common_action}")
        
        # Save evaluation results
        evaluation_results = {
            "average_reward": total_reward / num_evaluations,
            "action_distribution": {k: v for k, v in sorted(
                action_distribution.items(), key=lambda item: item[1], reverse=True)},
            "persona_performance": {
                name: {
                    "avg_reward": sum(data["rewards"]) / len(data["rewards"]) if data["rewards"] else 0,
                    "most_common_action": Counter(data["actions"]).most_common(1)[0][0] if data["actions"] else "None",
                    "action_counts": dict(Counter(data["actions"]))
                }
                for name, data in persona_matches.items()
            }
        }
        
        with open(os.path.join(self.model_dir, "evaluation_results.json"), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
