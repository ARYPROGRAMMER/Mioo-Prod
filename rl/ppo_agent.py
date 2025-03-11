import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional

# Constants
STATE_DIM = 11  # Updated to match the state vector in learning_system.py
ACTION_DIM = 5  # Output action dimensions (teaching strategies)
HIDDEN_LAYERS = [64, 32]
LEARNING_RATE = 0.001
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 10
BATCH_SIZE = 32

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[1], ACTION_DIM)
        )
        
    def forward(self, x):
        return torch.softmax(self.layers(x), dim=-1)
    
    def evaluate(self, state, action):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, entropy

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[1], 1)
        )
        
    def forward(self, x):
        return self.layers(x)

class PPOMemory:
    def __init__(self, batch_size=32):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_log_probs = []
        self.values = []
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, action_log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.action_log_probs.append(action_log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.values.clear()
    
    def get_batch(self):
        batch_size = min(len(self.states), self.batch_size)
        indices = random.sample(range(len(self.states)), batch_size)
        return (
            torch.stack([self.states[i] for i in indices]),
            torch.tensor([self.actions[i] for i in indices]),
            torch.tensor([self.rewards[i] for i in indices]),
            torch.stack([self.next_states[i] for i in indices]),
            torch.tensor([self.action_log_probs[i] for i in indices]),
            torch.tensor([self.values[i] for i in indices])
        )
    
    def __len__(self):
        return len(self.states)

class PPOAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, load_path=None):
        self.policy_net = PolicyNetwork()
        self.value_net = ValueNetwork()
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
        self.memory = PPOMemory(BATCH_SIZE)
        
        # Load pre-trained models if path is provided
        if load_path:
            self.load_model(load_path)
    
    def select_action(self, state, user_state=None):
        """Select action with user preference consideration"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
            
        with torch.no_grad():
            probs = self.policy_net(state)
            
            # Adjust probabilities based on user preferences
            if user_state and user_state.get("preferred_style"):
                preferred_idx = self._get_preferred_strategy_idx(user_state["preferred_style"])
                if preferred_idx is not None:
                    probs[0][preferred_idx] *= 1.2  # Boost preferred strategy probability
                    probs = torch.softmax(probs, dim=-1)  # Renormalize
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value_net(state)
            
        return action.item(), log_prob.item(), value.item()
    
    def compute_reward(self, 
                      knowledge_gain, 
                      engagement_delta, 
                      response_quality, 
                      exploration_score, 
                      emotional_improvement,
                      user_feedback=None,
                      user_state=None):
        """Enhanced reward calculation with user preferences"""
        # Base reward calculation
        base_reward = (
            3.0 * knowledge_gain +
            2.0 * engagement_delta +
            1.0 * response_quality +
            0.5 * exploration_score +
            1.5 * emotional_improvement
        )
        
        # User preference multiplier
        preference_multiplier = 1.0
        if user_state and user_state.get("interests"):
            # Check if response matched user interests
            interest_alignment = any(
                interest.lower() in user_state.get("last_response", "").lower() 
                for interest in user_state.get("interests", [])
            )
            if interest_alignment:
                preference_multiplier *= 1.2
        
        # Explicit feedback multiplier
        if user_feedback:
            feedback_multiplier = 1.2 if user_feedback == "like" else 0.8
            return base_reward * preference_multiplier * feedback_multiplier
        
        return base_reward * preference_multiplier

    async def train(self):
        """Enhanced PPO training with feedback integration"""
        if len(self.memory) < self.memory.batch_size:
            return None
            
        states, actions, rewards, next_states, old_log_probs, values = self.memory.get_batch()
        
        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for _ in range(PPO_EPOCHS):
            # Get current policy probabilities and values
            current_probs = self.policy_net(states)
            current_values = self.value_net(states)
            
            # Calculate policy and value losses with clipping
            policy_loss = self._compute_policy_loss(current_probs, old_log_probs, rewards, values)
            value_loss = self._compute_value_loss(current_values, values, rewards)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        self.memory.clear()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_reward": rewards.mean().item()
        }
    
    def store_experience(self, state, action, reward, next_state, log_prob, value):
        """Store transition in memory buffer"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        self.memory.add(state, action, reward, next_state, log_prob, value)
        
    def save_model(self, path):
        """Save model to disk"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)
        
    def load_model(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

    def _get_preferred_strategy_idx(self, preferred_style: str) -> Optional[int]:
        """Map preferred style to strategy index"""
        style_map = {
            "detailed": 0,
            "concise": 1,
            "interactive": 2,
            "analogy-based": 3,
            "step-by-step": 4
        }
        return style_map.get(preferred_style)
