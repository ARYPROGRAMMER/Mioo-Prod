import numpy as np
import random
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PersonaFeatureExtractor:
    """Extract normalized features from student personas for model input."""
    
    def __init__(self):
        self.math_subjects = [
            "Algebra", "Geometry", "Trigonometry", "Precalculus", 
            "Data Analysis", "Statistics", "Linear Algebra"
        ]
        self.grade_mapping = {
            "Grade 7": 0, "Grade 8": 1, "Grade 9": 2, 
            "Grade 10": 3, "Grade 11": 4, "Unknown": 0
        }
        self.learning_style_types = [
            "examples", "visual", "step-by-step", 
            "simple language", "analogies", "stories", 
            "bullet points", "interactive"
        ]
    
    def extract_features(self, persona: Dict[str, Any]) -> np.ndarray:
        """Convert a persona to a normalized feature vector."""
        features = []
        
        # Math mastery levels
        math_features = []
        for subject in self.math_subjects:
            level = persona.get("math_mastery", {}).get(subject, "Unknown")
            normalized_level = self.grade_mapping.get(level, 0) / 4.0  # Normalize to [0, 1]
            math_features.append(normalized_level)
        features.extend(math_features)
        
        # Learning style (one-hot encoded)
        learning_style = persona.get("learning_style", "").lower()
        learning_style_encoding = [0] * len(self.learning_style_types)
        for i, style_keyword in enumerate(self.learning_style_types):
            if style_keyword in learning_style:
                learning_style_encoding[i] = 1
        features.extend(learning_style_encoding)
        
        # SAT/ACT timeline - binary for now (0: not preparing, 1: preparing)
        timeline = 1 if persona.get("sat_act_timeline", "") == "0-2 years" else 0
        features.append(timeline)
        
        return np.array(features, dtype=np.float32)

class PersonaQNetwork(nn.Module):
    """Q-Network for persona-based content adaptation."""
    
    def __init__(self, input_size: int, output_size: int):
        super(PersonaQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PersonaRLAgent:
    """Reinforcement Learning agent that adapts content based on student personas."""
    
    def __init__(self, 
                 action_size: int,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.1,
                 batch_size: int = 32,
                 memory_size: int = 10000):
        self.feature_extractor = PersonaFeatureExtractor()
        self.input_size = len(self.feature_extractor.math_subjects) + \
                          len(self.feature_extractor.learning_style_types) + 1
        self.action_size = action_size
        
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        self.model = PersonaQNetwork(self.input_size, self.action_size)
        self.target_model = PersonaQNetwork(self.input_size, self.action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()
    
    def update_target_model(self):
        """Copy weights from model to target model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, persona: Dict[str, Any]) -> int:
        """Choose action based on persona features using epsilon-greedy policy."""
        state = self.feature_extractor.extract_features(persona)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values[0]).item()
    
    def replay(self) -> float:
        """Train the model from experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        minibatch = random.sample(self.memory, self.batch_size)
        loss_sum = 0.0
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.discount_factor * torch.max(
                        self.target_model(next_state_tensor)).item()
            
            current_q = self.model(state_tensor)
            target_f = current_q.clone().detach()
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = F.mse_loss(current_q, target_f)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss_sum / self.batch_size
    
    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_model()
