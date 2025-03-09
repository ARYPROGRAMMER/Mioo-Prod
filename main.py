from fastapi import FastAPI, HTTPException, Depends, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import settings

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
from bson import json_util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from motor.motor_asyncio import AsyncIOMotorClient
import logging
import random
from emotion_handler import EmotionDetector, EmotionalResponseAdjuster
from memory_manager import MemoryManager
import re
import copy
from user_management import router as user_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Personalized AI Tutor")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.ai_tutor_db
# Collections
users_collection = db.users
chat_history_collection = db.chat_history
user_states_collection = db.user_states

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def serialize_datetime(dt):
    """Convert datetime to ISO format string"""
    if isinstance(dt, str):
        return dt
    return dt.isoformat() if dt else None

def parse_datetime(dt_str):
    """Parse ISO format string to datetime"""
    if isinstance(dt_str, datetime):
        return dt_str
    return datetime.fromisoformat(dt_str) if dt_str else None

# Pydantic models with JSON serialization
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    class Config:
        json_encoders = {
            datetime: serialize_datetime
        }
        arbitrary_types_allowed = True
    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        if d.get('timestamp'):
            d['timestamp'] = serialize_datetime(d['timestamp'])
        return d

class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    teaching_strategy: Dict[str, Any]

class UserState(BaseModel):
    user_id: str
    knowledge_level: float = 0.5
    engagement: float = 0.5
    interests: List[str] = Field(default_factory=list)
    recent_topics: List[str] = Field(default_factory=list)
    performance: float = 0.5
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    class Config:
        json_encoders = {
            datetime: serialize_datetime
        }
        arbitrary_types_allowed = True

# State representation for RL
STATE_DIM = 10
ACTION_DIM = 5

# RL Networks
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.zeros(1, ACTION_DIM))
    def forward(self, state):
        action_mean = self.network(state)
        action_probs = nn.functional.softmax(action_mean, dim=-1)
        return action_probs
    def evaluate(self, state, action):
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, dist_entropy
    def get_action(self, state):
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, state):
        return self.network(state)

# PPO specific parameters
PPO_EPOCHS = 10
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 32
BUFFER_SIZE = 1000

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

# Initialize networks and memory
policy_net = PolicyNetwork()
value_net = ValueNetwork()
policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
memory = PPOMemory(BATCH_SIZE)

# User state management (in-memory)
user_states = {}

# Define teaching strategies
strategies = [
    {"style": "detailed", "complexity": "high", "examples": "many"},
    {"style": "concise", "complexity": "low", "examples": "few"},
    {"style": "interactive", "complexity": "medium", "examples": "some"},
    {"style": "analogy-based", "complexity": "medium", "examples": "many"},
    {"style": "step-by-step", "complexity": "adjustable", "examples": "many"}
]

async def sync_with_db(user_id: str):
    """Synchronize in-memory state with database"""
    try:
        if user_id in user_states:
            state = user_states[user_id].copy()
            # Ensure all required fields exist
            if 'chat_history' not in state:
                state['chat_history'] = []
            if 'interests' not in state:
                state['interests'] = []
            if 'recent_topics' not in state:
                state['recent_topics'] = []
            # Update last_updated timestamp
            state['last_updated'] = datetime.utcnow()
            await user_states_collection.update_one(
                {"user_id": user_id},
                {"$set": state},
                upsert=True
            )
    except Exception as e:
        logger.error(f"Error syncing with database: {str(e)}")
        raise HTTPException(status_code=500, detail="Database synchronization failed")

async def load_from_db(user_id: str):
    """Load user state from database"""
    try:
        state_doc = await user_states_collection.find_one({"user_id": user_id})
        if state_doc:
            state_doc.pop('_id', None)
            user_states[user_id] = state_doc
        else:
            # Initialize new user state
            user_states[user_id] = UserState(user_id=user_id).dict()
    except Exception as e:
        logger.error(f"Error loading from database: {str(e)}")
        raise HTTPException(status_code=500, detail="Database loading failed")

async def save_chat_history(user_id: str, message: Message):
    """Save chat message to database"""
    try:
        message_doc = message.dict()
        message_doc["user_id"] = user_id
        await chat_history_collection.insert_one(message_doc)
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save chat history")

# Enhanced State Space Encoding
class StateEncoder:
    def __init__(self, state_dim=10):
        self.state_dim = state_dim
    def encode_state(self, user_state: Dict) -> torch.Tensor:
        # Knowledge components (3 dimensions)
        knowledge_level = float(user_state.get('knowledge_level', 0.5))
        recent_performance = float(user_state.get('performance', 0.5))
        topic_mastery = len(user_state.get('recent_topics', [])) / 10.0
        
        # Engagement components (3 dimensions)
        current_engagement = float(user_state.get('engagement', 0.5))
        interaction_frequency = min(1.0, len(user_state.get('chat_history', [])) / 50.0)
        interest_diversity = len(user_state.get('interests', [])) / 10.0
        
        # Learning style components (4 dimensions)
        avg_response_length = min(1.0, sum(len(msg.get('content', '')) 
                                         for msg in user_state.get('chat_history', [])[-5:]) / 2500)
        preferred_complexity = self._calculate_preferred_complexity(user_state)
        learning_speed = self._calculate_learning_speed(user_state)
        interaction_style = self._calculate_interaction_style(user_state)
        
        return torch.tensor([
            knowledge_level,
            recent_performance,
            topic_mastery,
            current_engagement,
            interaction_frequency,
            interest_diversity,
            avg_response_length,
            preferred_complexity,
            learning_speed,
            interaction_style
        ], dtype=torch.float32)

    def _calculate_preferred_complexity(self, user_state: Dict) -> float:
        chat_history = user_state.get('chat_history', [])
        if not chat_history:
            return 0.5
        # Analyze last 5 successful interactions
        complexity_scores = []
        for msg in chat_history[-10:]:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Estimate complexity based on sentence length and vocabulary
                words = content.split()
                avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
                complexity = min(1.0, (avg_word_length - 3) / 5)
                complexity_scores.append(complexity)
        return sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.5

    def _calculate_learning_speed(self, user_state: Dict) -> float:
        knowledge_history = user_state.get('knowledge_history', [])
        if len(knowledge_history) < 2:
            return 0.5
        # Calculate average knowledge gain per interaction
        gains = [k2 - k1 for k1, k2 in zip(knowledge_history[:-1], knowledge_history[1:])]
        avg_gain = sum(gains) / len(gains)
        return min(1.0, max(0.0, avg_gain * 5))  # Normalize to [0,1]

    def _calculate_interaction_style(self, user_state: Dict) -> float:
        chat_history = user_state.get('chat_history', [])
        if not chat_history:
            return 0.5
        # Analyze user messages for interaction style
        user_messages = [msg for msg in chat_history if msg.get('role') == 'user']
        if not user_messages:
            return 0.5
        # Calculate ratio of questions to statements
        question_count = sum(1 for msg in user_messages if '?' in msg.get('content', ''))
        return question_count / len(user_messages)

# Enhanced Reward Function
class RewardCalculator:
    def __init__(self):
        self.engagement_history = []
        self.knowledge_history = []
        self.reward_scale = 1.0
        self.emotional_impact_history = []
        
    def calculate_reward(
        self,
        old_state: Dict[str, float],
        new_state: Dict[str, float],
        interaction_metrics: Dict[str, float]
    ) -> float:
        # Track histories
        self.engagement_history.append(new_state['engagement'])
        self.knowledge_history.append(new_state['knowledge_level'])
        
        # Base rewards
        knowledge_reward = self._calculate_knowledge_reward(old_state, new_state)
        engagement_reward = self._calculate_engagement_reward(old_state, new_state)
        quality_reward = self._calculate_quality_reward(interaction_metrics)
        exploration_reward = self._calculate_exploration_reward(new_state)
        emotional_reward = self._calculate_emotional_reward(old_state, new_state)
        
        # Combine rewards with dynamic weighting
        total_reward = (
            3.0 * knowledge_reward +
            2.0 * engagement_reward +
            1.0 * quality_reward +
            0.5 * exploration_reward +
            1.5 * emotional_reward  # Significant weight for emotional impact
        )
        
        # Normalize reward using adaptive scaling
        self.update_reward_scale(total_reward)
        normalized_reward = total_reward / self.reward_scale
        
        return float(np.clip(normalized_reward, -5.0, 5.0))

    def _calculate_emotional_reward(self, old_state: Dict, new_state: Dict) -> float:
        """Calculate reward based on emotional improvement"""
        # Get emotional context data
        old_emotions = old_state.get("emotional_context", {})
        new_emotions = new_state.get("emotional_context", {})
        
        # Get sentiment trends
        old_sentiments = old_emotions.get("sentiment_trend", [])
        new_sentiments = new_emotions.get("sentiment_trend", [])
        
        # If we don't have enough history, return neutral reward
        if not old_sentiments or not new_sentiments:
            return 0.0
            
        # Calculate average sentiment change
        old_avg = sum(old_sentiments[-3:]) / max(1, len(old_sentiments[-3:]))
        new_avg = sum(new_sentiments[-3:]) / max(1, len(new_sentiments[-3:]))
        sentiment_change = new_avg - old_avg
        
        # Check for specific emotional improvements
        old_frustration_count = len(old_emotions.get("frustration_points", []))
        new_frustration_count = len(new_emotions.get("frustration_points", []))
        frustration_reduction = old_frustration_count > new_frustration_count
        
        # Calculate combined emotional reward
        emotional_reward = sentiment_change * 2.0
        if frustration_reduction:
            emotional_reward += 0.5
            
        # Keep history for adaptive scaling
        self.emotional_impact_history.append(emotional_reward)
        if len(self.emotional_impact_history) > 100:
            self.emotional_impact_history.pop(0)
            
        return emotional_reward

    def _calculate_knowledge_reward(self, old_state: Dict, new_state: Dict) -> float:
        knowledge_delta = new_state['knowledge_level'] - old_state['knowledge_level']
        
        # Progressive reward scaling
        if len(self.knowledge_history) > 1:
            avg_gain = np.mean(np.diff(self.knowledge_history[-10:]))
            if knowledge_delta > avg_gain:
                return knowledge_delta * 1.5  # Bonus for above-average gains
        return knowledge_delta

    def _calculate_engagement_reward(self, old_state: Dict, new_state: Dict) -> float:
        engagement_delta = new_state['engagement'] - old_state['engagement']
        current_engagement = new_state['engagement']
        
        # Dynamic engagement rewards
        if current_engagement < 0.2:
            return -2.0  # Severe penalty for very low engagement
        elif current_engagement > 0.8:
            return engagement_delta * 1.5  # Bonus for maintaining high engagement
        
        return engagement_delta

    def _calculate_quality_reward(self, metrics: Dict) -> float:
        response_length = metrics.get('response_length', 0)
        used_interests = metrics.get('used_interests', False)
        
        quality_score = 0.0
        
        # Response length quality
        if 50 <= response_length <= 500:
            quality_score += 0.5
        elif response_length > 1000:
            quality_score -= 0.5
        
        # Context utilization
        if used_interests:
            quality_score += 1.0
        
        return quality_score

    def _calculate_exploration_reward(self, state: Dict) -> float:
        # Reward for exploring new topics and maintaining diverse interests
        topics_count = len(state.get('recent_topics', []))
        interests_count = len(state.get('interests', []))
        
        exploration_score = 0.0
        if topics_count > 0:
            exploration_score += min(1.0, topics_count / 10.0)
        if interests_count > 0:
            exploration_score += min(1.0, interests_count / 10.0)
        
        return exploration_score / 2.0

    def update_reward_scale(self, reward: float):
        # Adaptive reward scaling
        if len(self.engagement_history) > 100:
            self.reward_scale = max(1.0, abs(reward) * 0.95 + self.reward_scale * 0.05)

# Initialize enhanced components
state_encoder = StateEncoder()
reward_calculator = RewardCalculator()
policy_net = PolicyNetwork()
memory = PPOMemory(BATCH_SIZE)

async def get_user_state(user_id: str) -> torch.Tensor:
    try:
        if user_id not in user_states:
            await load_from_db(user_id)
        return state_encoder.encode_state(user_states[user_id])
    except Exception as e:
        logger.error(f"Error getting user state: {str(e)}")
        return torch.tensor([0.5] * STATE_DIM, dtype=torch.float32)

async def calculate_reward(
    old_state: Dict[str, float],
    new_state: Dict[str, float],
    interaction_metrics: Dict[str, float]
) -> float:
    return reward_calculator.calculate_reward(old_state, new_state, interaction_metrics)

async def update_user_state(user_id: str, response_length: int, message_complexity: float):
    """Update user state based on interaction"""
    try:
        if user_id in user_states:
            state = user_states[user_id]
            engagement_delta = min(response_length / 1000, 0.1)
            state['engagement'] = min(1.0, state.get('engagement', 0.5) + engagement_delta)
            knowledge_delta = message_complexity * 0.05
            state['knowledge_level'] = min(1.0, state.get('knowledge_level', 0.5) + knowledge_delta)
            await sync_with_db(user_id)
    except Exception as e:
        logger.error(f"Error updating user state: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update user state")

def select_teaching_strategy(state: torch.Tensor) -> Dict[str, str]:
    """Select teaching strategy using the policy network"""
    try:
        with torch.no_grad():
            action, _ = policy_net.get_action(state)
            strategy_idx = action.item()
        return strategies[strategy_idx]
    except Exception as e:
        logger.error(f"Error selecting teaching strategy: {str(e)}")
        return strategies[0]  # Default to first strategy on error

async def generate_personalized_response(
    message: str,
    user_id: str,
    teaching_strategy: Dict[str, str]
) -> str:
    """Generate personalized response using OpenAI's GPT-4"""
    try:
        user_context = user_states.get(user_id, {})
        
        system_prompt = f"""You are a personalized AI tutor. Use the following teaching strategy:
        - Teaching style: {teaching_strategy['style']}
        - Complexity level: {teaching_strategy['complexity']}
        - Number of examples: {teaching_strategy['examples']}
        
        User's interests: {', '.join(user_context.get('interests', []))}
        Recent topics: {', '.join(user_context.get('recent_topics', []))}
        
        Provide explanations that relate to the user's interests and maintain engagement.
        Use analogies and examples that connect to their personal context when possible."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        chat_history = user_context.get('chat_history', [])[-5:]
        messages[1:1] = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

        # Use synchronous call through the orchestrator
        response = await model_orchestrator.get_response(messages)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

# RL Components
class Experience:
    def __init__(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class ExperienceBuffer:
    def __init__(self, capacity: int = 1000):
        self.buffer = []
        self.capacity = capacity
        
    def add(self, experience: Experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self):
        return len(self.buffer)

# Initialize experience buffer
experience_buffer = ExperienceBuffer()

async def train_policy():
    """Train both policy and value networks using PPO"""
    if len(memory) < BATCH_SIZE:
        return
    
    for _ in range(PPO_EPOCHS):
        states, actions, rewards, next_states, old_log_probs, old_values = memory.get_batch()
        
        # Calculate advantages
        with torch.no_grad():
            next_values = value_net(next_states)
            advantages = rewards.unsqueeze(1) + 0.99 * next_values - old_values.unsqueeze(1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy distributions
        new_log_probs, dist_entropy = policy_net.evaluate(states, actions)
        
        # Calculate policy loss with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss
        new_values = value_net(states)
        value_loss = 0.5 * (rewards.unsqueeze(1) - new_values).pow(2).mean()
        
        # Calculate total loss
        loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * dist_entropy.mean()
        
        # Update networks
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), MAX_GRAD_NORM)
        
        policy_optimizer.step()
        value_optimizer.step()
        
    # Clear memory after updates
    memory.clear()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': dist_entropy.mean().item()
    }

# Periodic training scheduler
last_training_time = datetime.utcnow()
TRAINING_INTERVAL = 300  # Train every 5 minutes if enough experiences

async def should_train() -> bool:
    """Check if it's time to train the policy"""
    global last_training_time
    current_time = datetime.utcnow()
    if (current_time - last_training_time).total_seconds() >= TRAINING_INTERVAL:
        if len(memory) >= 32:  # Minimum batch size
            last_training_time = current_time
            return True
    return False

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add new endpoints
@app.get("/learning-progress/{user_id}")
@limiter.limit(f"{settings.RATE_LIMIT_MINUTE}/minute")
async def get_learning_progress(
    request: Request,
    user_id: str,
    topic: Optional[str] = None
):
    """Get user's learning progress"""
    try:
        user_state = await state_manager.get_state(user_id)
        if topic:
            return {
                "topic": topic,
                "mastery": user_state.get("topic_mastery", {}).get(topic, 0.0),
                "recent_interactions": [
                    msg for msg in user_state.get("chat_history", [])[-5:]
                    if topic.lower() in msg.get("content", "").lower()
                ]
            }
        return {
            "overall_knowledge": user_state.get("knowledge_level", 0.5),
            "topics_mastered": [
                k for k, v in user_state.get("topic_mastery", {}).items()
                if v >= 0.8
            ],
            "topics_in_progress": [
                k for k, v in user_state.get("topic_mastery", {}).items()
                if 0.3 <= v < 0.8
            ]
        }
    except Exception as e:
        logger.error(f"Error getting learning progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get learning progress")

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.RATE_LIMIT_MINUTE}/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """Handle chat requests"""
    try:
        # Verify user exists
        user = await user_states_collection.find_one({"user_id": chat_request.user_id})
        if not user:
            user_state = UserState(user_id=chat_request.user_id).dict()
            await user_states_collection.insert_one(user_state)
            user_states[chat_request.user_id] = user_state

        # Get initial state
        initial_state = await get_user_state(chat_request.user_id)
        initial_state_dict = user_states.get(chat_request.user_id, {}).copy()
        
        # Select teaching strategy
        with torch.no_grad():
            action, action_log_prob = policy_net.get_action(initial_state)
            strategy_idx = action.item()
            teaching_strategy = strategies[strategy_idx]
            value = value_net(initial_state)
        
        # Generate response
        response = await generate_personalized_response(
            chat_request.message,
            chat_request.user_id,
            teaching_strategy
        )
        
        # Create and save messages
        current_time = datetime.utcnow()
        user_message = Message(role="user", content=chat_request.message, timestamp=current_time)
        assistant_message = Message(role="assistant", content=response, timestamp=current_time)
        
        # Update chat history and state
        if chat_request.user_id not in user_states:
            user_states[chat_request.user_id] = UserState(user_id=chat_request.user_id).dict()
        user_states[chat_request.user_id]['chat_history'].append(user_message.dict())
        user_states[chat_request.user_id]['chat_history'].append(assistant_message.dict())
        
        # Save to database
        await save_chat_history(chat_request.user_id, user_message)
        await save_chat_history(chat_request.user_id, assistant_message)
        
        # Update interests
        words = chat_request.message.lower().split()
        used_interests = False
        if len(words) > 2:
            potential_interest = ' '.join(words[:2])
            if potential_interest not in user_states[chat_request.user_id].get('interests', []):
                user_states[chat_request.user_id].setdefault('interests', []).append(potential_interest)
                used_interests = True
        
        # Update user state
        message_complexity = len(response.split()) / 100
        await update_user_state(chat_request.user_id, len(response), message_complexity)
        
        # Calculate reward and get final state
        final_state = await get_user_state(chat_request.user_id)
        final_state_dict = user_states[chat_request.user_id].copy()
        
        interaction_metrics = {
            'response_length': len(response),
            'used_interests': used_interests,
        }
        reward = await calculate_reward(initial_state_dict, final_state_dict, interaction_metrics)
        
        # Store experience in PPO memory
        memory.add(
            initial_state,
            action.item(),
            reward,
            final_state,
            action_log_prob.item(),
            value.item()
        )
        
        # Train policy if it's time
        if await should_train():
            await train_policy()
        
        return ChatResponse(
            response=response,
            teaching_strategy=teaching_strategy
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def generate_enhanced_response(message: str, user_id: str, teaching_strategy: Dict, emotion_data: Dict = None) -> str:
    """Generate personalized response with emotional intelligence and long-term memory"""
    try:
        user_state = await state_manager.get_state(user_id)
        message_analysis = await analyze_message_intent(message)
        
        # If emotion data wasn't provided, detect it now
        if emotion_data is None:
            emotion_data = emotion_detector.detect_emotion(message)
        
        # Retrieve relevant memories
        relevant_memories = memory_manager.retrieve_relevant_memories(user_id, message)
        memories_context = memory_manager.format_memories_for_prompt(relevant_memories)
        
        # Select model based on complexity, context and emotional state
        model_type = 'primary'
        if emotion_data and emotion_data.get("dominant_emotion") == "frustrated":
            model_type = 'specialized'  # Use more capable model for frustrated users
        elif message_analysis['complexity'] == 'high':
            model_type = 'specialized'
        elif message_analysis['intent'] == 'general':
            model_type = 'fast'

        # Enhanced context building
        recent_interactions = user_state.get('chat_history', [])[-5:]
        topic_context = extract_topic_context(recent_interactions)
        mastery_levels = calculate_topic_mastery(user_state)
        
        # Get communication style preferences
        comm_style = user_state.get("communication_style", {})
        vocabulary_level = comm_style.get("vocabulary_level", 0.5)
        formality = comm_style.get("formality", 0.5)
        verbosity = comm_style.get("verbosity", 0.5)
        
        # Get emotional context
        emotional_context = user_state.get("emotional_context", {})
        recent_emotions = emotional_context.get("recent_emotions", [])
        sentiment_trend = emotional_context.get("sentiment_trend", [])
        
        # Personalized system prompt
        system_prompt = f"""You are an advanced AI tutor optimizing for learning outcomes.

Teaching Parameters:
- Style: {teaching_strategy['style']}
- Complexity: {teaching_strategy['complexity']}
- Examples: {teaching_strategy['examples']}

User Context:
- Knowledge Level: {user_state['knowledge_level']:.2f}
- Current Topics: {', '.join(user_state['recent_topics'])}
- Interests: {', '.join(user_state['interests'])}
- Learning Style: {format_learning_style(user_state['learning_style'])}

Communication Preferences:
- Vocabulary Level: {"Advanced" if vocabulary_level > 0.7 else "Intermediate" if vocabulary_level > 0.4 else "Basic"}
- Formality: {"Formal" if formality > 0.7 else "Conversational" if formality > 0.4 else "Casual"}
- Verbosity: {"Detailed" if verbosity > 0.7 else "Balanced" if verbosity > 0.4 else "Concise"}

Emotional Context:
- Current Emotional State: {emotion_data.get("dominant_emotion", "neutral")}
- Recent Pattern: {get_emotional_pattern(recent_emotions)}

Topic Mastery:
{format_topic_mastery(mastery_levels)}

{memories_context}

Instructions:
1. Adapt explanation depth based on topic mastery
2. Use relevant examples from user's interests
3. Match the user's communication style in vocabulary and formality
4. Address emotional state appropriately
5. Maintain consistent complexity level
6. Include specific checkpoints for understanding
7. Encourage active engagement through questions

Previous Context:
{format_recent_context(topic_context)}"""

        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": msg["role"], "content": msg["content"]} for msg in recent_interactions],
            {"role": "user", "content": message}
        ]

        raw_response = await model_orchestrator.get_response(
            messages=messages,
            model_type=model_type,
            temperature=get_dynamic_temperature(teaching_strategy, emotion_data)
        )

        # Apply emotional adjustments to the response if needed
        adjusted_response = emotional_adjuster.adjust_response_style(raw_response, emotion_data, user_state)

        # Validate response quality
        if not validate_response_quality(adjusted_response, teaching_strategy):
            logger.warning("Response quality check failed, regenerating...")
            return await generate_enhanced_response(message, user_id, teaching_strategy, emotion_data)

        # Store this interaction in long-term memory
        memory_context = {
            "emotion": emotion_data.get("dominant_emotion", "neutral"),
            "topic": message_analysis.get("topic", "general"),
            "teaching_strategy": teaching_strategy
        }
        memory_manager.add_memory(user_id, f"Q: {message}\nA: {adjusted_response[:200]}", memory_context)

        return adjusted_response

    except Exception as e:
        logger.error(f"Error in response generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

def extract_topic_context(recent_interactions: List[Dict]) -> Dict[str, Any]:
    """Extract relevant context from recent interactions"""
    context = {}
    
    if not recent_interactions:
        return context
    
    # Collect all user messages
    user_messages = [msg['content'] for msg in recent_interactions if msg.get('role') == 'user']
    
    # Extract key topics using simple keyword frequency
    if user_messages:
        all_text = ' '.join(user_messages)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        word_freq = {}
        
        # Count word frequencies excluding common words
        common_words = {'what', 'when', 'where', 'which', 'while', 'with', 'would', 'could', 'should', 'about'}
        for word in words:
            if word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        context['keywords'] = [word for word, _ in top_keywords]
        
        # Get the most recent question
        context['last_question'] = user_messages[-1]
    
    return context

def format_topic_mastery(mastery_levels: Dict[str, float]) -> str:
    """Format topic mastery levels for prompt"""
    if not mastery_levels:
        return "No topic mastery data available."
    
    result = ""
    for topic, level in mastery_levels.items():
        mastery_desc = "Beginner" if level < 0.3 else "Intermediate" if level < 0.7 else "Advanced"
        result += f"- {topic}: {mastery_desc} ({level:.2f})\n"
    
    return result

def format_recent_context(context: Dict) -> str:
    """Format recent context for prompt"""
    if not context:
        return "No recent context available."
    
    result = ""
    
    if 'keywords' in context:
        result += f"Key topics: {', '.join(context['keywords'])}\n"
    
    if 'last_question' in context:
        result += f"Previous question: {context['last_question']}"
    
    return result

def calculate_text_complexity(text: str) -> float:
    """Calculate text complexity score (0-1)"""
    # Simple implementation based on sentence length and word length
    sentences = re.split(r'[.!?]+', text)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
    
    # Normalize scores (empirically derived thresholds)
    sentence_score = min(1.0, avg_sentence_length / 25.0) # 25 words → score of 1
    word_score = min(1.0, (avg_word_length - 3) / 5.0)    # 8-letter words → score of 1
    
    # Combine scores (weighted average)
    complexity = 0.6 * sentence_score + 0.4 * word_score
    
    return complexity

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests"""
    try:
        user_id = request.user_id
        message = request.message
        
        # Analyze message for emotional content
        emotion_data = emotion_detector.detect_emotion(message)
        
        # Get current user state
        user_state = await state_manager.get_state(user_id)
        old_state = copy.deepcopy(user_state)
        
        # Update emotional context and communication style
        await state_manager.update_emotional_context(user_id, emotion_data)
        await state_manager.update_communication_style(user_id, message)
        
        # Get current state representation for RL
        state = state_encoder.encode_state(user_state)
        
        # Select teaching strategy with current policy
        with torch.no_grad():
            action_probs = policy_net(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            action_log_prob = dist.log_prob(torch.tensor(action)).item()
        
        # Convert action to teaching strategy
        teaching_strategy = strategies[action]
        
        # Generate personalized response with emotion adjustment
        response = await generate_enhanced_response(message, user_id, teaching_strategy, emotion_data)
        
        # Update user state with the new interaction
        user_state["chat_history"].append({"role": "user", "content": message})
        user_state["chat_history"].append({"role": "assistant", "content": response})
        user_state["session_metrics"]["messages_count"] += 1
        await state_manager.update_state(user_id, user_state)
        
        # Get next state representation
        next_state = state_encoder.encode_state(user_state)
        
        # Calculate reward
        interaction_metrics = {"response_quality": 0.8, "response_time": 1.2}  # Example metrics
        reward = reward_calculator.calculate_reward(old_state, user_state, interaction_metrics)
        
        # Add experience to memory
        value = value_net(state).item()
        memory.add(state, action, reward, next_state, action_log_prob, value)
        
        # Train policy if necessary
        if await should_train():
            training_stats = await train_policy_enhanced()
            logger.info(f"Training completed: {training_stats}")
        
        return ChatResponse(
            response=response,
            teaching_strategy=teaching_strategy,
            metrics=calculate_interaction_metrics(user_state)
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
        
def calculate_interaction_metrics(user_state: Dict) -> Dict:
    """Calculate metrics for the current interaction session"""
    return {
        "message_count": user_state.get("session_metrics", {}).get("messages_count", 0),
        "knowledge_level": user_state.get("knowledge_level", 0.5),
        "engagement": user_state.get("engagement", 0.5),
        "topics_explored": len(user_state.get("recent_topics", [])),
        "session_duration_minutes": (datetime.utcnow() - user_state.get("session_start", datetime.utcnow())).total_seconds() / 60
    }

def compute_value_loss(new_values, old_values, rewards):
    """Compute value loss with clipping"""
    value_pred_clipped = old_values + torch.clamp(new_values - old_values, -CLIP_EPSILON, CLIP_EPSILON)
    value_losses = (new_values - rewards.unsqueeze(1)).pow(2)
    value_losses_clipped = (value_pred_clipped - rewards.unsqueeze(1)).pow(2)
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
    return value_loss

def calculate_gae(rewards, values, next_values, gamma, gae_lambda):
    """Calculate Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] - values[step]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, dtype=torch.float32)

class MultiModelOrchestrator:
    def __init__(self, openai_client):
        self.client = openai_client
        self.models = {
            'primary': "gpt-4",
            'fast': "gpt-3.5-turbo",
            'analysis': "gpt-4-turbo-preview",
            'specialized': "gpt-4-1106-preview"
        }
        self._initialized = False

    async def initialize(self):
        """Initialize the model orchestrator"""
        try:
            # Simple initialization test without API call
            self._initialized = True
            logger.info("Model orchestrator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model orchestrator: {str(e)}")
            raise

    async def get_response(self, messages, model_type='primary', temperature=0.7, max_tokens=2000):
        """Get response from the appropriate model"""
        model = self.models.get(model_type, self.models['primary'])
        try:
            # Use synchronous call but wrap in async context
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with {model}: {str(e)}")
            if model_type != 'fast':
                return await self.get_response(messages, 'fast', temperature, max_tokens)
            raise

class EnhancedUserStateManager:
    def __init__(self, db):
        self.db = db
        self.cache = {}
        self.state_encoder = StateEncoder()
        self._initialized = False

    async def initialize(self):
        """Initialize the state manager"""
        try:
            await self.db.user_states.create_index("user_id", unique=True)
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize state manager: {str(e)}")
            raise

    async def get_state(self, user_id: str) -> Dict:
        if not self._initialized:
            await self.initialize()
            
        if user_id not in self.cache:
            state = await self.db.user_states.find_one({"user_id": user_id})
            if not state:
                state = self._create_initial_state(user_id)
            self.cache[user_id] = state
        return self.cache[user_id]

    async def update_state(self, user_id: str, updates: Dict):
        if not self._initialized:
            await self.initialize()
            
        current_state = await self.get_state(user_id)
        current_state.update(updates)
        self.cache[user_id] = current_state
        
        # Sync with database
        await self.db.user_states.update_one(
            {"user_id": user_id},
            {"$set": current_state},
            upsert=True
        )

    async def update_emotional_context(self, user_id: str, emotion_data: Dict):
        state = await self.get_state(user_id)
        if "emotional_context" not in state:
            state["emotional_context"] = {
                "recent_emotions": [],
                "sentiment_trend": [],
                "frustration_points": []
            }
        
        # Update emotional context
        state["emotional_context"]["recent_emotions"].append({
            "emotion": emotion_data.get("dominant_emotion"),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only recent emotions
        state["emotional_context"]["recent_emotions"] = state["emotional_context"]["recent_emotions"][-10:]
        
        # Update sentiment trend
        sentiment = emotion_data.get("sentiment", {}).get("compound", 0)
        state["emotional_context"]["sentiment_trend"].append(sentiment)
        state["emotional_context"]["sentiment_trend"] = state["emotional_context"]["sentiment_trend"][-20:]
        
        await self.update_state(user_id, state)

    def _create_initial_state(self, user_id: str) -> Dict:
        return {
            "user_id": user_id,
            "knowledge_level": 0.5,
            "engagement": 0.5,
            "interests": [],
            "recent_topics": [],
            "learning_style": {
                "visual": 0.5,
                "verbal": 0.5,
                "active": 0.5,
                "reflective": 0.5
            },
            "emotional_context": {
                "recent_emotions": [],
                "sentiment_trend": [],
                "frustration_points": []
            },
            "communication_style": {
                "vocabulary_level": 0.5,
                "formality": 0.5,
                "verbosity": 0.5
            },
            "session_metrics": {
                "messages_count": 0,
                "avg_response_time": 0,
                "topic_mastery": {},
                "engagement_trend": []
            },
            "chat_history": [],
            "last_updated": datetime.utcnow().isoformat()
        }

# Initialize components
state_encoder = StateEncoder()
reward_calculator = RewardCalculator()
policy_net = PolicyNetwork()
value_net = ValueNetwork()
memory = PPOMemory(BATCH_SIZE)
model_orchestrator = MultiModelOrchestrator(openai_client)
state_manager = EnhancedUserStateManager(db)
emotion_detector = EmotionDetector()
emotional_adjuster = EmotionalResponseAdjuster()
memory_manager = MemoryManager()

# Update the chat endpoint to use enhanced response generation
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests"""
    try:
        # ...existing try block code...
        
        # Generate personalized response with emotion adjustment
        response = await generate_enhanced_response(
            message=request.message,
            user_id=request.user_id,
            teaching_strategy=teaching_strategy,
            emotion_data=emotion_data
        )
        
        # ...rest of existing code...

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# Add startup event handler for initialization
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    try:
        # Initialize database connections
        await db.command("ping")
        
        # Create indexes
        await users_collection.create_index("user_id")
        await chat_history_collection.create_index([("user_id", 1), ("timestamp", -1)])
        await user_states_collection.create_index("user_id", unique=True)
        
        # Initialize stateful components
        await state_manager.initialize()
        await model_orchestrator.initialize()
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Add user management routes
app.include_router(user_router, tags=["users"])

# Update PPO configuration
PPO_CONFIG = {
    'batch_size': BATCH_SIZE,
    'epochs': PPO_EPOCHS,
    'clip_epsilon': CLIP_EPSILON,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'value_loss_coef': VALUE_LOSS_COEF,
    'entropy_coef': ENTROPY_COEF,
    'max_grad_norm': MAX_GRAD_NORM,
    'learning_rate': 0.001
}

@app.get("/user/{user_id}")
async def get_user_data(user_id: str):
    """Get user data including state and chat history"""
    try:
        user_state = await state_manager.get_state(user_id)
        if not user_state:
            raise HTTPException(status_code=404, detail="User not found")
        return user_state
    except Exception as e:
        logger.error(f"Error getting user data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)