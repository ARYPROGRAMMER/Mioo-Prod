import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
import re

# Use absolute imports instead of relative imports
from rl.ppo_agent import PPOAgent
from llm.llm_client import MultiModelOrchestrator
from db.mongodb_client import MongoDB

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearningSystem:
    """Main orchestration system for the adaptive learning platform"""
    
    def __init__(self, db_client: MongoDB, llm_client: MultiModelOrchestrator, rl_agent = None):
        """Initialize the learning system with all components"""
        self.db = db_client
        self.llm = llm_client
        self.rl_agent = rl_agent or PPOAgent()
        
        # Teaching strategies matching the frontend definitions
        self.teaching_strategies = [
            {"style": "detailed", "complexity": "high", "examples": "many"},
            {"style": "concise", "complexity": "low", "examples": "few"},
            {"style": "interactive", "complexity": "medium", "examples": "some"},
            {"style": "analogy-based", "complexity": "medium", "examples": "some"},
            {"style": "step-by-step", "complexity": "adjustable", "examples": "many"}
        ]
        
        # Topic detection patterns
        self.topics_keywords = {
            "mathematics": ["math", "algebra", "calculus", "geometry", "equation"],
            "programming": ["code", "programming", "function", "algorithm", "variable"],
            "science": ["science", "physics", "chemistry", "biology", "experiment"],
            "history": ["history", "war", "civilization", "century", "ancient"],
            "literature": ["book", "novel", "author", "character", "story"],
            "language": ["grammar", "vocabulary", "language", "word", "sentence"],
            "art": ["art", "painting", "design", "color", "composition"],
            "music": ["music", "note", "instrument", "rhythm", "melody"],
            "philosophy": ["philosophy", "ethics", "logic", "meaning", "existence"],
            "economics": ["economics", "market", "finance", "money", "investment"],
        }
    
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Process a user message and generate a response with adaptive learning"""
        try:
            # Initialize or get user
            user = await self.db.get_user(user_id)
            if not user:
                user = await self.db.create_user(user_id)
                
            # 1. Convert user state to tensor for RL
            state_tensor = self._user_state_to_tensor(user)
            
            # 2. Select action (teaching strategy) using RL agent
            action_idx, log_prob, value = self.rl_agent.select_action(state_tensor)
            teaching_strategy = self.teaching_strategies[action_idx]
            
            # 3. Generate response using LLM with the selected strategy
            llm_response = await self.llm.generate_response(
                message=message,
                user_state=user,
                teaching_strategy=teaching_strategy,
                chat_history=user.get("chat_history", [])[-5:]  # Use last 5 messages as context
            )
            
            # 4. Extract topics and calculate metrics
            detected_topics = self._detect_topics(message, llm_response["response"])
            
            # 5. Update user state with new topics
            for topic in detected_topics:
                if topic not in user.get("recent_topics", []):
                    user.setdefault("recent_topics", []).append(topic)
                    
            # 6. Store the message in chat history
            user_message = {
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.db.add_message_to_history(user_id, user_message)
            
            assistant_message = {
                "role": "assistant",
                "content": llm_response["response"],
                "timestamp": datetime.utcnow().isoformat(),
                "teaching_strategy": teaching_strategy,
            }
            await self.db.add_message_to_history(user_id, assistant_message)
            
            # 7. Calculate metrics based on interaction
            metrics = self._calculate_metrics(user, message, llm_response["response"], teaching_strategy)
            
            # 8. Update user state with new metrics
            updated_state = self._update_user_state(user, metrics, detected_topics)
            await self.db.update_user(user_id, updated_state)
            await self.db.store_learning_metrics(user_id, metrics)
            
            # 9. Store RL experience
            next_state_tensor = self._user_state_to_tensor(updated_state)
            reward = self.rl_agent.compute_reward(
                metrics["knowledge_gain"],
                metrics["engagement_level"] - user["engagement"],
                metrics["interaction_quality"],
                len(detected_topics) * 0.1,  # Exploration score
                0.0  # Emotional improvement (not implemented in this version)
            )
            
            self.rl_agent.store_experience(
                state_tensor, 
                action_idx, 
                reward, 
                next_state_tensor, 
                log_prob, 
                value
            )
            
            # 10. Periodically train the RL agent
            training_metrics = None
            if len(self.rl_agent.memory) >= self.rl_agent.memory.batch_size:
                training_metrics = await self.rl_agent.train()
                logger.info(f"RL agent trained: {training_metrics}")
            
            # 11. Return the response, metrics and strategy
            return {
                "response": llm_response["response"],
                "teaching_strategy": teaching_strategy,
                "metrics": metrics,
                "detected_topics": detected_topics,
                "model_used": llm_response["model"],
                "rl_training": training_metrics
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Return a fallback response
            return {
                "response": "I'm experiencing some technical difficulties. Could you please try again?",
                "teaching_strategy": self.teaching_strategies[0],
                "metrics": {
                    "knowledge_gain": 0.0,
                    "engagement_level": 0.0,
                    "performance_score": 0.0,
                    "strategy_effectiveness": 0.0,
                    "interaction_quality": 0.0,
                },
                "error": str(e)
            }
    
    def _user_state_to_tensor(self, user: Dict[str, Any]) -> torch.Tensor:
        """Enhanced user state representation"""
        # Core metrics
        knowledge_level = user.get("knowledge_level", 0.5)
        engagement = user.get("engagement", 0.5)
        performance = user.get("performance", 0.5)
        
        # Learning style and preferences
        learning_style_vec = self._encode_learning_style(user.get("learning_style", {}))
        topic_mastery = np.mean([v for v in user.get("topic_mastery", {}).values()] or [0.5])
        feedback_ratio = self._calculate_feedback_ratio(user.get("feedback_history", []))
        
        # Session metrics
        session_metrics = user.get("session_metrics", {})
        avg_response_time = session_metrics.get("avg_response_time", 30) / 60  # Normalize to [0,1]
        interaction_depth = min(1.0, session_metrics.get("max_conversation_turns", 0) / 20)
        
        features = [
            knowledge_level,
            engagement,
            performance,
            topic_mastery,
            feedback_ratio,
            *learning_style_vec,  # Unpack learning style vector
            avg_response_time,
            interaction_depth
        ]
        
        return torch.FloatTensor(features)

    def _calculate_feedback_ratio(self, feedback_history: List[str]) -> float:
        """Calculate ratio of positive feedback"""
        if not feedback_history:
            return 0.5
        positive = sum(1 for f in feedback_history if f == "like")
        return positive / len(feedback_history)

    def _encode_learning_style(self, learning_style: Dict[str, float]) -> List[float]:
        """Encode learning style preferences"""
        default_style = {
            "visual": 0.5,
            "interactive": 0.5,
            "theoretical": 0.5,
            "practical": 0.5
        }
        style = {**default_style, **learning_style}
        return list(style.values())

    async def process_feedback(self, user_id: str, message_id: str, feedback: str) -> Dict[str, Any]:
        """Process user feedback and update the learning system"""
        user = await self.db.get_user(user_id)
        if not user:
            raise ValueError("User not found")

        # Update feedback history
        user.setdefault("feedback_history", []).append(feedback)
        
        # Get the interaction this feedback is for
        interaction = await self.db.get_interaction(message_id)
        if interaction:
            # Update RL agent with feedback
            state_tensor = self._user_state_to_tensor(interaction["user_state"])
            next_state_tensor = self._user_state_to_tensor(user)
            
            # Recompute reward with feedback
            reward = self.rl_agent.compute_reward(
                interaction["metrics"]["knowledge_gain"],
                interaction["metrics"]["engagement_delta"],
                interaction["metrics"]["response_quality"],
                interaction["metrics"]["exploration_score"],
                interaction["metrics"]["emotional_improvement"],
                feedback
            )
            
            # Store updated experience
            self.rl_agent.store_experience(
                state_tensor,
                interaction["action_idx"],
                reward,
                next_state_tensor,
                interaction["action_log_prob"],
                interaction["value"]
            )
            
            # Trigger training if enough experiences
            await self.rl_agent.train()

        # Update user state
        await self.db.update_user(user_id, user)
        
        return {"status": "success", "feedback_processed": True}

    def _detect_topics(self, user_message: str, assistant_response: str) -> List[str]:
        """Detect topics in the conversation"""
        combined_text = (user_message + " " + assistant_response).lower()
        
        detected_topics = []
        for topic, keywords in self.topics_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', combined_text):
                    detected_topics.append(topic)
                    break
        
        return list(set(detected_topics))  # Remove duplicates
    
    def _calculate_metrics(self, user: Dict[str, Any], 
                         user_message: str, 
                         assistant_response: str, 
                         teaching_strategy: Dict[str, str]) -> Dict[str, float]:
        """Calculate learning metrics based on interaction"""
        # For a real system, these would be more sophisticated
        
        # Basic metrics calculation
        knowledge_gain = min(1.0, 0.4 + len(assistant_response) / 2000)  # Longer answers might indicate more knowledge
        
        # Message complexity as a proxy for knowledge gain
        word_count = len(assistant_response.split())
        sentence_count = max(1, len(re.findall(r'[.!?]+', assistant_response)))
        avg_sentence_length = word_count / sentence_count
        
        # Adjust knowledge gain based on teaching strategy and complexity
        if teaching_strategy["complexity"] == "high":
            knowledge_gain *= 1.2
        elif teaching_strategy["complexity"] == "low":
            knowledge_gain *= 0.8
            
        # Engagement based on message length and question marks
        question_count = len(re.findall(r'\?', user_message))
        engagement_level = min(1.0, 0.3 + question_count * 0.1 + len(user_message) / 500)
        
        # Strategy effectiveness depends on alignment with user's state
        strategy_matched = False
        if user.get("knowledge_level", 0.5) < 0.3 and teaching_strategy["complexity"] == "low":
            strategy_matched = True
        elif user.get("knowledge_level", 0.5) > 0.7 and teaching_strategy["complexity"] == "high":
            strategy_matched = True
        elif teaching_strategy["complexity"] == "medium":
            strategy_matched = True
            
        strategy_effectiveness = 0.5 + (0.3 if strategy_matched else 0.0)
        
        # Interaction quality - would be more sophisticated in production
        interaction_quality = min(1.0, (knowledge_gain + engagement_level) / 2)
        
        # Performance score
        performance_score = min(1.0, (knowledge_gain + strategy_effectiveness) / 2)
        
        # Topic mastery - simplified version
        topics = self._detect_topics(user_message, assistant_response)
        topic_mastery = {}
        
        # Update topic mastery for each detected topic
        for topic in topics:
            current_mastery = 0.3  # Default starting mastery
            
            # Check if we already have mastery data for this topic
            if user.get("topic_mastery") and topic in user["topic_mastery"]:
                current_mastery = user["topic_mastery"][topic]
                
            # Increment mastery based on interaction
            new_mastery = min(1.0, current_mastery + knowledge_gain * 0.2)
            topic_mastery[topic] = new_mastery
        
        return {
            "knowledge_gain": knowledge_gain,
            "engagement_level": engagement_level,
            "performance_score": performance_score,
            "strategy_effectiveness": strategy_effectiveness,
            "interaction_quality": interaction_quality,
            "topic_mastery": topic_mastery
        }
    
    def _update_user_state(self, user: Dict[str, Any], 
                         metrics: Dict[str, Any], 
                         new_topics: List[str]) -> Dict[str, Any]:
        """Update user state with new metrics and topics"""
        # Create a copy to avoid modifying the original directly
        updated_state = user.copy()
        
        # Update knowledge level - exponential moving average
        alpha = 0.3  # Weight for new observation
        updated_state["knowledge_level"] = (1 - alpha) * user.get("knowledge_level", 0.5) + \
                                           alpha * metrics["knowledge_gain"]
                                           
        # Update engagement level
        updated_state["engagement"] = (1 - alpha) * user.get("engagement", 0.5) + \
                                     alpha * metrics["engagement_level"]
                                     
        # Update performance
        updated_state["performance"] = (1 - alpha) * user.get("performance", 0.5) + \
                                      alpha * metrics["performance_score"]
        
        # Update topic mastery
        if "topic_mastery" not in updated_state:
            updated_state["topic_mastery"] = {}
            
        updated_state["topic_mastery"].update(metrics.get("topic_mastery", {}))
        
        # Update session metrics
        if "session_metrics" not in updated_state:
            updated_state["session_metrics"] = {}
            
        # Update topics covered
        updated_state["session_metrics"].setdefault("topics_covered", [])
        for topic in new_topics:
            if topic not in updated_state["session_metrics"]["topics_covered"]:
                updated_state["session_metrics"]["topics_covered"].append(topic)
        
        # Update learning rate
        history_len = len(updated_state.get("learning_history", []))
        if history_len >= 2:
            recent_gains = [entry.get("knowledge", 0) for entry in updated_state["learning_history"][-3:]]
            updated_state["session_metrics"]["learning_rate"] = sum(recent_gains) / len(recent_gains)
        else:
            updated_state["session_metrics"]["learning_rate"] = metrics["knowledge_gain"] / 2
            
        # Update engagement trend
        updated_state["session_metrics"].setdefault("engagement_trend", [])
        updated_state["session_metrics"]["engagement_trend"].append(metrics["engagement_level"])
        
        # Keep only the last 10 engagement points for trend analysis
        if len(updated_state["session_metrics"]["engagement_trend"]) > 10:
            updated_state["session_metrics"]["engagement_trend"] = \
                updated_state["session_metrics"]["engagement_trend"][-10:]
        
        return updated_state
