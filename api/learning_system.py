import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
import re
import uuid

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
            # Get user and full chat history
            user = await self.db.get_user(user_id)
            if not user:
                user = await self.db.create_user(user_id)
                
            # Ensure chat history is loaded
            chat_history = await self.db.get_chat_history(user_id)
            user["chat_history"] = chat_history
            
            # Track conversation thread for context
            conversation_thread = {
                "current_topic": None,
                "last_assistant_message": None,
                "context_switches": 0
            }
            
            # Get last assistant message for context
            if chat_history:
                for msg in reversed(chat_history):
                    if msg["role"] == "assistant":
                        conversation_thread["last_assistant_message"] = msg
                        break
                        
            # Detect topic changes
            current_focus = self._detect_topic(message)
            if conversation_thread["current_topic"] and current_focus != conversation_thread["current_topic"]:
                conversation_thread["context_switches"] += 1
            conversation_thread["current_topic"] = current_focus
            
            # Add conversation thread to user state for context
            user["conversation_thread"] = conversation_thread
            
            # Check for user introduction and update state
            intro_info = self._check_introduction(message)
            if intro_info:
                user.update(intro_info)
                await self.db.update_user(user_id, user)
                
            # Log user state for debugging personalization    
            logger.info(f"Processing message for user {user_id} with state: knowledge={user.get('knowledge_level', 0.5):.2f}, engagement={user.get('engagement', 0.5):.2f}")
            
            # Message analysis for context understanding
            message_context = self._analyze_message_context(message, user.get("chat_history", []))
            logger.info(f"Message context: {message_context}")
            
            # 1. Convert user state to tensor for RL
            state_tensor = self._user_state_to_tensor(user)
            
            # 2. Select action (teaching strategy) using RL agent - this is the personalization point
            action_idx, log_prob, value = self.rl_agent.select_action(state_tensor)
            teaching_strategy = self.teaching_strategies[action_idx]
            
            logger.info(f"Selected teaching strategy for user {user_id}: {teaching_strategy['style']} (complexity: {teaching_strategy['complexity']})")
            
            # Enhanced context building
            user_context = {
                "name": user.get("name", "there"),
                "interests": user.get("interests", []),
                "knowledge_level": user.get("knowledge_level", 0.5),
                "recent_topics": user.get("recent_topics", []),
                "is_math_question": any(op in message for op in ['+', '-', '*', '/', '=']),
                "is_identity_query": "who am i" in message.lower()
            }

            # Special handling for identity queries
            if user_context["is_identity_query"]:
                interests = user_context["interests"]
                response = f"You're {user_context['name']}"
                if interests:
                    response += f", and you're interested in {', '.join(interests)}"
                response += f". Your current knowledge level is {self._describe_level(user_context['knowledge_level'])}."
                if user_context["recent_topics"]:
                    response += f" We've recently discussed {', '.join(user_context['recent_topics'])}."
                
                return {
                    "response": response,
                    "teaching_strategy": teaching_strategy,
                    "metrics": metrics
                }

            # 3. Generate response using LLM with the selected strategy and FULL chat history
            llm_response = await self.llm.generate_response(
                message=message,
                user_state=user_context,  # Pass complete user state for personalization
                teaching_strategy=teaching_strategy,
                chat_history=user.get("chat_history", [])  # Pass all available chat history for context
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
                "message_id": str(uuid.uuid4())  # Add unique ID to message
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
                metrics["engagement_level"] - user.get("engagement", 0.5),
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
            
            # Store interaction for future feedback - Ensure we clean the state before storing
            try:
                await self.db.store_interaction(
                    user_id=user_id,
                    message_id=assistant_message["message_id"],
                    action_idx=action_idx,
                    action_log_prob=log_prob,
                    value=value,
                    metrics={
                        "knowledge_gain": metrics["knowledge_gain"],
                        "engagement_delta": metrics["engagement_level"] - user.get("engagement", 0.5),
                        "response_quality": metrics["interaction_quality"],
                        "exploration_score": len(detected_topics) * 0.1,
                        "emotional_improvement": 0.0
                    },
                    user_state=user
                )
            except Exception as e:
                # Log the error but don't fail the entire message processing
                logger.error(f"Failed to store interaction: {e}")
            
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
        """Enhanced user state representation - ensure exactly 11 dimensions"""
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
        topics_count = min(1.0, len(user.get("recent_topics", [])) / 10)
        
        # Ensure exactly 11 dimensions for the state vector
        features = [
            knowledge_level,       # 1
            engagement,            # 2
            performance,           # 3
            topic_mastery,         # 4
            feedback_ratio,        # 5
            learning_style_vec[0], # 6
            learning_style_vec[1], # 7
            learning_style_vec[2], # 8
            learning_style_vec[3], # 9
            avg_response_time,     # 10
            topics_count           # 11
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
        # Return exactly 4 values to maintain consistent tensor dimensions
        default_style = {
            "visual": 0.5,
            "interactive": 0.5,
            "theoretical": 0.5,
            "practical": 0.5
        }
        style = {**default_style, **learning_style}
        return [style["visual"], style["interactive"], 
                style["theoretical"], style["practical"]]

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
        updated_state = user.copy()
        
        # Persist core user identity
        identity = {
            "name": user.get("name"),
            "interests": user.get("interests", []),
            "preferred_topics": user.get("preferred_topics", []),
            "last_interactions": user.get("last_interactions", [])[-5:]
        }
        updated_state.update(identity)
        
        # Track interaction context
        current_interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "topics": new_topics,
            "context": self._build_active_context(user)
        }
        updated_state.setdefault("last_interactions", []).append(current_interaction)
        updated_state["last_interactions"] = updated_state["last_interactions"][-5:]  # Keep last 5
        
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

    def _analyze_message_context(self, message: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the message for contextual cues to better personalize responses"""
        context = {
            "length": len(message),
            "is_question": "?" in message,
            "is_short_response": len(message.split()) <= 3,
            "sentiment": "neutral",  # Default sentiment
            "requires_context_switch": False
        }
        
        # Check for very short negative responses that might indicate dissatisfaction
        if context["is_short_response"]:
            negative_responses = ["no", "wrong", "incorrect", "not right", "nope"]
            if any(neg in message.lower() for neg in negative_responses):
                context["sentiment"] = "negative"
                context["requires_context_switch"] = True
        
        # Check conversation continuity
        if chat_history and len(chat_history) >= 2:
            last_bot_message = next((msg for msg in reversed(chat_history) 
                                   if msg["role"] == "assistant"), None)
            if last_bot_message:
                # If user gives very short response after long bot message, might indicate disengagement
                if len(last_bot_message.get("content", "")) > 200 and context["is_short_response"]:
                    context["possible_disengagement"] = True
        
        return context

    def _check_introduction(self, message: str) -> Optional[Dict[str, Any]]:
        """Check for user introduction and extract information"""
        intro_info = {}
        
        # Name patterns
        name_match = re.search(r"(?i)(?:i am|my name is|i'm) (\w+)", message)
        if name_match:
            intro_info["name"] = name_match.group(1).capitalize()
            
        # Interest patterns
        interests = []
        interest_matches = re.finditer(r"(?i)i (?:like|love|enjoy) (\w+(?:\+\+)?)", message)
        interests.extend(match.group(1).lower() for match in interest_matches)
        
        if interests:
            intro_info["interests"] = list(set(interests))  # Deduplicate
            
        return intro_info if intro_info else None

    def _is_introduction(self, message: str) -> bool:
        """Check if message contains user introduction"""
        patterns = [
            r"(?i)i am \w+",
            r"(?i)my name is \w+",
            r"(?i)i(?:'m)? like \w+",
            r"(?i)i(?:'m)? interested in \w+"
        ]
        return any(re.search(pattern, message) for pattern in patterns)

    def _build_active_context(self, user: Dict[str, Any], current_message: str = "") -> Dict[str, Any]:
        """Build rich context for current interaction"""
        return {
            "user_identity": {
                "name": user.get("name"),
                "interests": user.get("interests", []),
                "known_topics": user.get("recent_topics", []),
                "preferred_style": self._determine_preferred_style(user)
            },
            "learning_state": {
                "knowledge_level": user.get("knowledge_level", 0.5),
                "engagement": user.get("engagement", 0.5),
                "mastered_topics": [t for t, v in user.get("topic_mastery", {}).items() if v > 0.8],
                "struggling_topics": [t for t, v in user.get("topic_mastery", {}).items() if v < 0.3]
            },
            "interaction_history": {
                "last_topics": user.get("recent_topics", [])[-3:],
                "preferred_examples": [i.lower() for i in user.get("interests", [])]
            }
        }

    def _determine_preferred_style(self, user: Dict[str, Any]) -> str:
        """Determine user's preferred teaching style based on engagement history"""
        history = user.get("learning_history", [])
        if not history:
            return "balanced"

        # Analyze which strategies led to highest engagement
        strategy_scores = {}
        for entry in history[-10:]:  # Look at last 10 interactions
            strategy = entry.get("strategy", {}).get("style", "balanced")
            engagement = entry.get("engagement", 0.5)
            strategy_scores[strategy] = strategy_scores.get(strategy, []) + [engagement]

        # Get average engagement per strategy
        avg_scores = {
            k: sum(v)/len(v) for k, v in strategy_scores.items()
        }

        return max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else "balanced"

    def _detect_topic(self, message: str) -> Optional[str]:
        """Enhanced topic detection"""
        message = message.lower()
        
        # Check for context references
        if any(ref in message for ref in ["you said", "earlier", "before", "previous"]):
            return "context_reference"
            
        # Check for identity queries
        if "who am i" in message or "what did i" in message:
            return "user_identity"
            
        # Check for previous answer references
        if any(ref in message for ref in ["the answer", "that answer", "your response"]):
            return "previous_response"
            
        # Add specific topic detection
        if "who am i" in message:
            return "user_identity"
        if any(op in message for op in ['+', '-', '*', '/', '=']):
            return "mathematics"
        for topic, keywords in self.topics_keywords.items():
            if any(keyword in message for keyword in keywords):
                return topic
        return None

    def _describe_level(self, level: float) -> str:
        if level < 0.3: return "at a beginner level"
        if level < 0.6: return "at an intermediate level"
        if level < 0.8: return "at an advanced level"
        return "at an expert level"
