import logging
from typing import Dict, Any, List
import torch
import numpy as np

logger = logging.getLogger(__name__)

class AssistantsFeedbackCollector:
    """Collects and processes feedback from Assistant interactions to improve RL"""
    
    def __init__(self):
        self.feedback_buffer = []
        self.feedback_stats = {
            "total_interactions": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "neutral_feedback": 0
        }
        
    async def process_interaction(self, 
                                 user_state_before: Dict[str, Any], 
                                 user_state_after: Dict[str, Any],
                                 message: str,
                                 response: str,
                                 teaching_strategy: Dict[str, str]) -> Dict[str, float]:
        """Process an interaction and extract feedback metrics for RL"""
        try:
            # Extract metrics that indicate effectiveness of the teaching strategy
            engagement_delta = user_state_after.get('engagement', 0.5) - user_state_before.get('engagement', 0.5)
            knowledge_delta = user_state_after.get('knowledge_level', 0.5) - user_state_before.get('knowledge_level', 0.5)
            
            # Calculate response effectiveness
            msg_length = len(message)
            resp_length = len(response)
            length_ratio = min(3.0, resp_length / max(1, msg_length))  # Cap at 3x
            
            # Calculate emotional impact
            emotional_context_before = user_state_before.get('emotional_context', {})
            emotional_context_after = user_state_after.get('emotional_context', {})
            
            emotions_before = emotional_context_before.get('recent_emotions', [])
            emotions_after = emotional_context_after.get('recent_emotions', [])
            
            emotional_improvement = 0.0
            if emotions_before and emotions_after:
                # Simple mapping of emotions to numerical values
                emotion_values = {
                    "happy": 1.0,
                    "satisfied": 0.8,
                    "neutral": 0.5,
                    "confused": 0.2,
                    "frustrated": 0.1,
                    "angry": 0.0
                }
                
                # Get the latest emotions
                last_emotion_before = emotions_before[-1] if isinstance(emotions_before[-1], str) else "neutral"
                last_emotion_after = emotions_after[-1] if isinstance(emotions_after[-1], str) else "neutral"
                
                # Calculate improvement
                emotional_improvement = emotion_values.get(last_emotion_after, 0.5) - emotion_values.get(last_emotion_before, 0.5)
            
            # Create feedback metrics
            feedback_metrics = {
                "engagement_delta": float(engagement_delta),
                "knowledge_delta": float(knowledge_delta),
                "response_relevance": self._calculate_relevance(message, response),
                "emotional_improvement": float(emotional_improvement),
                "length_ratio": float(length_ratio),
                "teaching_strategy_match": self._strategy_match_score(teaching_strategy, message, response)
            }
            
            # Store feedback for future analysis
            self.feedback_buffer.append({
                "metrics": feedback_metrics,
                "teaching_strategy": teaching_strategy,
                "timestamp": user_state_after.get('timestamp', None)
            })
            
            # Update stats
            self.feedback_stats["total_interactions"] += 1
            if sum(feedback_metrics.values()) > 0:
                self.feedback_stats["positive_feedback"] += 1
            elif sum(feedback_metrics.values()) < 0:
                self.feedback_stats["negative_feedback"] += 1
            else:
                self.feedback_stats["neutral_feedback"] += 1
                
            return feedback_metrics
                
        except Exception as e:
            logger.error(f"Error processing interaction feedback: {str(e)}")
            return {
                "engagement_delta": 0.0,
                "knowledge_delta": 0.0,
                "response_relevance": 0.5,
                "emotional_improvement": 0.0,
                "length_ratio": 1.0,
                "teaching_strategy_match": 0.5
            }
    
    def _calculate_relevance(self, message: str, response: str) -> float:
        """Calculate how relevant the response is to the message"""
        # Simple heuristic: look for word overlap
        message_words = set(message.lower().split())
        response_words = set(response.lower().split())
        
        if not message_words:
            return 0.5
            
        overlap = len(message_words.intersection(response_words))
        return min(1.0, overlap / len(message_words))
    
    def _strategy_match_score(self, strategy: Dict[str, str], message: str, response: str) -> float:
        """Calculate how well the response matches the teaching strategy"""
        style = strategy.get('style', '').lower()
        complexity = strategy.get('complexity', '').lower()
        examples = strategy.get('examples', '').lower()
        
        score = 0.5  # Default neutral score
        
        # Check style match
        if style == 'detailed' and len(response) > 500:
            score += 0.2
        elif style == 'concise' and len(response) < 300:
            score += 0.2
        elif style == 'interactive' and '?' in response:
            score += 0.2
            
        # Check complexity match
        if complexity == 'high' and self._has_complex_terms(response):
            score += 0.15
        elif complexity == 'low' and not self._has_complex_terms(response):
            score += 0.15
            
        # Check examples match
        example_indicators = ['for example', 'such as', 'like', 'instance', 'consider']
        example_count = sum(response.lower().count(indicator) for indicator in example_indicators)
        
        if examples == 'many' and example_count >= 2:
            score += 0.15
        elif examples == 'few' and 0 < example_count < 2:
            score += 0.15
            
        return min(1.0, max(0.0, score))
    
    def _has_complex_terms(self, text: str) -> bool:
        """Check if text has complex terminology"""
        # This is a simple heuristic - in production, use a more sophisticated approach
        long_words = [word for word in text.split() if len(word) > 8]
        return len(long_words) >= 5
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        return self.feedback_stats
    
    def clear_buffer(self):
        """Clear the feedback buffer"""
        self.feedback_buffer.clear()
