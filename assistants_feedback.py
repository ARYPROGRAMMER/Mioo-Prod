from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AssistantsFeedbackCollector:
    def __init__(self):
        self.feedback_history: List[Dict[str, Any]] = []
    
    def add_feedback(self, interaction_data: Dict[str, Any]) -> None:
        """Add feedback for an assistant interaction"""
        self.feedback_history.append(interaction_data)
    
    def get_assistant_performance(self, assistant_id: str) -> Dict[str, Any]:
        """Get performance metrics for an assistant"""
        relevant_feedback = [
            f for f in self.feedback_history 
            if f.get("assistant_id") == assistant_id
        ]
        
        if not relevant_feedback:
            return {"error": "No feedback found for this assistant"}
            
        total_interactions = len(relevant_feedback)
        avg_satisfaction = sum(f.get("satisfaction", 0) for f in relevant_feedback) / total_interactions
        
        return {
            "total_interactions": total_interactions,
            "average_satisfaction": avg_satisfaction,
            "feedback_samples": relevant_feedback[-5:]  # Last 5 feedback entries
        }
