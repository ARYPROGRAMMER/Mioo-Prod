import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, Tuple, List

class EmotionDetector:
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        
    def detect_emotion(self, message: str) -> Dict[str, float]:
        # Basic sentiment analysis
        sentiment = self.sia.polarity_scores(message)
        
        # Detect specific emotions
        emotions = {
            "confused": self._contains_confusion(message),
            "frustrated": self._contains_frustration(message),
            "curious": self._contains_curiosity(message),
            "satisfied": self._contains_satisfaction(message)
        }
        
        return {
            "sentiment": sentiment,
            "emotions": emotions,
            "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[0] if any(emotions.values()) else "neutral"
        }
    
    def _contains_confusion(self, message: str) -> float:
        confusion_patterns = [r'\bconfused\b', r'\bdon\'t\s+understand\b', r'\bunclear\b', r'\bhuh\b', r'\bwhat\?\b']
        return self._pattern_match_score(message, confusion_patterns)
    
    def _contains_frustration(self, message: str) -> float:
        frustration_patterns = [r'\bfrustrat(ed|ing)\b', r'\bdifficult\b', r'\bhard\b', r'\bcan\'t\b', r'\bstuck\b']
        return self._pattern_match_score(message, frustration_patterns)
    
    def _contains_curiosity(self, message: str) -> float:
        curiosity_patterns = [r'\bcurious\b', r'\binterested\b', r'\btell me more\b', r'\bhow does\b', r'\bwhy\b']
        return self._pattern_match_score(message, curiosity_patterns)
    
    def _contains_satisfaction(self, message: str) -> float:
        satisfaction_patterns = [r'\bgot it\b', r'\bunderstand\b', r'\bthanks\b', r'\bclear\b', r'\bmakes sense\b']
        return self._pattern_match_score(message, satisfaction_patterns)
    
    def _pattern_match_score(self, message: str, patterns: List[str]) -> float:
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, message.lower()):
                score += 0.25
        return min(1.0, score)

class EmotionalResponseAdjuster:
    def adjust_response_style(self, response: str, emotion_data: Dict, user_state: Dict) -> str:
        dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
        sentiment = emotion_data.get("sentiment", {}).get("compound", 0)
        
        if dominant_emotion == "confused":
            return self._simplify_explanation(response)
        elif dominant_emotion == "frustrated":
            return self._add_encouragement(response)
        elif dominant_emotion == "curious":
            return self._expand_information(response, user_state)
        elif sentiment < -0.3:
            return self._add_empathy(response)
        
        return response
    
    def _simplify_explanation(self, response: str) -> str:
        # Add simpler explanation
        simplified = response.split('\n\n')[0] if '\n\n' in response else response
        return f"{simplified}\n\nTo put it more simply: {self._get_simplified_version(response)}"
    
    def _add_encouragement(self, response: str) -> str:
        encouragements = [
            "I know this can be challenging, but you're making progress.",
            "Don't worry if this feels difficult at first - it's a common sticking point.",
            "Let's break this down step by step to make it easier to follow."
        ]
        import random
        return f"{random.choice(encouragements)}\n\n{response}"
    
    def _expand_information(self, response: str, user_state: Dict) -> str:
        interests = user_state.get("interests", [])
        if interests:
            return f"{response}\n\nSince you're interested in {interests[0]}, you might also want to know: {self._get_additional_info(response, interests[0])}"
        return response
    
    def _add_empathy(self, response: str) -> str:
        empathy_phrases = [
            "I understand this might be frustrating.",
            "I see you're having trouble with this concept.",
            "Let me try a different approach that might work better for you."
        ]
        import random
        return f"{random.choice(empathy_phrases)}\n\n{response}"
    
    def _get_simplified_version(self, text: str) -> str:
        # Placeholder for more sophisticated simplification logic
        sentences = text.split('. ')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        return text
    
    def _get_additional_info(self, response: str, interest: str) -> str:
        # Placeholder for generating additional interest-related content
        return f"This concept also relates to {interest} in interesting ways that we can explore further if you'd like."
