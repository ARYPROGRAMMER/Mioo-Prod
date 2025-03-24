from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter

from ..personas.student_personas import StudentPersona
from ..trainers.persona_trainer import ContentAction

class PersonaAdaptationEvaluator:
    """Evaluates the quality of persona-based adaptations."""
    
    def __init__(self):
        self.style_match_weights = {
            "examples": ["Detailed step-by-step", "Real-world application"],
            "visual": ["Visual aid focused"],
            "step-by-step": ["Detailed step-by-step"],
            "simple language": ["Simplified language"],
            "bullet points": ["Bullet point summary"],
            "analogies": ["Analogy-based"],
            "interactive": ["Interactive questioning"],
            "stories": ["Story context"]
        }
        
        self.theme_match_weights = {
            "sports": "Sports",
            "football": "Sports",
            "video games": "Video Games",
            "fortnite": "Video Games",
            "minecraft": "Video Games",
            "smartphones": "Technology",
            "mobile": "Technology",
            "apps": "Technology",
            "food": "Food",
            "nutrition": "Food",
            "mcdonald": "Food",
            "starbucks": "Food",
            "streaming": "Entertainment",
            "netflix": "Entertainment",
            "spotify": "Entertainment",
            "music": "Music",
            "sneaker": "Fashion",
            "shoe": "Fashion",
            "nike": "Fashion",
            "adidas": "Fashion",
            "beats": "Music",
            "audio": "Music",
            "theme park": "Theme Parks",
            "roller coaster": "Theme Parks",
            "disney": "Theme Parks",
            "six flags": "Theme Parks",
            "cars": "Transportation",
            "driving": "Transportation",
            "tesla": "Transportation",
            "vehicles": "Transportation",
            "photo": "Social Media",
            "instagram": "Social Media",
            "filter": "Social Media"
        }
    
    def evaluate_persona_match(self, persona: StudentPersona, action: ContentAction) -> Dict[str, Any]:
        """Evaluate how well a selected action matches a student persona."""
        # Calculate style match score
        style_score = self._calculate_style_match(persona.learning_style, action.style)
        
        # Calculate theme match score
        theme_score = self._calculate_theme_match(persona.likes_theme, action.theme)
        
        # Calculate difficulty appropriateness
        difficulty_score = self._calculate_difficulty_match(persona.math_mastery, action.difficulty)
        
        # Calculate overall score (weighted average)
        overall_score = 0.4 * style_score + 0.3 * theme_score + 0.3 * difficulty_score
        
        return {
            "overall_match_score": overall_score,
            "style_match": style_score,
            "theme_match": theme_score,
            "difficulty_match": difficulty_score,
            "style_details": self._get_style_details(persona.learning_style, action.style),
            "theme_details": self._get_theme_details(persona.likes_theme, action.theme),
            "difficulty_details": self._get_difficulty_details(persona.math_mastery, action.difficulty)
        }
    
    def _calculate_style_match(self, learning_style: str, selected_style: str) -> float:
        """Calculate how well the selected style matches the student's learning style preference."""
        learning_style = learning_style.lower()
        selected_style = selected_style.lower()
        
        match_score = 0.0
        matched_preferences = []
        
        for preference, matching_styles in self.style_match_weights.items():
            if preference in learning_style:
                matched_preferences.append(preference)
                for style in matching_styles:
                    if style.lower() in selected_style:
                        match_score += 1.0
                        break
        
        # Normalize score
        return match_score / max(1, len(matched_preferences)) if matched_preferences else 0.5
    
    def _calculate_theme_match(self, student_theme: str, selected_theme: str) -> float:
        """Calculate how well the selected theme matches the student's preferred theme."""
        student_theme = student_theme.lower()
        
        # Find matching theme keywords
        matched_theme = None
        for keyword, theme in self.theme_match_weights.items():
            if keyword in student_theme:
                if theme == selected_theme:
                    matched_theme = theme
                    break
        
        # Direct match
        if matched_theme:
            return 1.0
        
        # Partial or related match
        for keyword, theme in self.theme_match_weights.items():
            if keyword in student_theme:
                # Give partial credit for thematically related content
                if (theme == "Sports" and selected_theme in ["Video Games", "Entertainment"]) or \
                   (theme == "Video Games" and selected_theme in ["Technology", "Entertainment"]) or \
                   (theme == "Food" and selected_theme in ["Social Media"]) or \
                   (theme == "Music" and selected_theme in ["Entertainment", "Technology"]) or \
                   (theme == "Social Media" and selected_theme in ["Technology", "Entertainment"]):
                    return 0.5
        
        return 0.0  # No match
    
    def _calculate_difficulty_match(self, math_mastery: Dict[str, str], difficulty: str) -> float:
        """Calculate how appropriate the difficulty level is based on the student's math mastery."""
        # Calculate average grade level
        grade_sum = 0
        count = 0
        
        for subject, grade in math_mastery.items():
            if grade.startswith("Grade "):
                try:
                    grade_sum += int(grade.split(" ")[1])
                    count += 1
                except ValueError:
                    pass
        
        avg_grade = grade_sum / count if count > 0 else 9  # Default to grade 9
        
        # Determine appropriate difficulty
        appropriate_difficulty = ""
        if avg_grade <= 7.5:
            appropriate_difficulty = "Basic"
        elif avg_grade <= 8.5:
            appropriate_difficulty = "Basic"
        elif avg_grade <= 9.5:
            appropriate_difficulty = "Standard"
        elif avg_grade <= 10.5:
            appropriate_difficulty = "Advanced"
        else:
            appropriate_difficulty = "Challenge"
        
        # Score based on how close the selected difficulty is to the appropriate one
        if difficulty == appropriate_difficulty:
            return 1.0
        elif (difficulty == "Standard" and appropriate_difficulty in ["Basic", "Advanced"]) or \
             (difficulty == "Advanced" and appropriate_difficulty == "Challenge"):
            return 0.7  # Adjacent difficulty levels
        elif (difficulty == "Basic" and appropriate_difficulty == "Challenge") or \
             (difficulty == "Challenge" and appropriate_difficulty == "Basic"):
            return 0.0  # Completely mismatched difficulty
        else:
            return 0.3  # Other partial matches
    
    def _get_style_details(self, learning_style: str, selected_style: str) -> str:
        """Provide detailed explanation of style match."""
        learning_style = learning_style.lower()
        selected_style = selected_style.lower()
        
        matched_keywords = []
        for preference in self.style_match_weights.keys():
            if preference in learning_style and preference in selected_style:
                matched_keywords.append(preference)
        
        if matched_keywords:
            return f"Style match: student prefers {', '.join(matched_keywords)}, which is addressed by the '{selected_style}' style."
        else:
            return f"Style mismatch: student learning style '{learning_style}' doesn't align well with '{selected_style}'."
    
    def _get_theme_details(self, student_theme: str, selected_theme: str) -> str:
        """Provide detailed explanation of theme match."""
        student_theme = student_theme.lower()
        
        for keyword, theme in self.theme_match_weights.items():
            if keyword in student_theme and theme == selected_theme:
                return f"Theme match: '{keyword}' in student interests aligns with the '{selected_theme}' theme."
        
        return f"Theme mismatch: student theme preferences '{student_theme}' don't clearly align with '{selected_theme}'."
    
    def _get_difficulty_details(self, math_mastery: Dict[str, str], difficulty: str) -> str:
        """Provide detailed explanation of difficulty appropriateness."""
        # Calculate average mastery
        grades = []
        for subject, grade in math_mastery.items():
            if grade.startswith("Grade "):
                try:
                    grades.append(int(grade.split(" ")[1]))
                except ValueError:
                    pass
        
        if not grades:
            return "Unable to assess difficulty match: no grade information available."
        
        avg_grade = sum(grades) / len(grades)
        
        if avg_grade <= 7.5 and difficulty == "Basic":
            return f"Appropriate difficulty: grade level {avg_grade:.1f} matches 'Basic' difficulty."
        elif 7.5 < avg_grade <= 9.0 and difficulty == "Standard":
            return f"Appropriate difficulty: grade level {avg_grade:.1f} matches 'Standard' difficulty."
        elif 9.0 < avg_grade <= 10.5 and difficulty == "Advanced":
            return f"Appropriate difficulty: grade level {avg_grade:.1f} matches 'Advanced' difficulty."
        elif avg_grade > 10.5 and difficulty == "Challenge":
            return f"Appropriate difficulty: grade level {avg_grade:.1f} matches 'Challenge' difficulty."
        else:
            return f"Suboptimal difficulty: grade level {avg_grade:.1f} would better match a different difficulty than '{difficulty}'."
    
    def evaluate_batch(self, personas: List[StudentPersona], actions: List[ContentAction]) -> Dict[str, Any]:
        """Evaluate a batch of personas and their corresponding actions."""
        if len(personas) != len(actions):
            raise ValueError("Number of personas and actions must match")
        
        individual_scores = []
        style_scores = []
        theme_scores = []
        difficulty_scores = []
        
        for persona, action in zip(personas, actions):
            evaluation = self.evaluate_persona_match(persona, action)
            individual_scores.append(evaluation["overall_match_score"])
            style_scores.append(evaluation["style_match"])
            theme_scores.append(evaluation["theme_match"])
            difficulty_scores.append(evaluation["difficulty_match"])
        
        return {
            "average_match_score": np.mean(individual_scores),
            "average_style_match": np.mean(style_scores),
            "average_theme_match": np.mean(theme_scores),
            "average_difficulty_match": np.mean(difficulty_scores),
            "match_distribution": {
                "excellent": sum(1 for s in individual_scores if s >= 0.8),
                "good": sum(1 for s in individual_scores if 0.6 <= s < 0.8),
                "moderate": sum(1 for s in individual_scores if 0.4 <= s < 0.6),
                "poor": sum(1 for s in individual_scores if s < 0.4)
            }
        }
