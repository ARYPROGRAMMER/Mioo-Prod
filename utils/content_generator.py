from typing import Dict, Any, List, Optional
import torch
import os

from ..models.rl_model import PersonaRLAgent
from ..personas.student_personas import StudentPersona
from ..trainers.persona_trainer import ContentAction

class PersonaContentGenerator:
    """Generates personalized educational content based on student personas using trained RL model."""
    
    def __init__(self, model_path: str):
        """Initialize with a trained model."""
        action_size = ContentAction.action_space_size()
        self.agent = PersonaRLAgent(action_size=action_size)
        
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            self.agent.epsilon = 0  # No exploration during generation
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def generate_content_plan(self, persona: StudentPersona) -> dict:
        """Generate content adaptation plan for a specific student persona."""
        action_id = self.agent.choose_action(persona.to_dict())
        content_action = ContentAction.from_action_id(action_id)
        
        # Create a personalized content plan
        theme_examples = self._generate_theme_examples(persona.likes_theme, content_action.theme)
        
        return {
            "student": persona.name,
            "content_style": content_action.style,
            "difficulty_level": content_action.difficulty,
            "theme": content_action.theme,
            "theme_examples": theme_examples,
            "learning_approach": self._get_learning_approach(persona.learning_style),
            "math_level": self._get_appropriate_math_level(persona)
        }
    
    def _generate_theme_examples(self, student_theme: str, content_theme: str) -> List[str]:
        """Generate examples that connect content theme with student's preferred theme."""
        themes = {
            "Sports": [
                "Calculating the trajectory of a football pass using quadratic functions",
                "Using statistics to analyze team performance and win probabilities",
                "Calculating player efficiency ratings with linear equations"
            ],
            "Video Games": [
                "Using coordinate geometry to navigate game worlds",
                "Probability calculations for random loot drops and critical hits",
                "Calculating optimal resource allocation for game strategy"
            ],
            "Technology": [
                "Binary number systems and digital logic in smartphone processors",
                "Exponential functions modeling the growth of app downloads",
                "Using linear programming to optimize battery life"
            ],
            "Food": [
                "Proportions and ratios in cooking recipes",
                "Calculating nutritional content percentages using fractions",
                "Exponential decay modeling food spoilage rates"
            ],
            "Music": [
                "Frequency ratios in musical intervals and harmony",
                "Logarithmic scales for measuring sound intensity",
                "Wave functions and periodic functions in sound patterns"
            ],
            "Fashion": [
                "Geometry and symmetry in clothing design",
                "Scaling and proportion in pattern adjustments",
                "Calculating markup percentages and profit margins"
            ],
            "Entertainment": [
                "Revenue projection models for streaming services",
                "Probability analysis for content recommendation algorithms",
                "Statistical analysis of viewer preferences and ratings"
            ],
            "Transportation": [
                "Vector calculations for vehicle navigation",
                "Fuel efficiency calculations using rate equations",
                "Optimizing routes using linear programming"
            ],
            "Social Media": [
                "Exponential growth models for viral content",
                "Statistical analysis of engagement metrics",
                "Network theory applied to follower connections"
            ],
            "Theme Parks": [
                "Physics equations behind roller coaster designs",
                "Queuing theory and wait time calculations",
                "Optimization problems for park layout and capacity"
            ]
        }
        
        # Default examples if specific theme not found
        default_examples = [
            "Applied problem solving with real-world scenarios",
            "Visual representations of mathematical concepts",
            "Step-by-step worked examples with clear explanations"
        ]
        
        return themes.get(content_theme, default_examples)
    
    def _get_learning_approach(self, learning_style: str) -> Dict[str, Any]:
        """Extract appropriate learning approach based on learning style preference."""
        approach = {
            "explanation_type": "standard",
            "example_count": "medium",
            "visual_aids": "some",
            "interactivity": "medium"
        }
        
        # Parse learning style text to customize approach
        learning_style = learning_style.lower()
        
        if "example" in learning_style:
            approach["example_count"] = "many"
            
        if "step-by-step" in learning_style:
            approach["explanation_type"] = "procedural"
            
        if "visual" in learning_style:
            approach["visual_aids"] = "many"
            
        if "concise" in learning_style or "simple language" in learning_style:
            approach["explanation_type"] = "simplified"
            
        if "interactive" in learning_style:
            approach["interactivity"] = "high"
            
        if "bullet point" in learning_style:
            approach["explanation_type"] = "bullet-points"
            
        if "analogy" in learning_style:
            approach["explanation_type"] = "analogy-based"
            
        if "stories" in learning_style:
            approach["explanation_type"] = "narrative"
            
        return approach
    
    def _get_appropriate_math_level(self, persona: StudentPersona) -> Dict[str, Any]:
        """Determine appropriate math level based on student's mastery levels."""
        # Calculate average grade level across math subjects
        grade_sum = 0
        count = 0
        
        for subject, grade in persona.math_mastery.items():
            if grade.startswith("Grade "):
                try:
                    grade_sum += int(grade.split(" ")[1])
                    count += 1
                except ValueError:
                    pass
        
        avg_grade = grade_sum / count if count > 0 else 9  # Default to grade 9
        
        # Determine appropriate level descriptors
        if avg_grade <= 7.5:
            level = "Foundational"
            prerequisites = ["Basic arithmetic", "Elementary algebra"]
        elif avg_grade <= 8.5:
            level = "Intermediate"
            prerequisites = ["Pre-algebra", "Basic geometry"]
        elif avg_grade <= 9.5:
            level = "Standard"
            prerequisites = ["Algebra I", "Basic trigonometry"]
        elif avg_grade <= 10.5:
            level = "Advanced"
            prerequisites = ["Algebra II", "Geometry", "Trigonometry"]
        else:
            level = "Advanced+"
            prerequisites = ["Pre-calculus", "Advanced algebra", "Trigonometry"]
        
        return {
            "level": level,
            "avg_grade": avg_grade,
            "prerequisites": prerequisites
        }
    
    def batch_generate_plans(self, personas: List[StudentPersona]) -> List[Dict[str, Any]]:
        """Generate content plans for multiple personas."""
        return [self.generate_content_plan(persona) for persona in personas]
