import json
from typing import Dict, List, Any, Optional

class StudentPersona:
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get('name', '')
        self.education_level = data.get('educationLevel', '')
        self.sat_act_timeline = data.get('satActTimeline', '')
        self.math_mastery = data.get('mathMastery', {})
        self.likes_theme = data.get('likesTheme', '')
        self.learning_style = data.get('learningStyle', '')
    
    def __repr__(self) -> str:
        return f"StudentPersona(name={self.name}, education_level={self.education_level})"
    
    def get_math_level(self, subject: str) -> str:
        """Get the student's mastery level for a specific math subject."""
        return self.math_mastery.get(subject, "Unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary for model input."""
        return {
            "name": self.name,
            "education_level": self.education_level,
            "sat_act_timeline": self.sat_act_timeline,
            "math_mastery": self.math_mastery,
            "likes_theme": self.likes_theme,
            "learning_style": self.learning_style
        }

class PersonaManager:
    def __init__(self):
        self.personas: List[StudentPersona] = []
    
    def load_from_json(self, json_data: str) -> None:
        """Load personas from JSON string."""
        data = json.loads(json_data)
        student_data = data.get('studentPersonas', [])
        self.personas = [StudentPersona(student) for student in student_data]
    
    def load_from_file(self, file_path: str) -> None:
        """Load personas from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        student_data = data.get('studentPersonas', [])
        self.personas = [StudentPersona(student) for student in student_data]
    
    def get_persona_by_name(self, name: str) -> Optional[StudentPersona]:
        """Get a specific persona by name."""
        for persona in self.personas:
            if persona.name.lower() == name.lower():
                return persona
        return None
