import os
from typing import Dict, List, Optional, Any
from openai import OpenAI
import logging
import json

logger = logging.getLogger(__name__)

class AssistantsManager:
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.assistants: Dict[str, Any] = {}
        self.threads: Dict[str, Any] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the assistants manager"""
        try:
            # Create a default assistant if it doesn't exist
            if not self._initialized:
                default_assistant = self._get_or_create_assistant(
                    name="Mioo - Personalized AI Tutor",
                    instructions="You are Mioo, an advanced AI tutor that personalizes learning for each student.",
                    model="gpt-4-turbo-preview"
                )
                self.assistants["default"] = default_assistant
                self._initialized = True
                logger.info("Assistants manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize assistants manager: {str(e)}")
            raise
    
    def _get_or_create_assistant(self, name: str, instructions: str, model: str) -> Any:
        """Get an existing assistant or create a new one"""
        # List assistants to check if one with this name already exists
        existing_assistants = self.client.beta.assistants.list(
            order="desc",
            limit=100,
        )
        
        for assistant in existing_assistants.data:
            if assistant.name == name:
                logger.info(f"Found existing assistant: {name}")
                return assistant
        
        # Create a new assistant
        logger.info(f"Creating new assistant: {name}")
        return self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
            tools=[{"type": "code_interpreter"}]
        )
    
    def get_or_create_thread(self, user_id: str) -> Any:
        """Get or create a thread for the user"""
        if user_id in self.threads:
            return self.threads[user_id]
        
        thread = self.client.beta.threads.create()
        self.threads[user_id] = thread
        return thread
    
    async def update_assistant_instructions(self, 
                                           assistant_id: str, 
                                           instructions: str, 
                                           teaching_strategy: Dict[str, str] = None) -> Any:
        """Update the assistant's instructions based on teaching strategy"""
        try:
            assistant = self.client.beta.assistants.update(
                assistant_id=assistant_id,
                instructions=instructions
            )
            return assistant
        except Exception as e:
            logger.error(f"Error updating assistant instructions: {str(e)}")
            raise
    
    async def send_message(self, 
                          user_id: str, 
                          message: str, 
                          teaching_strategy: Dict[str, str],
                          user_state: Dict[str, Any]) -> str:
        """Send a message to the assistant and get the response"""
        try:
            # Get or create thread for this user
            thread = self.get_or_create_thread(user_id)
            
            # Create a message in the thread
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message
            )
            
            # Create personalized instructions based on teaching strategy and user state
            instructions = self._create_personalized_instructions(teaching_strategy, user_state)
            
            # Update the assistant with personalized instructions
            assistant = await self.update_assistant_instructions(
                assistant_id=self.assistants["default"].id,
                instructions=instructions
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for the run to complete
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                if run_status.status in ["failed", "cancelled", "expired"]:
                    raise Exception(f"Assistant run failed with status: {run_status.status}")
            
            # Retrieve messages
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Get the latest assistant message
            for message in messages.data:
                if message.role == "assistant":
                    return message.content[0].text.value
            
            return "I'm sorry, I couldn't generate a response."
        
        except Exception as e:
            logger.error(f"Error in assistant communication: {str(e)}")
            raise
    
    def _create_personalized_instructions(self, 
                                         teaching_strategy: Dict[str, str], 
                                         user_state: Dict[str, Any]) -> str:
        """Create personalized instructions based on teaching strategy and user state"""
        # Format learning style
        learning_style = user_state.get('learning_style', {})
        style_formatted = ", ".join([f"{k}: {v:.1f}" for k, v in learning_style.items()])
        
        # Get communication preferences
        comm_style = user_state.get("communication_style", {})
        vocabulary_level = comm_style.get("vocabulary_level", 0.5)
        formality = comm_style.get("formality", 0.5)
        verbosity = comm_style.get("verbosity", 0.5)
        
        # Get emotional context
        emotional_context = user_state.get("emotional_context", {})
        recent_emotions = emotional_context.get("recent_emotions", [])
        dominant_emotion = "neutral"
        if recent_emotions:
            dominant_emotion = recent_emotions[-1] if isinstance(recent_emotions[-1], str) else "neutral"
        
        # Format mastery levels
        mastery_levels = user_state.get("topic_mastery", {})
        mastery_formatted = ", ".join([f"{topic}: {level:.1f}" for topic, level in mastery_levels.items()])
        
        instructions = f"""You are Mioo, an advanced AI tutor optimizing for personalized learning outcomes.

Teaching Parameters:
- Style: {teaching_strategy['style']}
- Complexity: {teaching_strategy['complexity']}
- Examples: {teaching_strategy['examples']}

User Context:
- Knowledge Level: {user_state.get('knowledge_level', 0.5):.1f}
- Current Topics: {', '.join(user_state.get('recent_topics', []))}
- Interests: {', '.join(user_state.get('interests', []))}
- Learning Style: {style_formatted}

Communication Preferences:
- Vocabulary Level: {"Advanced" if vocabulary_level > 0.7 else "Intermediate" if vocabulary_level > 0.4 else "Basic"}
- Formality: {"Formal" if formality > 0.7 else "Conversational" if formality > 0.4 else "Casual"}
- Verbosity: {"Detailed" if verbosity > 0.7 else "Balanced" if verbosity > 0.4 else "Concise"}

Emotional Context:
- Current Emotional State: {dominant_emotion}

Topic Mastery:
{mastery_formatted if mastery_formatted else "No specific topic mastery data available yet."}

Instructions:
1. Adapt explanation depth based on topic mastery
2. Use relevant examples from user's interests
3. Match the user's communication style in vocabulary and formality
4. Address emotional state appropriately
5. Maintain consistent complexity level
6. Include specific checkpoints for understanding
7. Encourage active engagement through questions
"""
        return instructions
