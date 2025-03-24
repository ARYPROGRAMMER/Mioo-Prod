from openai import OpenAI
from typing import Dict, List, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class AssistantsManager:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self._initialized = False
        self.assistants: Dict[str, str] = {}
        
    async def initialize(self) -> bool:
        """Initialize assistants"""
        try:
            # Create default tutoring assistant
            tutor = await self._create_tutor_assistant()
            self.assistants["tutor"] = tutor
            self._initialized = True
            logger.info("Assistants initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize assistants: {str(e)}")
            return False
            
    async def _create_tutor_assistant(self) -> str:
        """Create a tutoring assistant"""
        try:
            assistant = self.client.beta.assistants.create(
                name="Math Tutor",
                instructions="""You are an adaptive AI math tutor. Your role is to:
                1. Explain concepts clearly using the specified teaching style
                2. Adapt explanations based on student's learning preferences
                3. Provide appropriate examples using student's interests
                4. Maintain consistent difficulty level as specified
                5. Encourage active learning and engagement
                6. Show empathy and patience""",
                model="gpt-4-1106-preview",
                tools=[{"type": "code_interpreter"}]
            )
            return assistant.id
        except Exception as e:
            logger.error(f"Error creating tutor assistant: {str(e)}")
            raise
    
    async def get_assistant(self, assistant_type: str = "tutor") -> Optional[str]:
        """Get assistant ID by type"""
        if not self._initialized:
            await self.initialize()
        return self.assistants.get(assistant_type)
