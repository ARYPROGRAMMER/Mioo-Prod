import openai
import json
import os
import asyncio
from typing import Dict, List, Any, Optional
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModelOrchestrator:
    """Orchestrates multiple LLM models to optimize cost and performance"""
    
    def __init__(self, openai_client=None, api_key=None):
        """Initialize the LLM orchestrator with OpenAI client"""
        self.openai_client = openai_client or openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.available_models = {
            'gpt-4o': {
                'context_window': 128000,
                'cost_per_1k_in': 0.005,
                'cost_per_1k_out': 0.015,
                'priority': 1  # Higher is better for complex tasks
            },
            'gpt-3.5-turbo': {
                'context_window': 16000,
                'cost_per_1k_in': 0.0010,
                'cost_per_1k_out': 0.0020,
                'priority': 0  # Lower priority, used for simpler tasks
            }
        }
    
    def _select_model(self, query_complexity: float, user_state: Dict[str, Any]) -> str:
        """Select the appropriate model based on query and user state"""
        default_model = 'gpt-4o'  # Default to most capable model
        
        # If query is simple and user is engaged, can use cheaper model
        if query_complexity < 0.4 and user_state.get('engagement', 0) > 0.7:
            return 'gpt-3.5-turbo'
            
        # If user is confused or knowledge level is low, use more capable model
        if user_state.get('knowledge_level', 0.5) < 0.3:
            return default_model
            
        # For most educational contexts, prefer the more capable model
        return default_model
    
    async def generate_response(self, 
                              message: str, 
                              user_state: Dict[str, Any], 
                              teaching_strategy: Dict[str, str],
                              chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate a response using the appropriate LLM model"""
        try:
            # Estimate query complexity (placeholder for more sophisticated analysis)
            query_complexity = min(1.0, len(message) / 1000 + 0.3)
            
            # Select appropriate model
            model = self._select_model(query_complexity, user_state)
            logger.info(f"Selected model: {model} for query complexity: {query_complexity}")
            
            # Prepare system prompt with teaching strategy
            system_prompt = self._create_system_prompt(teaching_strategy, user_state)
            
            # Prepare context from chat history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history context if available
            if chat_history:
                # Limit to last 10 messages to manage context size
                for msg in chat_history[-10:]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                    
            # Add the current user message
            messages.append({"role": "user", "content": message})
            
            # Make the API call
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=0.95
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Calculate token usage for monitoring
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # Calculate estimated cost
            model_info = self.available_models[model]
            estimated_cost = (
                (input_tokens / 1000) * model_info['cost_per_1k_in'] + 
                (output_tokens / 1000) * model_info['cost_per_1k_out']
            )
            
            return {
                "response": content,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": estimated_cost,
                "completion_id": response.id
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response
            return {
                "response": "I'm sorry, I'm having trouble processing your request right now. Could you please try again?",
                "model": "fallback",
                "error": str(e)
            }
    
    def _create_system_prompt(self, teaching_strategy: Dict[str, str], user_state: Dict[str, Any]) -> str:
        """Create a tailored system prompt based on teaching strategy and user state"""
        knowledge_level = user_state.get('knowledge_level', 0.5)
        recent_topics = user_state.get('recent_topics', [])
        interests = user_state.get('interests', [])
        
        # Convert knowledge level to descriptive term
        if knowledge_level < 0.3:
            knowledge_desc = "beginner"
        elif knowledge_level < 0.7:
            knowledge_desc = "intermediate"
        else:
            knowledge_desc = "advanced"
            
        style = teaching_strategy.get('style', 'detailed')
        complexity = teaching_strategy.get('complexity', 'medium')
        examples = teaching_strategy.get('examples', 'some')
        
        # Build the system prompt
        system_prompt = f"""You are an AI tutor specialized in personalized education.

TEACHING STYLE: {style}
COMPLEXITY LEVEL: {complexity}
EXAMPLES: {examples}

LEARNER PROFILE:
- Knowledge level: {knowledge_desc}
- Recent topics: {', '.join(recent_topics) if recent_topics else 'None yet'}
- Interests: {', '.join(interests) if interests else 'Not specified yet'}

TEACHING GUIDELINES BASED ON STRATEGY:
"""
        
        # Add specific guidelines based on teaching style
        if style == 'detailed':
            system_prompt += "- Provide thorough explanations with in-depth coverage\n- Make connections between concepts\n- Define technical terms clearly\n"
        elif style == 'concise':
            system_prompt += "- Be direct and to the point\n- Focus on key takeaways\n- Use brief, clear language\n"
        elif style == 'interactive':
            system_prompt += "- Ask guiding questions throughout your response\n- Create opportunities for reflection\n- Include small exercises or challenges\n"
        elif style == 'analogy-based':
            system_prompt += "- Use metaphors and comparisons to familiar concepts\n- Connect abstract ideas to concrete examples\n- Create visual mental models\n"
        elif style == 'step-by-step':
            system_prompt += "- Break down concepts into sequential steps\n- Number your points clearly\n- Build progressively from basic to advanced\n"
            
        # Add complexity guidelines
        system_prompt += f"\nCOMPLEXITY GUIDELINES ({complexity}):\n"
        if complexity == 'low':
            system_prompt += "- Use simple vocabulary and short sentences\n- Avoid jargon unless necessary and always explain it\n- Focus on foundational concepts\n"
        elif complexity == 'medium':
            system_prompt += "- Balance technical accuracy with accessibility\n- Introduce some field-specific terminology with explanations\n- Make connections to prerequisite knowledge\n"
        elif complexity == 'high':
            system_prompt += "- Use precise technical language appropriate for the subject\n- Explore nuances and edge cases\n- Reference advanced concepts when relevant\n"
            
        # Add examples guidelines
        system_prompt += f"\nEXAMPLES APPROACH ({examples}):\n"
        if examples == 'few':
            system_prompt += "- Use 1-2 focused, clear examples\n- Choose the most illustrative cases\n"
        elif examples == 'some':
            system_prompt += "- Use 2-3 diverse examples to illustrate key points\n- Include both simple and moderately complex cases\n"
        elif examples == 'many':
            system_prompt += "- Provide multiple examples covering different aspects\n- Include basic, intermediate, and edge cases\n- Use real-world applications where possible\n"
            
        system_prompt += "\nYour goal is to help the learner understand concepts deeply and develop their knowledge. Adapt to their questions while following the teaching strategy outlined above."
        
        return system_prompt
