import openai
import json
import os
import asyncio
from typing import Dict, List, Any, Optional
import logging
import re

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
    
    def _get_preferred_learning_style(self, user_state: Dict[str, Any]) -> str:
        """Determine user's preferred learning style based on their state"""
        learning_style = user_state.get("learning_style", {})
        
        if not learning_style:
            return "balanced"
            
        styles = {
            "visual": learning_style.get("visual", 0.5),
            "interactive": learning_style.get("interactive", 0.5),
            "theoretical": learning_style.get("theoretical", 0.5),
            "practical": learning_style.get("practical", 0.5)
        }
        
        # Find the dominant learning style
        max_style = max(styles.items(), key=lambda x: x[1])
        
        # If there's a clear preference (>0.6), return that style
        if max_style[1] > 0.6:
            return max_style[0]
            
        # If visual and interactive are both high
        if styles["visual"] > 0.5 and styles["interactive"] > 0.5:
            return "visual-interactive"
            
        # If theoretical and practical are both high
        if styles["theoretical"] > 0.5 and styles["practical"] > 0.5:
            return "conceptual-applied"
            
        return "balanced"
    
    def _build_conversation_context(self, chat_history: List[Dict[str, str]], user_state: Dict[str, Any]) -> str:
        """Build detailed conversation context from history"""
        if not chat_history:
            return "No previous context available."
            
        context_parts = []
        
        # Get last query and response
        last_exchange = chat_history[-2:] if len(chat_history) >= 2 else chat_history
        
        # Track conversation thread
        current_topic = None
        for msg in last_exchange:
            if msg["role"] == "user":
                current_topic = self._detect_current_focus(msg["content"], user_state)
            context_parts.append(f"{msg['role'].capitalize()}: {msg['content'][:100]}...")
            
        # Add user's interests if relevant to current topic
        if current_topic and user_state.get("interests"):
            relevant_interests = [
                interest for interest in user_state["interests"]
                if interest.lower() in current_topic.lower()
            ]
            if relevant_interests:
                context_parts.append(f"Relevant interests: {', '.join(relevant_interests)}")
                
        return "\n".join(context_parts)

    async def generate_response(self, 
                              message: str, 
                              user_state: Dict[str, Any], 
                              teaching_strategy: Dict[str, str],
                              chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate a response using the appropriate LLM model with enhanced personalization"""
        try:
            # Build rich context
            conversation_context = self._build_conversation_context(chat_history, user_state)
            current_focus = self._detect_current_focus(message, user_state)
            
            # Enhanced system prompt with better context awareness
            system_prompt = f"""You are a highly personalized AI tutor.

CURRENT CONTEXT:
{conversation_context}

CURRENT FOCUS: {current_focus}

USER PROFILE:
- Name: {user_state.get('name', 'User')}
- Interests: {', '.join(user_state.get('interests', ['Not specified']))}
- Knowledge Level: {self._describe_knowledge_level(user_state.get('knowledge_level', 0.5))}

RESPONSE GUIDELINES:
1. Always maintain context continuity from previous messages
2. Reference relevant past interactions when appropriate
3. Use examples from user's interests: {', '.join(user_state.get('interests', []))}
4. Follow teaching style: {teaching_strategy['style']}

Your task is to provide a coherent, contextual response that builds on the conversation history.
"""
            # Extract current interaction context
            current_context = self._build_interaction_context(message, user_state, chat_history)
            
            # Build personalized system prompt
            system_prompt = self._build_personalized_prompt(
                user_state, 
                teaching_strategy,
                current_context
            )

            # Estimate query complexity (placeholder for more sophisticated analysis)
            query_complexity = min(1.0, len(message) / 1000 + 0.3)
            
            # Select appropriate model
            model = self._select_model(query_complexity, user_state)
            logger.info(f"Selected model: {model} for query complexity: {query_complexity}")
            
            # Create detailed user profile from state for stronger personalization
            knowledge_level = user_state.get('knowledge_level', 0.5)
            engagement = user_state.get('engagement', 0.5)
            
            # Personalization factors
            topics_string = ", ".join(user_state.get("recent_topics", ["general knowledge"]))
            preferred_style = self._get_preferred_learning_style(user_state)
            
            # Create more detailed conversation context
            conversation_context = ""
            if chat_history and len(chat_history) > 0:
                # Extract conversation topics and detect patterns
                recent_topics = set()
                for msg in chat_history[-5:]:
                    if msg.get("content"):
                        # Simple keyword extraction for topic detection
                        words = msg["content"].lower().split()
                        important_words = [w for w in words if len(w) > 4 and w not in ("about", "would", "could", "should")]
                        recent_topics.update(important_words[:3])  # Add up to 3 important words
                
                conversation_context = f"The conversation has covered these topics: {', '.join(recent_topics)[:100]}. "
                
                # Get the most recent assistant and user messages for context
                last_assistant_msg = next((msg["content"] for msg in reversed(chat_history) 
                                        if msg["role"] == "assistant"), "")
                last_user_msg = next((msg["content"] for msg in reversed(chat_history) 
                                   if msg["role"] == "user"), "")
                
                # Add contextual information about how the conversation is going
                if last_user_msg and last_user_msg.strip().lower() in ["no", "wrong", "incorrect"]:
                    conversation_context += "The user has expressed disagreement with your previous response. "
                    conversation_context += "Try a completely different approach to explain the concept. "
            
            # Create enhanced personalized system prompt with better context awareness
            system_prompt = f"""You are a highly personalized AI tutor with perfect memory of the conversation.

USER PROFILE:
- Knowledge level: {knowledge_level:.2f}/1.0 ({self._describe_knowledge_level(knowledge_level)})
- Engagement level: {engagement:.2f}/1.0 ({self._describe_engagement_level(engagement)})
- Learning style preference: {preferred_style}
- Topics of interest: {topics_string}

CONVERSATION CONTEXT:
{conversation_context}

TEACHING APPROACH:
- Style: {teaching_strategy['style']}
- Complexity: {teaching_strategy['complexity']}
- Examples: {teaching_strategy['examples']}

IMPORTANT INSTRUCTIONS:
1. Maintain continuity with previous messages - don't repeat information
2. If the user expresses disagreement, try a completely different approach
3. Adapt to the user's learning style and knowledge level
4. Stay focused on the topics relevant to the conversation
5. Be concise but thorough, respecting the user's time and intelligence

Your goal is to help this specific user learn effectively based on their individual profile and our conversation history.
"""
            
            # Prepare context from chat history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history context if available
            if chat_history:
                # Include more history for better context
                for msg in chat_history[-8:]:  # Include last 8 messages for better context
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                    
            # Add the current user message
            messages.append({"role": "user", "content": message})
            
            # Make the API call with enhanced parameters
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=0.6,  # Slightly lower temperature for more consistent responses
                max_tokens=1200,  # Increased max tokens for more comprehensive responses
                top_p=0.95,
                presence_penalty=0.6,  # Added presence penalty to reduce repetition
                frequency_penalty=0.5   # Added frequency penalty to encourage diversity
            )
            
            # Extract response content
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
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
    
    def _describe_knowledge_level(self, level: float) -> str:
        """Convert numerical knowledge level to descriptive text"""
        if level < 0.3:
            return "beginner, needs foundational concepts"
        elif level < 0.6:
            return "intermediate, building understanding"
        elif level < 0.8:
            return "advanced, refining knowledge"
        else:
            return "expert, seeking nuanced insights"
    
    def _describe_engagement_level(self, level: float) -> str:
        """Convert numerical engagement level to descriptive text"""
        if level < 0.3:
            return "low engagement, needs motivation"
        elif level < 0.6:
            return "moderately engaged, responds to interactive elements"
        else:
            return "highly engaged, responds well to challenging content"
    
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

    def _extract_user_info(self, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract user information from chat history"""
        info = {"name": None, "interests": [], "skill_level": {}}
        
        if not chat_history:
            return info
            
        for msg in chat_history:
            if msg["role"] != "user":
                continue
                
            content = msg["content"].lower()
            
            # Extract name
            name_patterns = [
                r"i am (\w+)",
                r"my name is (\w+)",
                r"i'm (\w+)",
            ]
            
            for pattern in name_patterns:
                if match := re.search(pattern, content):
                    info["name"] = match.group(1).capitalize()
                    
            # Extract interests
            interest_patterns = [
                r"i (?:like|love|enjoy|prefer) (\w+(?:\+\+)?)",
                r"i'm interested in (\w+(?:\+\+)?)",
            ]
            
            for pattern in interest_patterns:
                if matches := re.finditer(pattern, content):
                    info["interests"].extend(match.group(1) for match in matches)
                    
        return info

    def _check_if_introduction(self, message: str) -> bool:
        """Check if message is a user introduction"""
        intro_patterns = [
            r"(?i)i am \w+",
            r"(?i)my name is \w+",
            r"(?i)i'm \w+",
            r"(?i)i like \w+",
        ]
        return any(re.search(pattern, message) for pattern in intro_patterns)

    def _create_user_context(self, user_state: Dict[str, Any], 
                           chat_history: List[Dict[str, str]]) -> str:
        """Create personalized context string"""
        context = []
        
        if name := user_state.get("name"):
            context.append(f"You're talking to {name}.")
            
        if interests := user_state.get("interests"):
            context.append(f"They are interested in: {', '.join(interests)}.")
            
        return "\n".join(context)
    
    def _extract_introduction_info(self, message: str) -> Dict[str, Any]:
        """Extract introduction information from a message"""
        intro_info = {}
        message = message.lower()
        
        # Extract name
        name_patterns = [
            r"(?:i am|my name is|i'm) (\w+)",
            r"(?:call me) (\w+)",
        ]
        
        for pattern in name_patterns:
            if match := re.search(pattern, message):
                intro_info["name"] = match.group(1).capitalize()
                break
                
        # Extract interests and preferences
        interest_patterns = [
            (r"i (?:like|love|enjoy|prefer) (\w+(?:\+\+)?)", "interests"),
            (r"i'm (?:interested in|learning|studying) (\w+(?:\+\+)?)", "interests"),
            (r"i'm (?:a|an) (\w+)(?: programmer| developer)?", "role"),
        ]
        
        for pattern, key in interest_patterns:
            if matches := re.finditer(pattern, message):
                if key not in intro_info:
                    intro_info[key] = []
                intro_info[key].extend(match.group(1).lower() for match in matches)
                
        # Remove duplicates from lists
        for key in ["interests", "role"]:
            if key in intro_info and isinstance(intro_info[key], list):
                intro_info[key] = list(dict.fromkeys(intro_info[key]))
                
        return intro_info

    def _get_relevant_history(self, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Extract relevant context from chat history"""
        if not chat_history or len(chat_history) == 0:
            return "This is the start of the conversation."
            
        # Get last few messages for context
        recent_messages = chat_history[-3:]  # Last 3 messages
        context_snippets = []
        
        # Create context from recent messages
        for msg in recent_messages:
            if msg["role"] == "assistant":
                # Summarize long responses
                content = msg["content"]
                if len(content) > 100:
                    content = content[:100] + "..."
                context_snippets.append(f"I explained: {content}")
            else:
                content = msg["content"]
                if len(content) > 50:
                    content = content[:50] + "..."
                context_snippets.append(f"User asked/said: {content}")
                
        # Return formatted context
        return " ".join(context_snippets)

    def _build_user_context(self, user_state: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Build comprehensive user context"""
        contexts = []
        
        # Personal context
        if name := user_state.get('name'):
            contexts.append(f"Current User: {name}")
        
        # Interest context
        if interests := user_state.get('interests'):
            contexts.append(f"Primary Interests: {', '.join(interests)}")
            if 'c++' in [i.lower() for i in interests]:
                contexts.append("Special Focus: Programming with C++ examples when relevant")
        
        # Interaction history summary
        if chat_history:
            last_topics = self._extract_recent_topics(chat_history[-3:])
            if last_topics:
                contexts.append(f"Recent Discussion Topics: {', '.join(last_topics)}")
        
        return '\n'.join(contexts)

    def _get_current_focus(self, chat_history: List[Dict[str, str]]) -> str:
        """Extract current conversation focus"""
        if not chat_history:
            return "Initial interaction"
            
        last_messages = chat_history[-2:]  # Look at last 2 messages
        topics = set()
        
        for msg in last_messages:
            content = msg['content'].lower()
            # Check for specific topics/questions
            if 'c++' in content:
                topics.add('C++ programming')
            elif any(math_term in content for math_term in ['=', '+', '-', '*', '/']):
                topics.add('Mathematics')
            elif '?' in content:
                topics.add('Question/Inquiry')
                
        return ', '.join(topics) if topics else "General conversation"

    def _extract_recent_topics(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract topics from recent messages"""
        topics = set()
        for msg in messages:
            content = msg['content'].lower()
            # Add topic detection logic here
            if 'c++' in content:
                topics.add('C++')
            if any(term in content for term in ['=', '+', '-', '*', '/']):
                topics.add('Mathematics')
            # Add more topic detection as needed
        return list(topics)

    def _build_interaction_context(self, 
                                 message: str, 
                                 user_state: Dict[str, Any],
                                 chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build comprehensive interaction context"""
        current_focus = self._detect_current_focus(message, user_state)
        interaction_type = self._classify_interaction(message)
        
        return {
            "user_profile": {
                "name": user_state.get("name", ""),
                "interests": user_state.get("interests", []),
                "expertise": self._get_expertise_areas(user_state),
                "knowledge_level": user_state.get("knowledge_level", 0.5)
            },
            "interaction_type": interaction_type,
            "current_focus": current_focus,
            "relevant_history": self._get_relevant_history(chat_history),
            "has_context": bool(chat_history and len(chat_history) > 0)
        }

    def _build_personalized_prompt(self, 
                                 user_state: Dict[str, Any],
                                 teaching_strategy: Dict[str, str],
                                 context: Dict[str, Any]) -> str:
        """Create highly personalized system prompt"""
        name = user_state.get("name", "the user")
        interests = user_state.get("interests", [])
        
        prompt = [
            f"You are a personalized AI tutor. When interacting with {name}:",
            "",
            "PERSONALIZATION RULES:",
            f"1. Always acknowledge them as {name}",
            f"2. Use examples from their interests: {', '.join(interests)}" if interests else "",
            f"3. Match their expertise level in {self._get_expertise_areas(user_state)}",
            "4. Maintain conversation continuity and context",
            "",
            "CURRENT CONTEXT:",
            f"- Interaction Type: {context['interaction_type']}",
            f"- Current Focus: {context['current_focus']}",
            "- Recent History: " + self._summarize_history(context['relevant_history']),
            "",
            "TEACHING APPROACH:",
            f"- Style: {teaching_strategy['style']}",
            f"- Complexity: {teaching_strategy['complexity']}",
            f"- Examples: {teaching_strategy['examples']}"
        ]
        
        return "\n".join([line for line in prompt if line])

    def _classify_interaction(self, message: str) -> str:
        """Classify the type of interaction"""
        message = message.lower()
        if "who am i" in message:
            return "identity_query"
        if any(op in message for op in "+-*/="):
            return "mathematical"
        if any(term in message for term in ["what is", "how do", "explain"]):
            return "educational"
        if any(term in message for term in ["hi ", "hello", "hey"]):
            return "greeting"
        return "general"

    def _get_expertise_areas(self, user_state: Dict[str, Any]) -> List[str]:
        """Determine user's areas of expertise"""
        expertise = []
        if "topic_mastery" in user_state:
            expertise.extend(
                topic for topic, level in user_state["topic_mastery"].items() 
                if level > 0.7
            )
        if "interests" in user_state:
            expertise.extend(user_state["interests"])
        return list(set(expertise))

    def _detect_current_focus(self, message: str, user_state: Dict[str, Any]) -> str:
        """Detect the current focus of conversation"""
        message = message.lower()
        
        # Identity queries
        identity_patterns = ["who am i", "what do i know", "my name", "about me"]
        if any(pattern in message for pattern in identity_patterns):
            return f"User Identity ({user_state.get('name', 'Unknown')})"
        
        # Programming/C++ focus
        if any(term in message for term in ["c++", "code", "program", "function", "class"]):
            if "c++" in user_state.get("interests", []):
                return "C++ Programming (User's Interest)"
            return "Programming Concepts"
        
        # Mathematical operations
        math_patterns = ['+', '-', '*', '/', '=', 'calculate', 'solve']
        if any(op in message for op in math_patterns):
            return "Mathematical Calculation"
        
        # Check against user interests
        for interest in user_state.get("interests", []):
            if interest.lower() in message:
                return f"{interest.capitalize()} (User Interest)"
        
        # Check recent topics for continuity
        recent_topics = user_state.get("recent_topics", [])[-3:]
        for topic in recent_topics:
            if topic.lower() in message:
                return f"Continuing {topic}"
            
        # Default focus
        if "?" in message:
            return "New Question"
        return "General Discussion"

    def _summarize_history(self, history: str) -> str:
        """Summarize conversation history for context"""
        if "start of the conversation" in history:
            return "New conversation"
            
        # Extract key points from history
        points = []
        for line in history.split(". "):
            if "explained:" in line.lower():
                summary = line.split("explained: ")[1][:50] + "..."
                points.append(f"Previous explanation about {summary}")
            elif "asked:" in line.lower():
                query = line.split("asked: ")[1][:30] + "..."
                points.append(f"Question about {query}")
                
        return " | ".join(points) if points else "Continuing discussion"
