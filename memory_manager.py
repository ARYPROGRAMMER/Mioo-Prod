import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class MemoryManager:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.long_term_memories = {}  # user_id -> list of memories
        self.memory_importance_threshold = 0.6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for a text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings.cpu().numpy()[0]
    
    def calculate_importance(self, memory: Dict) -> float:
        """Calculate memory importance based on various factors"""
        # Base importance from stored value
        importance = memory.get("importance", 0.5)
        
        # Decay based on age
        days_old = (datetime.utcnow() - memory["timestamp"]).days
        age_factor = max(0.1, 1.0 - (days_old / 30.0))  # Decay over a month
        
        # Boost for emotional content
        emotion_boost = 0.0
        if memory.get("emotion"):
            if memory["emotion"] in ["frustrated", "confused"]:
                emotion_boost = 0.2
            elif memory["emotion"] in ["satisfied", "curious"]:
                emotion_boost = 0.1
                
        # Boost for repeated topics
        repetition_boost = min(0.3, memory.get("access_count", 0) * 0.05)
        
        # Calculate final importance
        final_importance = importance * age_factor + emotion_boost + repetition_boost
        return min(1.0, final_importance)
    
    def add_memory(self, user_id: str, text: str, context: Dict) -> None:
        """Add new memory for a user"""
        if user_id not in self.long_term_memories:
            self.long_term_memories[user_id] = []
            
        # Create embedding
        embedding = self.create_embedding(text)
        
        # Create memory entry
        memory = {
            "text": text,
            "embedding": embedding,
            "timestamp": datetime.utcnow(),
            "context": context,
            "importance": 0.5,  # Initial importance
            "access_count": 0,
            "emotion": context.get("emotion", "neutral")
        }
        
        # Store memory
        self.long_term_memories[user_id].append(memory)
        
        # Limit memory size and remove least important memories if needed
        if len(self.long_term_memories[user_id]) > 100:
            # Calculate importance for all memories
            for mem in self.long_term_memories[user_id]:
                mem["current_importance"] = self.calculate_importance(mem)
                
            # Sort by importance and keep most important
            self.long_term_memories[user_id].sort(key=lambda x: x["current_importance"], reverse=True)
            self.long_term_memories[user_id] = self.long_term_memories[user_id][:100]
    
    def retrieve_relevant_memories(self, user_id: str, query: str, max_results: int = 3) -> List[Dict]:
        """Retrieve memories relevant to the current query"""
        if user_id not in self.long_term_memories or not self.long_term_memories[user_id]:
            return []
            
        # Create query embedding
        query_embedding = self.create_embedding(query)
        
        # Calculate similarity with all memories
        memories = self.long_term_memories[user_id]
        similarities = []
        
        for i, memory in enumerate(memories):
            embedding = memory["embedding"]
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            current_importance = self.calculate_importance(memory)
            
            # Combined relevance score
            relevance = similarity * 0.7 + current_importance * 0.3
            similarities.append((i, relevance))
            
        # Sort by relevance
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return most relevant memories
        result = []
        for i, relevance in similarities[:max_results]:
            if relevance > 0.5:  # Only return if relevance is above threshold
                memory = memories[i].copy()
                memory["relevance"] = relevance
                # Update access count
                memories[i]["access_count"] = memories[i].get("access_count", 0) + 1
                result.append(memory)
                
        return result
    
    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """Format memories for inclusion in a prompt"""
        if not memories:
            return "No relevant past interactions."
            
        result = "Relevant past interactions:\n"
        for memory in memories:
            days_ago = (datetime.utcnow() - memory["timestamp"]).days
            time_str = f"{days_ago} days ago" if days_ago > 0 else "today"
            result += f"- {time_str}: {memory['text'][:100]}{'...' if len(memory['text']) > 100 else ''}\n"
            
        return result
