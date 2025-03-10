import motor.motor_asyncio
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from bson.objectid import ObjectId
import json

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB client for storing and retrieving user data"""
    
    def __init__(self, connection_string=None):
        """Initialize MongoDB connection"""
        conn_str = connection_string or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.environ.get("MONGODB_DB", "mioo_tutor")
        
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(conn_str)
            self.db = self.client[db_name]
            self.users_collection = self.db["users"]
            self.sessions_collection = self.db["learning_sessions"]
            self.metrics_collection = self.db["learning_metrics"]
            self.feedback_collection = self.db["feedback"]
            logger.info(f"Connected to MongoDB: {db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def create_user(self, user_id: str, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new user with default or provided initial state"""
        if await self.get_user(user_id):
            logger.info(f"User {user_id} already exists")
            return await self.get_user(user_id)
            
        default_state = {
            "user_id": user_id,
            "knowledge_level": 0.5,
            "engagement": 0.5,
            "interests": [],
            "recent_topics": [],
            "performance": 0.5,
            "chat_history": [],
            "learning_history": [],
            "last_updated": datetime.utcnow().isoformat(),
            "session_metrics": {
                "messages_count": 0,
                "average_response_time": 0,
                "topics_covered": [],
                "learning_rate": 0,
                "engagement_trend": [],
            }
        }
        
        # Merge default with provided initial state if any
        user_state = {**default_state, **(initial_state or {})}
        
        try:
            await self.users_collection.insert_one(user_state)
            logger.info(f"Created new user: {user_id}")
            return user_state
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = await self.users_collection.find_one({"user_id": user_id})
            if user:
                # Convert ObjectId to string for JSON serialization
                if "_id" in user:
                    user["_id"] = str(user["_id"])
                    
            return user
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None
    
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user data"""
        try:
            # Ensure we're not overwriting the user_id
            if 'user_id' in update_data:
                del update_data['user_id']
                
            # Add last updated timestamp
            update_data['last_updated'] = datetime.utcnow().isoformat()
            
            result = await self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                logger.warning(f"No user found with ID {user_id}")
                return False
                
            logger.info(f"Updated user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            return False
    
    async def add_message_to_history(self, user_id: str, message: Dict[str, Any]) -> bool:
        """Add a message to user's chat history"""
        try:
            result = await self.users_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {"chat_history": message},
                    "$inc": {"session_metrics.messages_count": 1},
                    "$set": {"last_updated": datetime.utcnow().isoformat()}
                }
            )
            
            return result.matched_count > 0
        except Exception as e:
            logger.error(f"Error adding message to history for user {user_id}: {e}")
            return False
    
    async def store_learning_metrics(self, user_id: str, metrics: Dict[str, Any]) -> str:
        """Store learning metrics with timestamp"""
        try:
            metrics["user_id"] = user_id
            metrics["timestamp"] = datetime.utcnow().isoformat()
            
            result = await self.metrics_collection.insert_one(metrics)
            
            # Update the user's learning history
            simplified_metrics = {
                "timestamp": metrics["timestamp"],
                "knowledge": metrics.get("knowledge_gain", 0),
                "engagement": metrics.get("engagement_level", 0),
                "performance": metrics.get("performance_score", 0)
            }
            
            await self.users_collection.update_one(
                {"user_id": user_id},
                {"$push": {"learning_history": simplified_metrics}}
            )
            
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing metrics for user {user_id}: {e}")
            return None
    
    async def get_learning_metrics(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get learning metrics history for a user"""
        try:
            cursor = self.metrics_collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            
            metrics = []
            async for document in cursor:
                document["_id"] = str(document["_id"])
                metrics.append(document)
                
            return metrics
        except Exception as e:
            logger.error(f"Error retrieving metrics for user {user_id}: {e}")
            return []
    
    async def store_feedback(self, user_id: str, message_id: str, feedback: str) -> bool:
        """Store user feedback for a specific message"""
        try:
            feedback_doc = {
                "user_id": user_id,
                "message_id": message_id,
                "feedback": feedback,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.feedback_collection.insert_one(feedback_doc)
            
            # Update the corresponding message in chat history with feedback
            await self.users_collection.update_one(
                {
                    "user_id": user_id, 
                    "chat_history.timestamp": message_id
                },
                {"$set": {"chat_history.$.feedback": feedback}}
            )
            
            return True
        except Exception as e:
            logger.error(f"Error storing feedback for user {user_id}, message {message_id}: {e}")
            return False
            
    async def get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """Get detailed learning progress for a user"""
        try:
            # Get user data
            user = await self.get_user(user_id)
            if not user:
                return None
                
            # Get recent metrics
            metrics = await self.get_learning_metrics(user_id, 10)
            
            # Extract topic mastery from metrics and user data
            topic_mastery = {}
            if user.get("recent_topics"):
                for topic in user["recent_topics"]:
                    # Start with a default value
                    topic_mastery[topic] = 0.3
            
            # Update with actual mastery values from metrics if available
            for metric in metrics:
                if "topic_mastery" in metric:
                    for topic, value in metric["topic_mastery"].items():
                        topic_mastery[topic] = value
            
            # Calculate aggregate metrics
            learning_speed = sum(m.get("knowledge_gain", 0) for m in metrics) / len(metrics) if metrics else 0
            interaction_quality = sum(m.get("interaction_quality", 0) for m in metrics) / len(metrics) if metrics else 0
            context_utilization = sum(m.get("context_utilization", 0) for m in metrics) / len(metrics) if metrics else 0
            
            return {
                "user_id": user_id,
                "knowledge_level": user.get("knowledge_level", 0.5),
                "topicMastery": topic_mastery,
                "learningSpeed": learning_speed,
                "interactionQuality": interaction_quality,
                "contextUtilization": context_utilization,
                "topics_in_progress": topic_mastery
            }
            
        except Exception as e:
            logger.error(f"Error retrieving learning progress for user {user_id}: {e}")
            return None
