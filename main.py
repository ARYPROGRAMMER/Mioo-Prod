import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import openai
import asyncio
from datetime import datetime, timedelta
import json
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from db.mongodb_client import MongoDB
from llm.llm_client import MultiModelOrchestrator
from rl.ppo_agent import PPOAgent
from api.learning_system import LearningSystem
from utils.json_encoder import CustomJSONEncoder
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mioo AI Tutor",
    description="An adaptive learning platform powered by RL and LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models for API
class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    teaching_strategy: Dict[str, str]
    metrics: Dict[str, float]
    
class FeedbackRequest(BaseModel):
    user_id: str
    message_id: str
    feedback: str

class UserUpdateRequest(BaseModel):
    interests: Optional[List[str]] = None
    knowledge_level: Optional[float] = None
    engagement: Optional[float] = None
    
# Router for user endpoints
user_router = APIRouter(prefix="/user", tags=["users"])

# Database dependency
async def get_db():
    db = MongoDB()
    try:
        yield db
    finally:
        pass  # No need to close Motor client

# LLM client dependency
def get_llm_client():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    client = openai.OpenAI(api_key=openai_api_key)
    llm = MultiModelOrchestrator(client)
    return llm

# Get or create learning system
async def get_learning_system():
    db = MongoDB()
    llm = get_llm_client()
    rl_agent = PPOAgent()
    system = LearningSystem(db, llm, rl_agent)
    return system

# Global variables
learning_system = None

# Initialize learning system on startup
@app.on_event("startup")
async def startup_event():
    global learning_system
    try:
        db = MongoDB()
        llm = get_llm_client()
        rl_agent = PPOAgent()
        learning_system = LearningSystem(db, llm, rl_agent)
        logger.info("Learning system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing learning system: {e}")
        raise

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests"""
    global learning_system
    
    if not learning_system:
        learning_system = await get_learning_system()
        
    logger.info(f"Processing message from user {request.user_id}")
    
    try:
        response = await learning_system.process_message(request.user_id, request.message)
        
        # Format response for frontend
        return ChatResponse(
            response=response["response"],
            teaching_strategy=response["teaching_strategy"],
            metrics={
                "knowledge_gain": response["metrics"]["knowledge_gain"],
                "engagement_level": response["metrics"]["engagement_level"],
                "performance_score": response["metrics"]["performance_score"],
                "strategy_effectiveness": response["metrics"]["strategy_effectiveness"],
                "interaction_quality": response["metrics"]["interaction_quality"],
            }
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Get user endpoint
@user_router.get("/{user_id}")
async def get_user_endpoint(user_id: str, db: MongoDB = Depends(get_db)):
    """Get user state"""
    try:
        user = await db.get_user(user_id)
        if not user:
            user = await db.create_user(user_id)
        
        # Use the CustomJSONEncoder directly on the user object
        serialized_user = CustomJSONEncoder.encode(user)
        return serialized_user
    except Exception as e:
        logger.error(f"Error getting user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Update user endpoint
@user_router.put("/{user_id}")
async def update_user_endpoint(
    user_id: str, 
    update: UserUpdateRequest, 
    db: MongoDB = Depends(get_db)
):
    """Update user data"""
    try:
        user = await db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        update_data = update.dict(exclude_none=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No data to update")
        
        success = await db.update_user(user_id, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update user")
        
        return {"message": "User updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoint
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, db: MongoDB = Depends(get_db)):
    """Submit user feedback for a specific message"""
    logger.info(f"Received feedback: {request.feedback} for message: {request.message_id} from user: {request.user_id}")
    try:
        # First store the feedback
        feedback_stored = await db.store_feedback(
            request.user_id,
            request.message_id,
            request.feedback
        )
        
        if not feedback_stored:
            logger.error(f"Failed to store feedback in database")
            raise HTTPException(status_code=500, detail="Failed to store feedback")
        
        # Then process it through the learning system
        try:
            system = await get_learning_system()
            result = await system.process_feedback(
                request.user_id,
                request.message_id,
                request.feedback
            )
            logger.info(f"Feedback processed successfully")
            return result
        except Exception as e:
            # Still return success even if RL processing fails
            # We already stored the feedback in DB
            logger.error(f"Error in RL feedback processing: {e}", exc_info=True)
            return {"status": "partial", "feedback_stored": True, "feedback_processed": False}
            
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process feedback: {str(e)}"
        )

# Learning progress endpoint
@app.get("/learning-progress/{user_id}")
async def learning_progress_endpoint(user_id: str, db: MongoDB = Depends(get_db)):
    """Get learning progress for a user"""
    try:
        progress = await db.get_learning_progress(user_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Learning progress not found")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting learning progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include user routes
app.include_router(user_router)

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)