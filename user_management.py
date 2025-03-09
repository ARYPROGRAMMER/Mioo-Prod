from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import uuid

router = APIRouter()

class UserProfile(BaseModel):
    user_id: str
    created_at: datetime
    preferences: Dict = {}
    active: bool = True

class UserCreate(BaseModel):
    preferences: Optional[Dict] = {}

class UserStats(BaseModel):
    total_interactions: int
    knowledge_level: float
    engagement_rate: float
    topics_explored: List[str]

async def generate_user_id() -> str:
    return f"user-{uuid.uuid4().hex[:8]}"

@router.post("/users/", response_model=UserProfile)
async def create_user(user_data: UserCreate, db=None):
    user_id = await generate_user_id()
    user = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "preferences": user_data.preferences,
        "active": True
    }
    
    # Initialize user state in database
    await db.users.insert_one(user)
    return UserProfile(**user)

@router.get("/users/{user_id}", response_model=UserProfile)
async def get_user(user_id: str, db=None):
    user = await db.users.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserProfile(**user)

@router.get("/users/{user_id}/stats", response_model=UserStats)
async def get_user_stats(user_id: str, db=None):
    user_state = await db.user_states.find_one({"user_id": user_id})
    if not user_state:
        raise HTTPException(status_code=404, detail="User stats not found")
    
    return UserStats(
        total_interactions=len(user_state.get("chat_history", [])),
        knowledge_level=user_state.get("knowledge_level", 0.5),
        engagement_rate=user_state.get("engagement", 0.5),
        topics_explored=user_state.get("recent_topics", [])
    )
