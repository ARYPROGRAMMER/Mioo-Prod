from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    RATE_LIMIT_MINUTE: int = 60
    RATE_LIMIT_HOUR: int = 1000
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # MongoDB Settings
    MONGODB_URL: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")  # Changed to match .env
    MONGODB_DB: str = os.getenv("MONGODB_DB", "mioo_db")  # Added to match .env
    
    # Model Settings
    DEFAULT_MODEL: str = "gpt-4"
    FALLBACK_MODEL: str = "gpt-3.5-turbo"
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7
    
    # Training Settings
    TRAINING_BATCH_SIZE: int = 32
    TRAINING_EPOCHS: int = 10
    INITIAL_EPSILON: float = 1.0
    EPSILON_DECAY: float = 0.997
    MIN_EPSILON: float = 0.01
    LEARNING_RATE: float = 0.001
    DISCOUNT_FACTOR: float = 0.99
    
    # Memory Settings
    MAX_MEMORY_SIZE: int = 10000
    MIN_MEMORY_FOR_TRAINING: int = 1000
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow extra fields from .env
        extra = "allow"

settings = Settings()
