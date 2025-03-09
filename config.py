from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Mioo - AI Tutor"
    APP_VERSION: str = "1.0.0"
    
    # OpenAI Config
    OPENAI_API_KEY: str = ""
    DEFAULT_MODEL: str = "gpt-4"
    MAX_TOKENS: int = 2000
    
    # MongoDB Config  
    MONGODB_URL: str = "mongodb://localhost:27017"
    DB_NAME: str = "ai_tutor_db"
    
    # RL Config
    PPO_CONFIG = {
        'batch_size': 32,
        'epochs': 10,
        'clip_epsilon': 0.2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'learning_rate': 0.001
    }
    
    # Rate Limiting
    RATE_LIMIT_MINUTE: int = 60
    RATE_LIMIT_HOUR: int = 1000
    
    # Model Temperature Settings
    TEMPERATURE_SETTINGS = {
        'creative': 0.8,
        'balanced': 0.7,
        'precise': 0.5
    }
    
    class Config:
        env_file = ".env"

settings = Settings()
