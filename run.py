import uvicorn
import asyncio
import nest_asyncio
from main import app

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            log_level="info",
            loop="asyncio"
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        raise
