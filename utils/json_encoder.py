import json
from bson import ObjectId
from datetime import datetime
from typing import Any, Dict

class CustomJSONEncoder:
    """Custom JSON encoder that handles MongoDB ObjectId and datetime objects"""
    
    @staticmethod
    def encode(obj: Any) -> Any:
        """Recursively encode an object to JSON-serializable types"""
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: CustomJSONEncoder.encode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [CustomJSONEncoder.encode(item) for item in obj]
        else:
            return obj
            
    @staticmethod
    def custom_default(obj: Any) -> Any:
        """Default serialization for JSONEncoder"""
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def get_json_encoder():
    """Returns a JSONEncoder class with custom encoding"""
    class Encoder(json.JSONEncoder):
        def default(self, obj):
            return CustomJSONEncoder.custom_default(obj)
    return Encoder
