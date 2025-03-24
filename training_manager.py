import asyncio
from typing import Dict, Optional
import json
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from trainers.persona_trainer import PersonaTrainer

class TrainingManager:
    def __init__(self):
        self.active_trainings: Dict[str, Dict] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        
    async def start_training(self, training_id: str, episodes: int, persona_data: str):
        """Start a new training session."""
        if training_id in self.active_trainings:
            return False
            
        self.active_trainings[training_id] = {
            "status": "running",
            "progress": 0,
            "start_time": datetime.utcnow().isoformat(),
            "episodes": episodes,
            "current_episode": 0,
            "last_reward": 0.0,
            "avg_reward": 0.0
        }
        
        # Start training in background task
        asyncio.create_task(self._run_training(training_id, episodes, persona_data))
        return True
        
    async def _run_training(self, training_id: str, episodes: int, persona_data: str):
        """Run the training process."""
        try:
            trainer = PersonaTrainer(persona_data)
            
            def progress_callback(episode: int, reward: float, avg_reward: float):
                self.active_trainings[training_id].update({
                    "current_episode": episode,
                    "progress": (episode / episodes) * 100,
                    "last_reward": reward,
                    "avg_reward": avg_reward
                })
                # Notify frontend if websocket connection exists
                if training_id in self.websocket_connections:
                    asyncio.create_task(
                        self._send_update(training_id)
                    )
            
            # Start training with progress callback
            trainer.train(
                episodes=episodes,
                progress_callback=progress_callback
            )
            
            self.active_trainings[training_id]["status"] = "completed"
            
        except Exception as e:
            self.active_trainings[training_id]["status"] = "failed"
            self.active_trainings[training_id]["error"] = str(e)
            
        finally:
            if training_id in self.websocket_connections:
                await self._send_update(training_id)
    
    async def register_websocket(self, training_id: str, websocket: WebSocket):
        """Register a WebSocket connection for training updates."""
        try:
            await websocket.accept()
            self.websocket_connections[training_id] = websocket
            
            # Send initial status if available
            if training_id in self.active_trainings:
                await self._send_update(training_id)
        except Exception as e:
            print(f"Error registering websocket: {e}")
            if training_id in self.websocket_connections:
                self.websocket_connections.pop(training_id)
    
    async def _send_update(self, training_id: str):
        """Send training status update through WebSocket."""
        if training_id in self.websocket_connections:
            try:
                ws = self.websocket_connections[training_id]
                status = self.active_trainings.get(training_id, {})
                await ws.send_json(status)
            except WebSocketDisconnect:
                self.websocket_connections.pop(training_id, None)
            except Exception as e:
                print(f"Error sending update: {e}")
                self.websocket_connections.pop(training_id, None)
    
    def get_training_status(self, training_id: str) -> Optional[Dict]:
        """Get current status of a training session."""
        return self.active_trainings.get(training_id)
