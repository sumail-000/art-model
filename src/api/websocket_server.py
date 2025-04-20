import asyncio
import json
import logging
import base64
import io
import numpy as np
from PIL import Image
import torch
import websockets
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pixelmind-x-api")

class WebSocketServer:
    """
    WebSocket server for bidirectional chat-art streaming
    """
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8765,
                 model_manager=None,
                 max_connections: int = 100):
        """
        Initialize WebSocket server
        
        Args:
            host: Host address to bind
            port: Port to listen on
            model_manager: Manager for PixelMind-X models
            max_connections: Maximum number of simultaneous connections
        """
        self.host = host
        self.port = port
        self.model_manager = model_manager
        self.max_connections = max_connections
        
        # Active connections
        self.active_connections = set()
        
        # User sessions (user_id -> session_data)
        self.user_sessions = {}
        
        # Message handlers
        self.message_handlers = {
            "generate_image": self.handle_generate_image,
            "update_image": self.handle_update_image,
            "feedback": self.handle_feedback,
            "chat_message": self.handle_chat_message,
            "sketch_update": self.handle_sketch_update,
            "style_preference": self.handle_style_preference
        }
        
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.connection_handler,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB max message size (for images)
            ping_interval=30,
            ping_timeout=10
        ):
            await asyncio.Future()  # Keep server running
            
    async def connection_handler(self, websocket, path):
        """Handle a WebSocket connection"""
        # Check if we have capacity
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Server at capacity")
            return
            
        # Add to active connections
        self.active_connections.add(websocket)
        
        user_id = None
        session_id = None
        
        try:
            # Perform authentication/session setup
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            if "user_id" in auth_data:
                user_id = auth_data["user_id"]
                
                # Create new session or load existing
                if user_id not in self.user_sessions:
                    session_id = str(uuid.uuid4())
                    self.user_sessions[user_id] = {
                        "session_id": session_id,
                        "start_time": datetime.now().isoformat(),
                        "last_activity": datetime.now().isoformat(),
                        "interactions": [],
                        "websocket": websocket
                    }
                else:
                    # Update session with new websocket
                    session_id = self.user_sessions[user_id]["session_id"]
                    self.user_sessions[user_id]["websocket"] = websocket
                    self.user_sessions[user_id]["last_activity"] = datetime.now().isoformat()
                
                # Send session confirmation
                await websocket.send(json.dumps({
                    "type": "session_start",
                    "user_id": user_id,
                    "session_id": session_id,
                    "status": "success"
                }))
                
                logger.info(f"User {user_id} connected with session {session_id}")
                
                # Main message loop
                async for message in websocket:
                    try:
                        # Parse message JSON
                        data = json.loads(message)
                        
                        # Update last activity
                        self.user_sessions[user_id]["last_activity"] = datetime.now().isoformat()
                        
                        # Process message based on type
                        if "type" in data and data["type"] in self.message_handlers:
                            await self.message_handlers[data["type"]](websocket, data, user_id, session_id)
                        else:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": f"Unknown message type: {data.get('type')}"
                            }))
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON message"
                        }))
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}", exc_info=True)
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Error processing message: {str(e)}"
                        }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Authentication failed: user_id required"
                }))
                
        except websockets.exceptions.ConnectionClosedError:
            logger.info(f"Connection closed for user {user_id}")
        except Exception as e:
            logger.error(f"Error in connection handler: {str(e)}", exc_info=True)
        finally:
            # Clean up
            self.active_connections.remove(websocket)
            if user_id in self.user_sessions and self.user_sessions[user_id]["websocket"] == websocket:
                # Don't remove the session, just mark that the websocket is no longer active
                self.user_sessions[user_id]["websocket"] = None
                logger.info(f"User {user_id} disconnected")
    
    async def handle_generate_image(self, websocket, data, user_id, session_id):
        """
        Handle image generation request
        
        Args:
            websocket: WebSocket connection
            data: Message data containing generation parameters
            user_id: User ID
            session_id: Session ID
        """
        # Extract generation parameters
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        style_id = data.get("style_id")
        seed = data.get("seed", np.random.randint(0, 2**32 - 1))
        width = data.get("width", 512)
        height = data.get("height", 512)
        
        # Optional sketch input
        sketch_data = data.get("sketch")
        sketch = None
        if sketch_data:
            try:
                sketch_bytes = base64.b64decode(sketch_data.split(",")[1])
                sketch = Image.open(io.BytesIO(sketch_bytes)).convert("RGB")
            except Exception as e:
                logger.error(f"Error decoding sketch: {str(e)}")
                
        # Record the interaction
        interaction_id = str(uuid.uuid4())
        self.user_sessions[user_id]["interactions"].append({
            "id": interaction_id,
            "type": "generate_image",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "style_id": style_id,
                "seed": seed,
                "width": width,
                "height": height,
                "has_sketch": sketch is not None
            }
        })
        
        # Acknowledge request immediately
        await websocket.send(json.dumps({
            "type": "generation_started",
            "interaction_id": interaction_id
        }))
        
        try:
            # First, we need to send the scaffolding image to show progress
            if self.model_manager:
                # Generate low-resolution scaffold
                scaffold_result = await self.model_manager.generate_scaffold(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    style_id=style_id,
                    seed=seed,
                    sketch=sketch,
                    user_id=user_id
                )
                
                # Convert scaffold image to base64
                buffered = io.BytesIO()
                scaffold_result["images"][0].save(buffered, format="PNG")
                scaffold_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Send scaffold preview
                await websocket.send(json.dumps({
                    "type": "generation_progress",
                    "interaction_id": interaction_id,
                    "stage": "scaffold",
                    "progress": 0.3,
                    "image": f"data:image/png;base64,{scaffold_b64}"
                }))
                
                # Generate mid-resolution
                mid_res_result = await self.model_manager.refine_to_mid_res(
                    scaffold_result["latents"],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    style_id=style_id
                )
                
                # Convert mid-res image to base64
                buffered = io.BytesIO()
                mid_res_result["images"][0].save(buffered, format="PNG")
                mid_res_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Send mid-res preview
                await websocket.send(json.dumps({
                    "type": "generation_progress",
                    "interaction_id": interaction_id,
                    "stage": "mid_res",
                    "progress": 0.6,
                    "image": f"data:image/png;base64,{mid_res_b64}"
                }))
                
                # Generate final high-resolution image
                final_result = await self.model_manager.refine_to_high_res(
                    mid_res_result["latents"],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    style_id=style_id
                )
                
                # Convert final image to base64
                buffered = io.BytesIO()
                final_result["images"][0].save(buffered, format="PNG")
                final_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Send final result
                await websocket.send(json.dumps({
                    "type": "generation_complete",
                    "interaction_id": interaction_id,
                    "image": f"data:image/png;base64,{final_b64}",
                    "parameters": {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "style_id": style_id,
                        "seed": seed
                    },
                    "metadata": {
                        "image_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat()
                    }
                }))
            else:
                # No model manager, send error
                await websocket.send(json.dumps({
                    "type": "error",
                    "interaction_id": interaction_id,
                    "message": "Model manager not available"
                }))
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "interaction_id": interaction_id,
                "message": f"Error generating image: {str(e)}"
            }))
    
    async def handle_update_image(self, websocket, data, user_id, session_id):
        """
        Handle image update request (refinement)
        
        Args:
            websocket: WebSocket connection
            data: Message data containing update parameters
            user_id: User ID
            session_id: Session ID
        """
        # Extract parameters
        image_id = data.get("image_id")
        prompt = data.get("prompt")
        changes = data.get("changes", [])
        
        interaction_id = str(uuid.uuid4())
        self.user_sessions[user_id]["interactions"].append({
            "id": interaction_id,
            "type": "update_image",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "image_id": image_id,
                "prompt": prompt,
                "changes": changes
            }
        })
        
        # Acknowledge request
        await websocket.send(json.dumps({
            "type": "update_started",
            "interaction_id": interaction_id
        }))
        
        try:
            if self.model_manager:
                # Apply refinements based on feedback
                refined_result = await self.model_manager.refine_image(
                    image_id=image_id,
                    prompt=prompt,
                    changes=changes,
                    user_id=user_id
                )
                
                # Convert refined image to base64
                buffered = io.BytesIO()
                refined_result["image"].save(buffered, format="PNG")
                refined_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Send refined result
                await websocket.send(json.dumps({
                    "type": "update_complete",
                    "interaction_id": interaction_id,
                    "image": f"data:image/png;base64,{refined_b64}",
                    "parameters": {
                        "image_id": refined_result["image_id"],
                        "prompt": prompt,
                        "changes": changes
                    },
                    "metadata": {
                        "timestamp": datetime.now().isoformat()
                    }
                }))
            else:
                # No model manager, send error
                await websocket.send(json.dumps({
                    "type": "error",
                    "interaction_id": interaction_id,
                    "message": "Model manager not available"
                }))
        except Exception as e:
            logger.error(f"Error updating image: {str(e)}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "interaction_id": interaction_id,
                "message": f"Error updating image: {str(e)}"
            }))
    
    async def handle_feedback(self, websocket, data, user_id, session_id):
        """
        Handle feedback on generated image
        
        Args:
            websocket: WebSocket connection
            data: Message data containing feedback
            user_id: User ID
            session_id: Session ID
        """
        # Extract feedback parameters
        image_id = data.get("image_id")
        rating = data.get("rating")
        feedback_text = data.get("feedback", "")
        feedback_categories = data.get("categories", [])
        biometric_data = data.get("biometric_data", {})
        
        interaction_id = str(uuid.uuid4())
        self.user_sessions[user_id]["interactions"].append({
            "id": interaction_id,
            "type": "feedback",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "image_id": image_id,
                "rating": rating,
                "feedback": feedback_text,
                "categories": feedback_categories,
                "has_biometric": len(biometric_data) > 0
            }
        })
        
        # Acknowledge feedback
        await websocket.send(json.dumps({
            "type": "feedback_received",
            "interaction_id": interaction_id
        }))
        
        try:
            if self.model_manager:
                # Send feedback to model manager for learning
                await self.model_manager.process_feedback(
                    image_id=image_id,
                    user_id=user_id,
                    rating=rating,
                    feedback_text=feedback_text,
                    feedback_categories=feedback_categories,
                    biometric_data=biometric_data
                )
                
                # Send confirmation
                await websocket.send(json.dumps({
                    "type": "feedback_processed",
                    "interaction_id": interaction_id,
                    "message": "Feedback incorporated successfully"
                }))
            else:
                # No model manager, just acknowledge
                await websocket.send(json.dumps({
                    "type": "feedback_processed",
                    "interaction_id": interaction_id,
                    "message": "Feedback recorded (model manager not available)"
                }))
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "interaction_id": interaction_id,
                "message": f"Error processing feedback: {str(e)}"
            }))
    
    async def handle_chat_message(self, websocket, data, user_id, session_id):
        """
        Handle chat message (for multimodal understanding)
        
        Args:
            websocket: WebSocket connection
            data: Message data containing chat content
            user_id: User ID
            session_id: Session ID
        """
        # Extract chat message
        message = data.get("message", "")
        
        interaction_id = str(uuid.uuid4())
        self.user_sessions[user_id]["interactions"].append({
            "id": interaction_id,
            "type": "chat_message",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "message": message
            }
        })
        
        # Process chat message through intent parser
        try:
            if self.model_manager:
                # Parse intent from message
                intent_result = await self.model_manager.parse_intent(
                    message=message,
                    user_id=user_id,
                    session_id=session_id
                )
                
                # Send response
                await websocket.send(json.dumps({
                    "type": "chat_response",
                    "interaction_id": interaction_id,
                    "message": intent_result.get("response", "I've received your message."),
                    "intent": intent_result.get("intent", "general_chat"),
                    "actions": intent_result.get("actions", [])
                }))
                
                # If there are suggested actions, send them
                if intent_result.get("suggested_image"):
                    # Auto-generate image from intent
                    image_generation_data = {
                        "type": "generate_image",
                        "prompt": intent_result["suggested_prompt"],
                        "negative_prompt": intent_result.get("suggested_negative", ""),
                        "style_id": intent_result.get("suggested_style")
                    }
                    
                    # Call the generation handler
                    await self.handle_generate_image(
                        websocket, image_generation_data, user_id, session_id
                    )
            else:
                # Simple echo response if no model manager
                await websocket.send(json.dumps({
                    "type": "chat_response",
                    "interaction_id": interaction_id,
                    "message": f"Received: {message}",
                    "intent": "echo"
                }))
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "interaction_id": interaction_id,
                "message": f"Error processing chat message: {str(e)}"
            }))
    
    async def handle_sketch_update(self, websocket, data, user_id, session_id):
        """
        Handle sketch updates for real-time collaboration
        
        Args:
            websocket: WebSocket connection
            data: Message data containing sketch update
            user_id: User ID
            session_id: Session ID
        """
        # Extract sketch data
        sketch_data = data.get("sketch")
        is_complete = data.get("complete", False)
        
        # For real-time updates, don't log every stroke
        if is_complete:
            interaction_id = str(uuid.uuid4())
            self.user_sessions[user_id]["interactions"].append({
                "id": interaction_id,
                "type": "sketch_update",
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "has_sketch": sketch_data is not None,
                    "complete": is_complete
                }
            })
        
        try:
            # If sketch is complete, we can generate a preview
            if is_complete and sketch_data and self.model_manager:
                # Decode sketch data
                sketch_bytes = base64.b64decode(sketch_data.split(",")[1])
                sketch = Image.open(io.BytesIO(sketch_bytes)).convert("RGB")
                
                # Generate a quick preview based on the sketch
                preview_result = await self.model_manager.generate_sketch_preview(
                    sketch=sketch,
                    user_id=user_id
                )
                
                # Convert preview to base64
                buffered = io.BytesIO()
                preview_result["image"].save(buffered, format="PNG")
                preview_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Send preview
                await websocket.send(json.dumps({
                    "type": "sketch_preview",
                    "image": f"data:image/png;base64,{preview_b64}",
                    "suggestions": preview_result.get("suggestions", [])
                }))
            else:
                # Just acknowledge sketch update
                await websocket.send(json.dumps({
                    "type": "sketch_update_received"
                }))
        except Exception as e:
            logger.error(f"Error processing sketch update: {str(e)}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Error processing sketch update: {str(e)}"
            }))
    
    async def handle_style_preference(self, websocket, data, user_id, session_id):
        """
        Handle style preference updates
        
        Args:
            websocket: WebSocket connection
            data: Message data containing style preferences
            user_id: User ID
            session_id: Session ID
        """
        # Extract style preferences
        style_id = data.get("style_id")
        liked = data.get("liked", True)
        
        interaction_id = str(uuid.uuid4())
        self.user_sessions[user_id]["interactions"].append({
            "id": interaction_id,
            "type": "style_preference",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "style_id": style_id,
                "liked": liked
            }
        })
        
        try:
            if self.model_manager:
                # Update style preferences in the model
                await self.model_manager.update_style_preference(
                    user_id=user_id,
                    style_id=style_id,
                    liked=liked
                )
                
                # Get recommended styles based on preference
                recommended_styles = await self.model_manager.get_recommended_styles(
                    user_id=user_id
                )
                
                # Send confirmation with recommendations
                await websocket.send(json.dumps({
                    "type": "style_preference_updated",
                    "interaction_id": interaction_id,
                    "recommended_styles": recommended_styles
                }))
            else:
                # Just acknowledge
                await websocket.send(json.dumps({
                    "type": "style_preference_updated",
                    "interaction_id": interaction_id,
                    "recommended_styles": []
                }))
        except Exception as e:
            logger.error(f"Error updating style preference: {str(e)}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "interaction_id": interaction_id,
                "message": f"Error updating style preference: {str(e)}"
            }))
    
    async def broadcast_message(self, message):
        """
        Broadcast a message to all connected clients
        
        Args:
            message: JSON-serializable message to broadcast
        """
        # Prepare JSON message
        json_message = json.dumps(message)
        
        # List of failed connections to clean up
        failed_connections = set()
        
        # Send to all active connections
        for websocket in self.active_connections:
            try:
                await websocket.send(json_message)
            except websockets.exceptions.ConnectionClosed:
                failed_connections.add(websocket)
                
        # Clean up failed connections
        for websocket in failed_connections:
            self.active_connections.remove(websocket)
    
    async def send_to_user(self, user_id, message):
        """
        Send a message to a specific user
        
        Args:
            user_id: User ID to send message to
            message: JSON-serializable message to send
        
        Returns:
            bool: Whether the message was sent successfully
        """
        if user_id in self.user_sessions and self.user_sessions[user_id]["websocket"]:
            try:
                await self.user_sessions[user_id]["websocket"].send(json.dumps(message))
                return True
            except websockets.exceptions.ConnectionClosed:
                # Websocket closed, mark as disconnected
                self.user_sessions[user_id]["websocket"] = None
                
        return False 