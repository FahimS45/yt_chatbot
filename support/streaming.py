# steaming.py

import json
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage


class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Synchronous callback handler for streaming LLM responses.
    Uses asyncio.Queue with thread-safe operations.
    """
    
    def __init__(self, loop):
        self.queue = asyncio.Queue()
        self.loop = loop
        self.done = False
    
    def _put_nowait(self, item):
        """Thread-safe queue put."""
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> None:
        """Called when chat model starts."""
        self._put_nowait({"type": "start", "content": ""})
    
    def on_llm_start(self, *args, **kwargs):
        """Called when LLM starts generating."""
        self._put_nowait({"type": "start", "content": ""})
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Called when a new token is generated."""
        self._put_nowait({"type": "token", "content": token})
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when LLM finishes generating."""
        self._put_nowait({"type": "end", "content": ""})
        self.done = True
    
    def on_llm_error(self, error: Exception, **kwargs):
        """Called when an error occurs."""
        self._put_nowait({
            "type": "error", 
            "content": str(error)
        })
        self.done = True


async def stream_chat_response(
    chatbot,
    question: str,
    session_id: str
) -> AsyncGenerator[str, None]:
    """
    Stream chat responses as Server-Sent Events.
    
    Args:
        chatbot: YouTubeChatbot instance
        question: User's question
        session_id: Session ID for conversation context
        
    Yields:
        SSE formatted strings
    """
    
    # Check if video is loaded first
    if chatbot.current_video_id is None:
        error_data = {
            "type": "error",
            "content": "No video loaded. Please load a video first."
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        return
    
    # Get current event loop for thread-safe queue operations
    loop = asyncio.get_event_loop()
    
    # Create streaming callback handler
    callback = StreamingCallbackHandler(loop)
    
    # Start the chat in background task
    async def run_chat():
        try:
            # Run chat_stream in executor (synchronous LangChain call)
            result = await loop.run_in_executor(
                None,
                chatbot.rag_chatbot.chat_stream,
                question,
                session_id,
                callback
            )
            
            # Send metadata after completion
            callback._put_nowait({
                "type": "metadata",
                "content": {
                    "session_id": session_id,
                    "video_id": chatbot.current_video_id,
                    "status": "success"
                }
            })
            
        except Exception as e:
            callback._put_nowait({
                "type": "error",
                "content": str(e)
            })
        finally:
            if not callback.done:
                callback._put_nowait({"type": "end", "content": ""})
                callback.done = True
    
    # Start chat task
    task = asyncio.create_task(run_chat())
    
    # Stream tokens from queue
    try:
        while not callback.done or not callback.queue.empty():
            try:
                # Wait for next token with timeout
                data = await asyncio.wait_for(
                    callback.queue.get(),
                    timeout=60.0
                )
                
                # Format as SSE
                yield f"data: {json.dumps(data)}\n\n"
                
            except asyncio.TimeoutError:
                # Send keepalive
                yield ": keepalive\n\n"
                
    except Exception as e:
        error_data = {"type": "error", "content": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
    
    finally:
        # Ensure task is complete
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def format_sse(data: Dict[str, Any], event: str = None) -> str:
    """
    Format data as Server-Sent Event.
    
    Args:
        data: Dictionary to send
        event: Optional event name
        
    Returns:
        SSE formatted string
    """
    msg = f"data: {json.dumps(data)}\n"
    if event:
        msg = f"event: {event}\n" + msg
    return msg + "\n"


async def stream_video_loading(
    chatbot,
    video_id: str,
    youtube_url: str
) -> AsyncGenerator[str, None]:
    """
    Stream video loading progress as SSE.
    
    Args:
        chatbot: YouTubeChatbot instance
        video_id: YouTube video ID
        youtube_url: Full YouTube URL
        
    Yields:
        SSE formatted progress updates
    """
    
    try:
        # Send start event
        yield format_sse({
            "stage": "starting",
            "message": "Starting video processing...",
            "progress": 0
        })
        
        # Check for subtitles
        yield format_sse({
            "stage": "checking_subtitles",
            "message": "Checking for YouTube subtitles...",
            "progress": 10
        })
        
        await asyncio.sleep(0.1)  # Small delay for UX
        
        # Process video (this will use appropriate method)
        # We'll wrap the synchronous call
        loop = asyncio.get_event_loop()
        
        # Send transcription start
        yield format_sse({
            "stage": "transcribing",
            "message": "Getting transcript...",
            "progress": 30
        })
        
        result = await loop.run_in_executor(
            None,
            chatbot.load_video,
            video_id,
            youtube_url
        )
        
        if result["status"] == "error":
            yield format_sse({
                "stage": "error",
                "message": result.get("error", "Failed to load video"),
                "progress": 0
            })
            return
        
        # Send translation update if needed
        if result.get("language") != "en":
            yield format_sse({
                "stage": "translating",
                "message": f"Translating from {result['language']} to English...",
                "progress": 60
            })
        
        # Send chunking update
        yield format_sse({
            "stage": "processing",
            "message": "Creating document chunks...",
            "progress": 80
        })
        
        # Send completion
        yield format_sse({
            "stage": "complete",
            "message": f"Video loaded successfully using {result['source']}",
            "progress": 100,
            "data": result
        })
        
    except Exception as e:
        yield format_sse({
            "stage": "error",
            "message": str(e),
            "progress": 0
        })