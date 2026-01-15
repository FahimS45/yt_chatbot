# api.py

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import uuid

from models import (
    LoadVideoRequest, LoadVideoResponse,                                             
    AskQuestionRequest, AskQuestionResponse,
    ClearSessionRequest, ClearSessionResponse,
    SessionsResponse, HealthResponse
)
from main import YouTubeChatbot
from utils import extract_video_id
from streaming import stream_chat_response, stream_video_loading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Video Chatbot API",
    description="RAG-based chatbot for YouTube video transcripts with conversation memory",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot instance (singleton)
chatbot = YouTubeChatbot()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YouTube Video Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is running",
        video_loaded=chatbot.current_video_id is not None
    )


@app.post("/load-video/stream", tags=["Video", "Streaming"])
async def load_video_stream(request: LoadVideoRequest):
    """
    Load a YouTube video with SSE progress updates.
    
    Returns Server-Sent Events with progress updates:
    - checking_subtitles
    - transcribing
    - translating (if needed)
    - processing
    - complete
    """
    try:
        video_id = extract_video_id(request.url)
        logger.info(f"Streaming video load for ID: {video_id}")
        
        return StreamingResponse(
            stream_video_loading(chatbot, video_id, request.url),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    except ValueError as e:
        logger.error(f"Invalid video URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in streaming video load: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/ask/stream", tags=["Chat", "Streaming"])
async def ask_question_stream(request: AskQuestionRequest):
    """
    Ask a question with streaming SSE response.
    
    Returns Server-Sent Events with:
    - start: Marks beginning of response
    - token: Each token as it's generated
    - end: Marks completion
    - metadata: Session info
    - error: If something goes wrong
    
    Event format:
    {
        "type": "token|start|end|metadata|error",
        "content": "token text or metadata"
    }
    """
    try:
        logger.info(f"Streaming question for session: {request.session_id}")
        
        return StreamingResponse(
            stream_chat_response(chatbot, request.question, request.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"Error in streaming ask: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/load-video", response_model=LoadVideoResponse, tags=["Video"])
async def load_video(request: LoadVideoRequest):
    """
    Load a YouTube video for Q&A.
    
    Extracts video ID from URL, attempts to fetch subtitles from YouTube.
    If subtitles are unavailable, uses Whisper to transcribe the audio.
    Then translates (if needed) and prepares the RAG system.
    """
    try:
        # Extract video ID
        video_id = extract_video_id(request.url)
        logger.info(f"Loading video with ID: {video_id}")
        
        # Load video using chatbot (pass both video_id and full URL)
        result = chatbot.load_video(video_id, request.url)
        
        if result["status"] == "error":
            logger.error(f"Failed to load video: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to load video")
            )
        
        logger.info(f"Successfully loaded video: {video_id}")
        return LoadVideoResponse(**result)
        
    except ValueError as e:
        logger.error(f"Invalid video URL/ID: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error loading video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/ask", response_model=AskQuestionResponse, tags=["Chat"])
async def ask_question(request: AskQuestionRequest):
    """
    Ask a question about the loaded video.
    
    Requires session_id from frontend to maintain conversation context.
    Frontend should generate and persist the session_id (e.g., in localStorage).
    """
    try:
        logger.info(f"Processing question for session: {request.session_id}")
        
        # Get answer from chatbot
        result = chatbot.ask(request.question, request.session_id)
        
        if result["status"] == "error":
            error_msg = result.get("error", "Failed to get answer")
            logger.error(f"Error answering question: {error_msg}")
            
            # Return appropriate status code
            if "No video loaded" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg
                )
        
        logger.info(f"Successfully answered question for session: {request.session_id}")
        return AskQuestionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error answering question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/clear-session", response_model=ClearSessionResponse, tags=["Chat"])
async def clear_session(request: ClearSessionRequest):
    """
    Clear conversation history for a specific session.
    """
    try:
        logger.info(f"Clearing session: {request.session_id}")
        
        result = chatbot.clear_session(request.session_id)
        
        if result["status"] == "error":
            logger.error(f"Failed to clear session: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to clear session")
            )
        
        logger.info(f"Successfully cleared session: {request.session_id}")
        return ClearSessionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error clearing session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/sessions", response_model=SessionsResponse, tags=["Chat"])
async def get_sessions():
    """
    Get list of all active session IDs.
    """
    try:
        logger.info("Fetching active sessions")
        
        result = chatbot.get_active_sessions()
        
        if result["status"] == "error":
            logger.error(f"Failed to get sessions: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to get sessions")
            )
        
        logger.info(f"Found {result['count']} active sessions")
        return SessionsResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/sessions", tags=["Chat"])
async def clear_all_sessions():
    """
    Clear all active sessions (admin endpoint).
    """
    try:
        logger.info("Clearing all sessions")
        
        sessions = chatbot.get_active_sessions()
        if sessions["status"] == "success":
            for session_id in sessions["sessions"]:
                chatbot.clear_session(session_id)
        
        logger.info("Successfully cleared all sessions")
        return {
            "status": "success",
            "message": "All sessions cleared",
            "count": sessions.get("count", 0)
        }
        
    except Exception as e:
        logger.error(f"Unexpected error clearing all sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.detail,
            "message": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "Internal server error",
            "message": str(exc)
        }
    )


# Run with: uvicorn api:app --reload 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)