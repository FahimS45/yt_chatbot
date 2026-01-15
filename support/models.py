# models.py

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from utils import extract_video_id


class LoadVideoRequest(BaseModel):
    """Request model for loading a YouTube video."""
    url: str = Field(
        ..., 
        description="YouTube video URL or 11-character video ID",
        example="https://www.youtube.com/watch?v=hmtuvNfytjM"
    )
    
    @validator('url')
    def validate_url(cls, v):
        """Validate and normalize YouTube URL."""
        from utils import extract_video_id

        try:
            video_id = extract_video_id(v)  # get only the 11-char ID
            # Normalize the URL
            normalized_url = f"https://www.youtube.com/watch?v={video_id}"
            return normalized_url
        except ValueError as e:
            raise ValueError(str(e))


class LoadVideoResponse(BaseModel):
    """Response model for video loading."""
    status: str = Field(..., description="success or error")
    video_id: Optional[str] = Field(None, description="Extracted video ID")
    num_chunks: Optional[int] = Field(None, description="Number of text chunks created")
    language: Optional[str] = Field(None, description="Original transcript language")
    source: Optional[str] = Field(None, description="Transcript source: youtube_manual, youtube_auto, or whisper")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if status is error")


class AskQuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(
        ..., 
        description="Question about the video",
        min_length=1,
        max_length=1000,
        example="What is GPT5?"
    )
    session_id: str = Field(
        ...,
        description="Session ID for conversation continuity (UUID4 format)",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
    
    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session ID is a proper UUID4."""
        from utils import validate_session_id
        if not validate_session_id(v):
            raise ValueError("Invalid session_id format. Must be a valid UUID4.")
        return v


class AskQuestionResponse(BaseModel):
    """Response model for question answering."""
    status: str = Field(..., description="success or error")
    question: Optional[str] = Field(None, description="Original question")
    answer: Optional[str] = Field(None, description="Answer from the chatbot")
    session_id: Optional[str] = Field(None, description="Session ID used")
    video_id: Optional[str] = Field(None, description="Current video ID")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if status is error")


class ClearSessionRequest(BaseModel):
    """Request model for clearing a session."""
    session_id: str = Field(
        ...,
        description="Session ID to clear",
        example="550e8400-e29b-41d4-a716-446655440000"
    )


class ClearSessionResponse(BaseModel):
    """Response model for session clearing."""
    status: str = Field(..., description="success or error")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if status is error")


class SessionsResponse(BaseModel):
    """Response model for active sessions list."""
    status: str = Field(..., description="success or error")
    sessions: Optional[List[str]] = Field(None, description="List of active session IDs")
    count: Optional[int] = Field(None, description="Number of active sessions")
    error: Optional[str] = Field(None, description="Error message if status is error")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="healthy or unhealthy")
    message: str = Field(..., description="Health status message")
    video_loaded: bool = Field(..., description="Whether a video is currently loaded")