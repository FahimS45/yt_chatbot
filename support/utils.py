# utils.py

import re
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from various URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - Direct VIDEO_ID (11 characters)
    
    Args:
        url: YouTube URL or video ID
        
    Returns:
        11-character video ID
        
    Raises:
        ValueError: If video ID cannot be extracted or is invalid
    """
    # If already a valid video ID (11 characters, alphanumeric + _ -)
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url
    
    # Parse URL
    parsed_url = urlparse(url)
    
    # Handle youtube.com/watch?v=VIDEO_ID format
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'm.youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                video_id = query_params['v'][0]
                if len(video_id) == 11:
                    return video_id
        
        # Handle youtube.com/embed/VIDEO_ID format
        elif parsed_url.path.startswith('/embed/'):
            video_id = parsed_url.path.split('/')[2]
            if len(video_id) >= 11:
                return video_id[:11]
        
        # Handle youtube.com/v/VIDEO_ID format
        elif parsed_url.path.startswith('/v/'):
            video_id = parsed_url.path.split('/')[2]
            if len(video_id) >= 11:
                return video_id[:11]
    
    # Handle youtu.be/VIDEO_ID format
    elif parsed_url.hostname in ['youtu.be', 'www.youtu.be']:
        video_id = parsed_url.path.lstrip('/')
        if len(video_id) >= 11:
            return video_id[:11]
    
    # Try regex as last resort
    regex_match = re.search(r'(?:v=|/)([A-Za-z0-9_-]{11})', url)
    if regex_match:
        return regex_match.group(1)
    
    raise ValueError(
        "Invalid YouTube URL or video ID. "
        "Please provide a valid YouTube URL or 11-character video ID."
    )


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format (UUID4).
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if valid, False otherwise
    """
    uuid_pattern = re.compile(
        r'^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(session_id))