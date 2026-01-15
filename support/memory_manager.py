# memory_manager.py

from langchain_core.chat_history import InMemoryChatMessageHistory
from config import MEMORY_WINDOW_SIZE


class SessionHistoryManager:
    """
    Manages chat history for multiple sessions with a sliding window approach.
    Each session maintains only the most recent K messages.
    """
    
    def __init__(self, window_size: int = MEMORY_WINDOW_SIZE):
        self.store = {}
        self.window_size = window_size
    
    def get_trimmed_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Get or create trimmed chat history for a session.
        
        Args:
            session_id: Unique identifier for the chat session
            
        Returns:
            InMemoryChatMessageHistory with trimmed messages
        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        
        full_history = self.store[session_id]
        
        # Manually trim to window size (keep only last K*2 messages - K exchanges)
        all_messages = full_history.messages
        
        # Keep only the most recent messages based on window size
        # window_size represents number of conversation turns (user + assistant pairs)
        max_messages = self.window_size * 2  # user + assistant per turn
        
        if len(all_messages) > max_messages:
            # Keep only the most recent messages
            trimmed_messages = all_messages[-max_messages:]
            
            # Create new history with trimmed messages
            new_history = InMemoryChatMessageHistory(messages=trimmed_messages)
            self.store[session_id] = new_history
            return new_history
        
        return full_history
    
    def clear_session(self, session_id: str) -> None:
        """Clear history for a specific session."""
        if session_id in self.store:
            del self.store[session_id]
    
    def get_all_sessions(self) -> list:
        """Get list of all active session IDs."""
        return list(self.store.keys())