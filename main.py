# main.py

import uuid
from support.transcript_processor import TranscriptProcessor
from support.rag_chain import YouTubeChatbotRAG


class YouTubeChatbot:
    """
    Main application class for YouTube video chatbot.
    Combines transcript processing and RAG chain with memory.
    """
    
    def __init__(self):
        self.transcript_processor = TranscriptProcessor()
        self.rag_chatbot = YouTubeChatbotRAG()
        self.current_video_id = None
    
    def load_video(self, video_id: str, youtube_url: str) -> dict:
        """
        Load and process a YouTube video.
        
        Args:
            video_id: YouTube video ID (11 characters)
            youtube_url: Full YouTube URL
            
        Returns:
            Dictionary with status and metadata
        """
        try:
            print(f"Loading video: {video_id}...")
            print(f"URL: {youtube_url}")
            
            # Process video transcript (with Whisper fallback)
            chunks, language, source = self.transcript_processor.process_video(
                video_id, youtube_url
            )
            
            print(f"✓ Processed {len(chunks)} chunks")
            print(f"✓ Source: {source}")
            print(f"✓ Language: {language}")
            
            # Setup vector store
            self.rag_chatbot.setup_vector_store(chunks)
            
            # Build RAG chain
            self.rag_chatbot.build_chain()
            
            self.current_video_id = video_id
            
            return {
                "status": "success",
                "video_id": video_id,
                "num_chunks": len(chunks),
                "language": language,
                "source": source,
                "message": f"Video loaded successfully using {source}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to load video"
            }
    
    def ask(self, question: str, session_id: str = None) -> dict:
        """
        Ask a question about the loaded video.
        
        Args:
            question: User's question
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.current_video_id is None:
            return {
                "status": "error",
                "error": "No video loaded",
                "message": "Please load a video first using load_video()"
            }
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            answer = self.rag_chatbot.chat(question, session_id)
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "session_id": session_id,
                "video_id": self.current_video_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to get answer"
            }
    
    def clear_session(self, session_id: str) -> dict:
        """
        Clear chat history for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Status dictionary
        """
        try:
            self.rag_chatbot.clear_session(session_id)
            return {
                "status": "success",
                "message": f"Session {session_id} cleared"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_active_sessions(self) -> dict:
        """Get list of active session IDs."""
        try:
            sessions = self.rag_chatbot.get_active_sessions()
            return {
                "status": "success",
                "sessions": sessions,
                "count": len(sessions)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = YouTubeChatbot()
    
    # Load a video
    video_id = "hmtuvNfytjM"  # Example video ID
    yt_url = "https://www.youtube.com/watch?v=hmtuvNfytjM&t=14s"
    result = chatbot.load_video(video_id, yt_url)
    print(result)
    
    # Create a session
    session_id = str(uuid.uuid4())
    print(f"\nSession ID: {session_id}\n")
    
    # Ask questions with memory
    # questions = [
    #     "What is GPT5?",
    #     "What are its main improvements?",
    #     "How does it compare to GPT4?",
    #     "What did you just tell me about GPT5?",  # Tests memory
    # ]
    
    while True:
        q = input("Enter your question (or 'exit' to quit): ")
        if q.lower() == 'exit':
            break
        print(f"Q: {q}")
        response = chatbot.ask(q, session_id)
        if response["status"] == "success":
            print(f"A: {response['answer']}\n")
        else:
            print(f"Error: {response['message']}\n")
    
    # Check active sessions
    print("\nActive sessions:")
    print(chatbot.get_active_sessions())
    
    # Clear session
    print("\nClearing session...")
    print(chatbot.clear_session(session_id))