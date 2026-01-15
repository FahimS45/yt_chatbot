# rag_chain.py

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from typing import List

from config import (
    GITHUB_TOKEN, GITHUB_ENDPOINT, MODEL_NAME,
    EMBEDDING_MODEL, RETRIEVER_K, LLM_TEMPERATURE
)
from memory_manager import SessionHistoryManager


class YouTubeChatbotRAG:
    """RAG-based chatbot for YouTube video transcripts with memory."""
    
    def __init__(self):
        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
            model_name=MODEL_NAME,
            openai_api_key=GITHUB_TOKEN,
            openai_api_base=GITHUB_ENDPOINT,
            temperature=LLM_TEMPERATURE,
            streaming=True,
        )
        
        # Initialize embeddings
        self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize memory manager
        self.history_manager = SessionHistoryManager()
        
        # Initialize vector store (will be set per video)
        self.vector_store = None
        self.retriever = None
        self.conversational_chain = None
    
    def setup_vector_store(self, chunks: List) -> None:
        """Create FAISS vector store from document chunks."""
        self.vector_store = FAISS.from_documents(chunks, self.embedder)
        # Create retriever immediately
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K}
        )

    def _create_prompt(self):
        """Create QA prompt with memory and context."""
        
        # Single prompt that handles everything
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful, friendly, and funny AI assistant answering questions about a YouTube video.\n\n"
             "INSTRUCTIONS:\n"
             "- Answer based ONLY on the video context provided below\n"
             "- Answer directly without repeating the question\n"
             "- Use conversation history to understand context and references\n"
             "- If asked about previous questions, refer to the chat history\n"
             "- If information is not in the video context, say so clearly without offering anything else.\n\n"
             "- IMPORTANT: You must not answer or offer anything outside the scope or context of the video. If you do so, you will be terminated.\n\n"
             "- Maintain politeness\n\n"
             "Video Context:\n{context}\n"
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ])
        
        return qa_prompt
    
    def build_chain(self) -> None:
        """Build the complete RAG chain with memory."""
        
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call setup_vector_store first.")
        
        # Create prompt
        qa_prompt = self._create_prompt()
        
        # Create document chain (combines docs with prompt)
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Create retrieval chain (retrieves docs + runs document chain)
        # This ensures context is ALWAYS retrieved for every question
        retrieval_chain = create_retrieval_chain(
            self.retriever,
            document_chain
        )
        
        # Wrap with message history
        self.conversational_chain = RunnableWithMessageHistory(
            retrieval_chain,
            self.history_manager.get_trimmed_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def get_conversational_chain(self):
        """Get the conversational chain for streaming."""
        if self.conversational_chain is None:
            raise ValueError("Chain not built. Call build_chain first.")
        return self.conversational_chain
    
    def chat(self, question: str, session_id: str) -> str:
        """Get answer for a question with session memory (non-streaming)."""
        if self.conversational_chain is None:
            raise ValueError("Chain not built. Call build_chain first.")
        
        response = self.conversational_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        return response["answer"]
    
    def chat_stream(self, question: str, session_id: str, callback):
        """Get streaming answer for a question with session memory."""
        if self.conversational_chain is None:
            raise ValueError("Chain not built. Call build_chain first.")
        
        # Invoke with streaming callback
        response = self.conversational_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [callback]
            }
        )
        
        return response["answer"]
    
    def clear_session(self, session_id: str) -> None:
        """Clear chat history for a session."""
        self.history_manager.clear_session(session_id)
    
    def get_active_sessions(self) -> list:
        """Get list of active session IDs."""
        return self.history_manager.get_all_sessions()