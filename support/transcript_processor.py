# transcript_processor.py

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple, Optional
from config import TRANSLATION_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from whisper_transcriber import WhisperTranscriber


class TranscriptProcessor:
    """Handles YouTube transcript fetching (with Whisper fallback), translation, and chunking."""
    
    def __init__(self):
        self.ytt_api = YouTubeTranscriptApi()
        self.whisper = WhisperTranscriber()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.translation_tokenizer = None
        self.translation_model = None
    
    def _load_translation_model(self):
        """Lazy load translation model only when needed."""
        if self.translation_model is None:
            self.translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
            self.translation_model = AutoModelForCausalLM.from_pretrained(
                TRANSLATION_MODEL,
                device_map="auto",
                dtype=torch.bfloat16
            )

    def get_transcript(self, video_id: str, yt_url: str) -> Tuple[str, str, str]:
        """
        Fetch transcript from YouTube video.
        Priority:
        1) Auto-generated YouTube captions
        2) Whisper transcription fallback
        """
        try:
            transcript_list = self.ytt_api.list(video_id)

            # Try auto-generated captions
            for transcript in transcript_list:
                if transcript.is_generated:
                    chunks = transcript.fetch()
                    full_text = " ".join(
                        chunk.text.replace("\n", " ") for chunk in chunks
                    )
                    language = transcript.language_code  # cleaner than split
                    source = "youtube_auto"
                    return full_text, language, source

        except TranscriptsDisabled:
            # Subtitles disabled → fall back to Whisper
            pass

        except Exception as e:
            # Any other YouTube transcript failure → fallback
            pass

        # Whisper fallback (single source of truth)
        try:
            full_text, language = self.whisper.process_youtube_url(yt_url)
            source = "whisper"
            return full_text, language, source

        except Exception as e:
            raise ValueError(f"Failed to fetch transcript via Whisper: {str(e)}")

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text to English using local model.
        
        Args:
            text: Text to translate
            source_lang: Source language code (ISO 639-1, e.g., 'es', 'fr', 'en')
            
        Returns:
            Translated text in English
        """
        # Check if already English (handle both 'en' and 'English')
        if source_lang.lower() in ['en', 'english']:
            return text
        
        self._load_translation_model()
        
        system_prompt = (
            "You are a language translator. Your only task is to translate "
            "the given text into the target language with the highest possible "
            "accuracy according to the user's request. Do not add explanations, "
            "comments, or any additional text. Output only the translated text."
        )
        
        user_message = f"Translate this {text} into English language"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        tokenized_chat = self.translation_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        outputs = self.translation_model.generate(
            tokenized_chat.to(self.translation_model.device),
            max_new_tokens=2048,
            do_sample=True,
            top_k=20,
            top_p=0.6,
            temperature=0.7,
            repetition_penalty=1.05,
        )
        
        response = self.translation_tokenizer.decode(
            outputs[0][tokenized_chat.shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def create_chunks(self, text: str) -> List:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Full transcript text
            
        Returns:
            List of Document chunks
        """
        return self.splitter.create_documents([text])
    
    def process_video(self, video_id: str, youtube_url: str) -> Tuple[List, str, str]:
        """
        Complete processing pipeline for a YouTube video.
        
        Args:
            video_id: YouTube video ID (11 characters)
            youtube_url: Full YouTube URL
            
        Returns:
            Tuple of (chunks, language, source)
            source: 'youtube_manual', 'youtube_auto', or 'whisper'
        """
        # Get transcript (with Whisper fallback)
        transcript, language, source = self.get_transcript(video_id, youtube_url)
        
        print(f"Transcript source: {source}")
        print(f"Original language: {language}")
        print(f"Transcript length: {len(transcript)} characters")
        
        # Translate if not English
        if language.lower() not in ['en', 'english']:
            print(f"Translating from {language} to English...")
            transcript = self.translate_to_english(transcript, language)
            print("Translation complete!")
        else:
            print("Already in English, skipping translation")
        
        # Create chunks
        print("Creating document chunks...")
        chunks = self.create_chunks(transcript)
        print(f"Created {len(chunks)} chunks")
        
        return chunks, language, source