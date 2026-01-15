# whisper_transcriber.py

from faster_whisper import WhisperModel, BatchedInferencePipeline
import subprocess
import os
import tempfile
from typing import Tuple
from config import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_BATCH_SIZE, WHISPER_BEAM_SIZE, TEMP_AUDIO_DIR
)


class WhisperTranscriber:
    """Handles audio transcription using Faster Whisper."""
    
    def __init__(self):
        self.model = None
        self.batched_model = None
        
        # Create temp audio directory
        os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    
    def _load_model(self):
        """Lazy load Whisper model."""
        if self.model is None:
            print(f"Loading Whisper model: {WHISPER_MODEL_SIZE}...")
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            self.batched_model = BatchedInferencePipeline(model=self.model)
            print("Whisper model loaded successfully!")
    
    def download_audio(self, youtube_url: str) -> str:
        """
        Download audio from YouTube video using yt-dlp.
        
        Args:
            youtube_url: Full YouTube URL
            
        Returns:
            Path to downloaded audio file
        """
        # Create unique temporary file
        audio_file = os.path.join(
            TEMP_AUDIO_DIR,
            f"yt_audio_{os.urandom(8).hex()}.wav"
        )
        
        try:
            print(f"Downloading audio from: {youtube_url}")
            
            # Use yt-dlp to download best audio
            cmd = [
                "yt-dlp",
                "-f", "bestaudio",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", audio_file,
                youtube_url
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            print(f"Audio downloaded: {audio_file}")
            return audio_file
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download audio: {e.stderr}")
        except Exception as e:
            # Clean up partial download
            if os.path.exists(audio_file):
                os.remove(audio_file)
            raise RuntimeError(f"Audio download error: {str(e)}")
    
    def transcribe_audio(self, audio_path: str) -> Tuple[str, str]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (transcript_text, detected_language)
        """
        self._load_model()
        
        try:
            print(f"Transcribing audio: {audio_path}")
            
            segments, info = self.batched_model.transcribe(
                audio_path,
                batch_size=WHISPER_BATCH_SIZE,
                beam_size=WHISPER_BEAM_SIZE,
                task="transcribe"  # Always transcribe in original language
            )
            
            # Collect all segments
            full_text = ""
            for seg in segments:
                full_text += seg.text + " "
            
            transcript = full_text.strip()
            language = info.language
            
            print(f"Transcription complete. Detected language: {language}")
            print(f"Transcript length: {len(transcript)} characters")
            
            return transcript, language
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
        finally:
            # Clean up audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Cleaned up audio file: {audio_path}")
                except Exception as e:
                    print(f"Warning: Could not delete audio file: {e}")
    
    def process_youtube_url(self, youtube_url: str) -> Tuple[str, str]:
        """
        Complete pipeline: download and transcribe YouTube video.
        
        Args:
            youtube_url: Full YouTube URL
            
        Returns:
            Tuple of (transcript_text, detected_language)
        """
        # Download audio
        audio_path = self.download_audio(youtube_url)
        
        # Transcribe
        transcript, language = self.transcribe_audio(audio_path)
        
        return transcript, language
    
    def cleanup(self):
        """Clean up temporary audio directory."""
        try:
            if os.path.exists(TEMP_AUDIO_DIR):
                for file in os.listdir(TEMP_AUDIO_DIR):
                    file_path = os.path.join(TEMP_AUDIO_DIR, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print("Cleaned up temporary audio files")
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")