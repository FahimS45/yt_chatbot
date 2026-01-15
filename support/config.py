# config.py

import os

# API Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "Your_GitHub_Token_Here")
GITHUB_ENDPOINT = "https://models.github.ai/inference"
MODEL_NAME = "openai/gpt-4.1-nano"

# Translation Model Configuration
TRANSLATION_MODEL = "tencent/HY-MT1.5-1.8B"

# Whisper Configuration
WHISPER_MODEL_SIZE = "turbo"  # or "large-v3-turbo"
WHISPER_DEVICE = "cuda"  # or "cpu"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BATCH_SIZE = 16
WHISPER_BEAM_SIZE = 5

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Audio Processing
TEMP_AUDIO_DIR = "temp_audio"

# Text Splitter Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever Configuration
RETRIEVER_K = 5

# Memory Configuration
MEMORY_WINDOW_SIZE = 10  # Number of recent messages to keep in memory

# LLM Configuration
LLM_TEMPERATURE = 0.2