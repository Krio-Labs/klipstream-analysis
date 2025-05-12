# Nebius Sentiment Analysis Implementation

## Overview

This document describes the implementation of sentiment analysis using the Nebius API with Llama and Gemma models. The implementation replaces the previous sentiment analysis module that used local BERT-based models.

## Motivation

The previous sentiment analysis implementation had several limitations:
1. It required downloading large model files (several GB) to the local machine
2. It was computationally intensive, requiring a GPU for reasonable performance
3. The BERT-based models incorrectly labeled excitement with swearing as negative
4. It was not optimized for streaming content analysis

The new implementation addresses these issues by:
1. Using Nebius-hosted LLMs (Llama and Gemma) via API calls
2. Implementing asynchronous processing for better performance
3. Using a custom system prompt that understands gaming stream context
4. Providing automatic fallback to a secondary model if the primary model fails

## Implementation Details

### Architecture

The implementation consists of two main components:

1. **Core Module**: `analysis_pipeline/audio/sentiment_nebius.py`
   - Contains the core functionality for sentiment analysis using the Nebius API
   - Implements asynchronous processing for better performance
   - Includes audio feature extraction and highlight score calculation
   - Provides fallback mechanisms for error handling

2. **Wrapper Module**: `nebius_sentiment.py`
   - Provides a compatibility layer for the main pipeline
   - Maintains the same interface as the original `audio_sentiment.py` module
   - Handles file path resolution and environment setup

### Key Features

- **Asynchronous Processing**: Uses Python's `asyncio` and `aiohttp` for concurrent API calls
- **Automatic Concurrency Control**: Dynamically adjusts the number of concurrent requests based on system resources
- **Model Fallback**: Automatically falls back to a secondary model if the primary model fails
- **Caching**: Implements caching to avoid redundant API calls for the same text
- **Progress Tracking**: Shows a progress bar during processing
- **Comprehensive Error Handling**: Provides detailed error messages and graceful degradation

### Models

The implementation uses two models hosted on Nebius:

1. **Primary Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - A state-of-the-art instruction-tuned language model
   - Provides high-quality sentiment and emotion analysis

2. **Fallback Model**: `google/gemma-2-9b-it`
   - Used if the primary model fails or times out
   - Provides comparable quality with different characteristics

### System Prompt

The system prompt is carefully designed to:
1. Understand the context of gaming streams
2. Correctly interpret gaming-specific language and expressions
3. Calibrate the baseline for emotions to avoid over-sensitivity
4. Provide consistent scoring across different segments

```
You are a *strict* sentiment & emotion analysis engine, specialized for Twitch gaming stream transcripts.

IMPORTANT CONTEXT: In gaming streams, phrases like "bad financial decisions" and spending in-game currency (like "1,000,000 credits") are POSITIVE events that generate excitement. Swear words like "fucking" often express excitement, not anger.

When given a 60-second transcript, you must:

1. **Calibrate your baseline** so that most segments score low (<0.3) on emotion categories unless there's clear, vivid evidence.
2. Only award high emotion scores (>0.7) when the language *strongly* signals that emotion.
3. **Decide `highlight_score`** solely on whether this segment has genuine "viral potential".

**Ranges:**
- `sentiment_score`: –1.0 (very negative) to +1.0 (very positive)
- all emotions & `highlight_score`: 0.0 to 1.0
```

## Usage

### Basic Usage

```python
from nebius_sentiment import sentiment_analysis

# Run sentiment analysis for a video
result = sentiment_analysis(video_id="1234567890")
```

### Advanced Usage

```python
from analysis_pipeline.audio.sentiment_nebius import analyze_audio_sentiment

# Run sentiment analysis with custom parameters
result = analyze_audio_sentiment(
    video_id="1234567890",
    input_file="path/to/segments.csv",
    output_dir="path/to/output",
    audio_file="path/to/audio.wav",
    use_nebius=True,
    model_key="llama",  # Use "gemma" for the backup model
    max_concurrent=5,   # Control concurrency
    timeout=60          # Set request timeout
)
```

### Command Line Usage

```bash
# Using the wrapper module
python nebius_sentiment.py --video-id 1234567890

# Using the core module directly
python -m analysis_pipeline.audio.sentiment_nebius --video-id 1234567890 --model llama
```

## Integration with the Pipeline

The new implementation is designed to be a drop-in replacement for the original sentiment analysis module. The main pipeline (`main.py`) now uses the Nebius implementation by default, with a fallback to the original implementation if needed:

```python
# Use Nebius-hosted Llama model for sentiment analysis (default)
sentiment_result = nebius_sentiment_analysis(video_id=video_id)

# If Nebius model fails, fall back to Gemma model
if not sentiment_result:
    logger.warning("Nebius sentiment analysis failed, falling back to Gemma model...")
    sentiment_result = gemma_sentiment_analysis(video_id=video_id)
```

## Requirements

- Python 3.8+
- `openai` package for API access
- `aiohttp` for asynchronous requests
- `tqdm` for progress tracking
- `psutil` for system resource monitoring
- Nebius API key set in the environment variable `NEBIUS_API_KEY`

## Performance Considerations

- The implementation automatically adjusts concurrency based on available system resources
- For a typical run with ~800 segments, processing takes approximately 5-10 minutes
- API rate limits may affect performance; the implementation includes retry logic
- The implementation uses caching to avoid redundant API calls

## Future Improvements

- Implement more sophisticated caching (e.g., persistent cache)
- Add support for more models and providers
- Improve error handling and recovery mechanisms
- Implement more advanced audio feature extraction
- Add support for real-time processing
