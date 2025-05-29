# Nebius Sentiment Analysis Implementation

## Overview

This document describes the implementation of sentiment analysis using the Nebius API with Llama and Gemma models. The implementation has fully replaced the previous sentiment analysis module that used local BERT-based models.

## Current Status

The Nebius sentiment analysis implementation is now the primary method for analyzing emotions and sentiment in the KlipStream Analysis pipeline. It has been fully integrated and includes several enhancements since the initial implementation:

1. **Enhanced Audio Feature Integration**: Now includes speech rate, absolute loudness, and relative loudness (z-score) directly in the prompt
2. **Additional Fallback Model**: Added a smaller Llama model (Llama-3.2-3B-Instruct) as a second fallback option
3. **Optimized Performance**: Improved caching and concurrency control for faster processing
4. **Refined Highlight Score Calculation**: Updated weighting of different factors for better highlight detection

## Motivation

The previous sentiment analysis implementation had several limitations:
1. It required downloading large model files (several GB) to the local machine
2. It was computationally intensive, requiring a GPU for reasonable performance
3. The BERT-based models incorrectly labeled excitement with swearing as negative
4. It was not optimized for streaming content analysis

The Nebius implementation addresses these issues by:
1. Using Nebius-hosted LLMs via API calls
2. Implementing asynchronous processing for better performance
3. Using a custom system prompt that understands gaming stream context
4. Providing automatic fallback to secondary models if the primary model fails
5. Incorporating audio features directly into the analysis

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
- **Multi-level Model Fallback**: Automatically falls back to secondary models if the primary model fails
- **Caching**: Implements caching to avoid redundant API calls for the same text
- **Progress Tracking**: Shows a progress bar during processing
- **Comprehensive Error Handling**: Provides detailed error messages and graceful degradation
- **Audio Feature Integration**: Includes speech rate, absolute loudness, and relative loudness in the analysis

### Models

The implementation uses three models hosted on Nebius:

1. **Primary Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`
   - A state-of-the-art instruction-tuned language model optimized for speed
   - Provides high-quality sentiment and emotion analysis

2. **First Fallback Model**: `google/gemma-2-9b-it`
   - Used if the primary model fails or times out
   - Provides comparable quality with different characteristics

3. **Second Fallback Model**: `meta-llama/Llama-3.2-3B-Instruct`
   - A smaller, faster model used as a last resort
   - Provides acceptable quality with faster response times

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

### Audio Feature Integration

The current implementation includes audio features directly in the prompt:

```python
# Create user prompt with audio metrics if available
if speech_rate is not None and abs_intensity is not None and rel_intensity is not None:
    # Format the metrics to 2 decimal places for readability
    user_prompt = f"""Analyze the following transcript segment from a Twitch stream and provide scores for sentiment and emotions.

Speech Rate: {speech_rate:.2f} words per second
Absolute Loudness: {abs_intensity:.2f}
Relative Loudness: {rel_intensity:.2f}

Transcript:
{text}"""
else:
    # Use the original prompt without audio metrics
    user_prompt = f"Analyze this Twitch gaming stream transcript:\n\n{text}"
```

This provides the LLM with crucial context about how something was said, not just what was said.

### Highlight Score Calculation

The highlight score is calculated using a weighted combination of factors:

```python
# Calculate base highlight score (without model score)
base_highlight_score = (
    (speech_rate_norm * 0.15) +
    (rel_intensity * 0.15) +
    (abs_intensity * 0.10) +
    (emotion_score * 0.10)
)

# Scale the base score to account for the components' weights
# This ensures the base score still contributes meaningfully
# when combined with the model score
base_highlight_score = base_highlight_score * (1.0 / 0.5)

# If model already provided a highlight score, blend them
if 'highlight_score' in sentiment and sentiment['highlight_score'] > 0:
    model_highlight = sentiment['highlight_score']
    # Blend 50% model score, 50% calculated score
    highlight_score = (model_highlight * 0.5) + (base_highlight_score * 0.5)
else:
    # If no model score, use the base score
    highlight_score = base_highlight_score
```

The current weighting is:
- Model score (50%)
- Speech rate (15%)
- Relative loudness (15%)
- Absolute loudness (10%)
- Emotional intensity (10%)

## Usage

### Basic Usage

```python
from analysis_pipeline.audio.sentiment_nebius import analyze_audio_sentiment

# Run sentiment analysis for a video
result = analyze_audio_sentiment(video_id="1234567890")
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
# Using the core module directly
python -m analysis_pipeline.audio.sentiment_nebius --video-id 1234567890 --model llama
```

## Integration with the Pipeline

The Nebius implementation is now fully integrated into the main pipeline. The pipeline uses a multi-level fallback approach:

1. First tries the primary Llama model
2. Falls back to the Gemma model if the primary model fails
3. Falls back to the smaller Llama model as a last resort

## Requirements

- Python 3.8+
- `openai` package for API access
- `aiohttp` for asynchronous requests
- `tqdm` for progress tracking
- `psutil` for system resource monitoring
- `librosa` for audio feature extraction
- Nebius API key set in the environment variable `NEBIUS_API_KEY`

## Performance Considerations

- The implementation automatically adjusts concurrency based on available system resources
- For a typical run with ~800 segments, processing takes approximately 5-10 minutes
- API rate limits may affect performance; the implementation includes retry logic
- The implementation uses caching to avoid redundant API calls
- Audio feature extraction is optimized to minimize memory usage

## Future Improvements

1. **Persistent Caching**: Implement a persistent cache to avoid reprocessing segments across different runs
2. **Model Fine-tuning**: Consider fine-tuning the models specifically for gaming content
3. **Additional Models**: Evaluate other models like Claude or GPT for comparison
4. **Multimodal Analysis**: Incorporate video frames for more comprehensive highlight detection
5. **Distributed Processing**: Implement distributed processing for very large datasets
6. **Real-time Processing**: Add support for real-time processing of live streams
