# Sentiment Analysis Implementation with Nebius-hosted LLMs

## Current Status

This implementation is now fully integrated into the KlipStream Analysis pipeline and has completely replaced the previous BERT-based models. The implementation has been enhanced with additional features since the initial deployment:

1. **Enhanced Audio Feature Integration**: Now includes speech rate, absolute loudness, and relative loudness (z-score) directly in the prompt
2. **Additional Fallback Model**: Added a smaller Llama model (Llama-3.2-3B-Instruct) as a second fallback option
3. **Optimized Performance**: Improved caching and concurrency control for faster processing
4. **Refined Highlight Score Calculation**: Updated weighting of different factors for better highlight detection

## Decision Summary

We've implemented a sentiment analysis module that uses Nebius-hosted LLMs to analyze emotions and sentiment in audio transcriptions. This approach has successfully replaced the previous implementation that used local BERT-based models.

## Motivation

The previous sentiment analysis implementation had several limitations:

1. **Misclassification of Gaming Content**: BERT-based models often misclassified excitement with swearing as negative sentiment, which is inappropriate for gaming streams where excited swearing is common and positive.
2. **Limited Emotion Detection**: The previous models had difficulty distinguishing between different positive emotions (excitement vs. happiness).
3. **Performance Limitations**: Processing large numbers of segments (800+) was time-consuming and resource-intensive.
4. **Scalability Issues**: The local models required significant GPU resources and didn't scale well.

## Implementation Details

### Architecture

The current implementation (`sentiment_nebius.py`) uses:

1. **Nebius-hosted LLMs**:
   - Primary model: `meta-llama/Meta-Llama-3.1-8B-Instruct-fast` (optimized for speed)
   - First fallback model: `google/gemma-2-9b-it`
   - Second fallback model: `meta-llama/Llama-3.2-3B-Instruct`

2. **Asynchronous Processing**:
   - Uses `asyncio` and `aiohttp` for efficient concurrent API calls
   - Implements dynamic concurrency control based on system resources
   - Processes segments in optimally-sized batches

3. **Audio Feature Integration**:
   - Calculates speech rate (words per second)
   - Measures absolute loudness (normalized audio intensity)
   - Computes relative loudness (z-score normalized against rolling window)
   - Includes these metrics directly in the prompt for better context

4. **Comprehensive Highlight Detection**:
   - Combines LLM sentiment analysis with audio features:
     - Model score (50%)
     - Speech rate (15%)
     - Relative loudness (15%)
     - Absolute loudness (10%)
     - Emotional intensity (10%)

5. **Robust Error Handling**:
   - Multi-level fallback system with two backup models
   - Comprehensive logging and diagnostics
   - Caching to avoid reprocessing identical segments

### System Prompt

The system prompt is carefully designed to calibrate the model for gaming stream content:

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

### Performance Optimizations

1. **Dynamic Concurrency**:
   - Automatically determines optimal number of concurrent requests based on:
     - Available CPU cores
     - Available memory
     - API rate limits

2. **Adaptive Batch Sizing**:
   - Small datasets (< 50 segments): 5 segments per batch
   - Medium datasets (< 200 segments): 10 segments per batch
   - Large datasets (< 500 segments): 20 segments per batch
   - Very large datasets (800+ segments): 30 segments per batch

3. **Caching**:
   - Implements in-memory caching to avoid reprocessing identical segments
   - Uses normalized text as cache keys to handle minor variations

4. **Progress Tracking**:
   - Uses `tqdm` for real-time progress visualization
   - Provides detailed logging of processing steps

## Comparison with Previous Implementation

### Accuracy

| Aspect | Previous (BERT) | Current (Llama) |
|--------|----------------|-------------|
| Excitement detection | Poor (often confused with anger) | Excellent |
| Humor detection | Moderate | Good |
| Gaming context understanding | Poor | Excellent |
| Highlight identification | Moderate | Very Good |
| Context awareness | Limited | Excellent |

### Performance

| Metric | Previous (BERT) | Current (Llama) |
|--------|----------------|-------------|
| Processing time for 800 segments | ~30 minutes | ~8-12 minutes |
| Memory usage | High | Low |
| GPU requirement | Yes | No (API-based) |
| Scalability | Limited | Excellent |
| Resilience to failures | Poor | Excellent (multi-level fallback) |

### Example Results

Testing with sample gaming stream content:

```
Text: "Oh my god! This is so amazing! I can't believe we just got a 20 bomb! Let's go!"
Speech Rate: 3.50 words per second
Absolute Loudness: 0.85
Relative Loudness: 0.90

Results:
  Sentiment: 0.90
  Excitement: 0.85
  Funny: 0.00
  Happiness: 0.90
  Anger: 0.00
  Sadness: 0.00
  Neutral: 0.10
  Highlight score: 0.85
```

```
Text: "What the hell was that? This game is so broken! The developers need to fix this garbage!"
Speech Rate: 3.20 words per second
Absolute Loudness: 0.75
Relative Loudness: 0.80

Results:
  Sentiment: -0.90
  Excitement: 0.00
  Funny: 0.10
  Happiness: 0.00
  Anger: 0.90
  Sadness: 0.40
  Neutral: 0.30
  Highlight score: 0.60
```

## Usage

The current implementation can be used as follows:

```python
from analysis_pipeline.audio.sentiment_nebius import analyze_audio_sentiment

# Basic usage
analyze_audio_sentiment(
    video_id="1234567890",
    model_key="llama",  # Use "gemma" for the first backup model, "llama_small" for the second
    max_concurrent=5,   # Optional: control concurrency
    timeout=60          # Optional: set request timeout
)
```

Or from the command line:

```bash
python -m analysis_pipeline.audio.sentiment_nebius --video-id 1234567890 --model llama --concurrency 5 --timeout 60
```

## Integration with the Pipeline

The Nebius implementation is now fully integrated into the main pipeline. The pipeline uses a multi-level fallback approach:

1. First tries the primary Llama model
2. Falls back to the Gemma model if the primary model fails
3. Falls back to the smaller Llama model as a last resort

## Future Improvements

1. **Persistent Caching**: Implement a persistent cache to avoid reprocessing segments across different runs
2. **Model Fine-tuning**: Consider fine-tuning the models specifically for gaming content
3. **Additional Models**: Evaluate other models like Claude or GPT for comparison
4. **Multimodal Analysis**: Incorporate video frames for more comprehensive highlight detection
5. **Distributed Processing**: Implement distributed processing for very large datasets
6. **Real-time Processing**: Add support for real-time processing of live streams

## Conclusion

The Nebius-hosted LLM implementation has significantly improved both the accuracy and performance of sentiment analysis for gaming streams. The asynchronous processing, dynamic resource allocation, and multi-level fallback system make it well-suited for processing large datasets efficiently and reliably.

The integration of audio features with LLM-based sentiment analysis provides a more comprehensive approach to highlight detection, resulting in better identification of moments with viral potential. This implementation has become a core component of the KlipStream Analysis pipeline and continues to be refined and improved.
