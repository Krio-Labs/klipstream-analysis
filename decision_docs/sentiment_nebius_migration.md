# Migration to Nebius Sentiment Analysis

## Current Status

The migration to Nebius-hosted LLMs for sentiment analysis has been fully completed. The legacy BERT-based implementation has been completely replaced, and the Nebius implementation is now the primary method for analyzing emotions and sentiment in the KlipStream Analysis pipeline.

Since the initial migration, several enhancements have been made:

1. **Enhanced Audio Feature Integration**: Now includes speech rate, absolute loudness, and relative loudness (z-score) directly in the prompt
2. **Additional Fallback Model**: Added a smaller Llama model (Llama-3.2-3B-Instruct) as a second fallback option
3. **Optimized Performance**: Improved caching and concurrency control for faster processing
4. **Refined Highlight Score Calculation**: Updated weighting of different factors for better highlight detection

## Background

The original sentiment analysis module (`sentiment.py`) used BERT-based models for emotion detection and RoBERTa for sentiment analysis. The new module (`sentiment_nebius.py`) uses Nebius-hosted LLMs for more accurate sentiment analysis with a multi-level fallback system:

1. Primary model: `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`
2. First fallback: `google/gemma-2-9b-it`
3. Second fallback: `meta-llama/Llama-3.2-3B-Instruct`

## Changes Made During Migration

### 1. Updated Import Statements

The following files were updated to use the new module:

- `analysis_pipeline/processor.py`: Updated import statement to use `sentiment_nebius.py` instead of `sentiment.py`
- `analysis_pipeline/audio/__init__.py`: Removed the legacy module import and export
- `main.py`: Updated import statements and function calls to use the new module

### 2. Audio Metrics Integration

The new module calculates and passes audio metrics to the LLM for more informed sentiment analysis:

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

These metrics are also saved to the output CSV file and used by the `analysis.py` module for highlight detection.

### 3. Redundant Calculations Elimination

The `analysis.py` module was updated to use the pre-calculated audio metrics from the sentiment file when available, eliminating redundant calculations and improving performance.

### 4. Multi-level Fallback System

A robust fallback system was implemented to handle API failures:

```python
# Try fallback model if primary model fails
if model_key == DEFAULT_MODEL:
    logger.info(f"Trying fallback model {FALLBACK_MODEL}...")
    try:
        fallback_result = analyze_sentiment_sync(
            text, client, FALLBACK_MODEL, timeout,
            speech_rate=speech_rate,
            abs_intensity=abs_intensity,
            rel_intensity=rel_intensity
        )
        return fallback_result
    except Exception as fallback_error:
        logger.error(f"Fallback model also failed: {str(fallback_error)}")

        # Try small Llama model as a second fallback
        logger.info(f"Trying small Llama model as second fallback...")
        try:
            small_model_result = analyze_sentiment_sync(
                text, client, "llama_small", timeout,
                speech_rate=speech_rate,
                abs_intensity=abs_intensity,
                rel_intensity=rel_intensity
            )
            return small_model_result
        except Exception as small_model_error:
            logger.error(f"Small Llama model also failed: {str(small_model_error)}")
```

### 5. Highlight Score Calculation

The highlight score calculation was refined to better combine LLM output with audio features:

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

### 6. Testing

Comprehensive tests were run to confirm that the entire pipeline works end-to-end with the new sentiment analysis module:

- `test_nebius_sentiment.py`: Tests the new sentiment analysis module directly
- `test_analysis_with_nebius.py`: Tests the integration with the analysis module
- End-to-end pipeline tests with various VODs

## Benefits

1. **Improved Accuracy**: The Nebius-hosted LLMs provide more accurate sentiment analysis than the BERT-based models, especially for gaming streams where excitement with swearing was often misclassified as negative sentiment.

2. **Better Context Understanding**: The LLM now has access to audio metrics (speech rate, loudness) that provide crucial context about how something was said, not just what was said.

3. **Improved Performance**: Eliminating redundant calculations improves performance and reduces memory usage.

4. **Consistent Results**: Using the same metrics in both the sentiment analysis and highlight detection phases ensures consistent results.

5. **Enhanced Reliability**: The multi-level fallback system ensures that the pipeline can continue even if the primary model fails.

## Integration with the Pipeline

The Nebius implementation is now fully integrated into the main pipeline. The pipeline uses a multi-level fallback approach:

1. First tries the primary Llama model
2. Falls back to the Gemma model if the primary model fails
3. Falls back to the smaller Llama model as a last resort

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

## Command Line Usage

```bash
python -m analysis_pipeline.audio.sentiment_nebius --video-id 1234567890 --model llama --concurrency 5 --timeout 60
```

## Future Improvements

1. **Persistent Caching**: Implement a persistent cache to avoid reprocessing segments across different runs
2. **Model Fine-tuning**: Consider fine-tuning the models specifically for gaming content
3. **Additional Models**: Evaluate other models like Claude or GPT for comparison
4. **Multimodal Analysis**: Incorporate video frames for more comprehensive highlight detection
5. **Distributed Processing**: Implement distributed processing for very large datasets
6. **Real-time Processing**: Add support for real-time processing of live streams

## Conclusion

The migration to Nebius-hosted LLMs for sentiment analysis has been a significant improvement to the KlipStream Analysis pipeline. The new implementation provides more accurate sentiment analysis, better context understanding, improved performance, and enhanced reliability. The multi-level fallback system ensures that the pipeline can continue even if the primary model fails, making it a robust solution for production use.
