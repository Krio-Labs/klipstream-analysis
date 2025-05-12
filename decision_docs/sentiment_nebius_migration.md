# Migration to Nebius Sentiment Analysis

## Overview

This document describes the migration from the legacy `sentiment.py` module to the new `sentiment_nebius.py` module for audio sentiment analysis in the Klipstream Analysis pipeline.

## Background

The original sentiment analysis module (`sentiment.py`) used BERT-based models for emotion detection and RoBERTa for sentiment analysis. The new module (`sentiment_nebius.py`) uses Nebius-hosted LLMs (meta-llama/Meta-Llama-3.1-8B-Instruct as primary and google/gemma-2-9b-it as backup) for more accurate sentiment analysis.

## Changes Made

### 1. Updated Import Statements

The following files were updated to use the new module:

- `analysis_pipeline/processor.py`: Updated import statement to use `sentiment_nebius.py` instead of `sentiment.py`
- `analysis_pipeline/audio/__init__.py`: Removed the legacy module import and export
- `main.py`: Updated import statements and function calls to use the new module

### 2. Audio Metrics Integration

The new module calculates and passes audio metrics (speech rate, absolute intensity, relative intensity) to the LLM for more informed sentiment analysis. These metrics are also saved to the output CSV file and used by the `analysis.py` module for highlight detection.

### 3. Redundant Calculations Elimination

The `analysis.py` module was updated to use the pre-calculated audio metrics from the sentiment file when available, eliminating redundant calculations and improving performance.

### 4. Testing

Tests were run to confirm that the entire pipeline works end-to-end with the new sentiment analysis module:

- `test_nebius_sentiment.py`: Tests the new sentiment analysis module directly
- `test_analysis_with_nebius.py`: Tests the integration with the analysis module

## Benefits

1. **Improved Accuracy**: The Nebius-hosted LLMs provide more accurate sentiment analysis than the BERT-based models, especially for gaming streams where excitement with swearing was often misclassified as negative sentiment.

2. **Better Context**: The LLM now has access to audio metrics (speech rate, loudness) that provide crucial context about how something was said, not just what was said.

3. **Improved Performance**: Eliminating redundant calculations improves performance and reduces memory usage.

4. **Consistent Results**: Using the same metrics in both the sentiment analysis and highlight detection phases ensures consistent results.

## Usage

The new implementation can be used as follows:

```python
from analysis_pipeline.audio.sentiment_nebius import analyze_audio_sentiment

# Basic usage
analyze_audio_sentiment(
    video_id="1234567890",
    model_key="llama",  # Use "gemma" for the backup model
    max_concurrent=5,   # Optional: control concurrency
    timeout=60          # Optional: set request timeout
)
```

## Command Line Usage

```bash
python -m analysis_pipeline.audio.sentiment_nebius --video-id 1234567890 --model llama
```

## Future Improvements

1. **Fine-tune the Prompt**: Consider fine-tuning the prompt based on the results to further improve the accuracy of the sentiment analysis.

2. **Performance Optimization**: Monitor the performance impact of including audio metrics in the prompt and optimize if necessary.

3. **Error Handling**: Improve error handling for cases where the Nebius API is unavailable or returns errors.
