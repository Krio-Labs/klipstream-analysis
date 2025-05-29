# Audio Highlight Score Analysis

## Overview

This document provides a detailed analysis of how audio highlight scores are calculated in the Klipstream analysis pipeline. The highlight score is a critical component used to identify noteworthy moments in Twitch streams for video editors.

## Highlight Score Calculation Process

The audio highlight score calculation is a multi-step process that combines several factors:

1. **Sliding Window Generation**: The system first creates sliding windows of audio/transcript data
2. **Audio Feature Extraction**: Audio intensity features are extracted from the waveform
3. **Emotion & Sentiment Analysis**: Text is analyzed for emotions and sentiment using Nebius API
4. **Highlight Score Calculation**: Multiple factors are combined with weighted importance
5. **Integration with Chat**: Audio highlight scores are combined with chat analysis for enhanced scores

Let's examine each component in detail.

## 1. Sliding Window Generation

Before calculating highlight scores, the system processes the transcript using a sliding window approach:

- **Window Size**: 60 seconds (configurable)
- **Overlap**: 30 seconds (configurable)
- **Implementation**: `sliding_window_generator.py`

This approach creates segments that are more effective for highlight detection than using individual paragraphs. Each window contains:
- Start and end times
- Combined text from overlapping paragraphs
- Word count and speech rate calculations

The sliding window generator is implemented in `raw_pipeline/sliding_window_generator.py` and creates a CSV file with segments that are used for subsequent analysis.

## 2. Audio Feature Extraction

Two key audio intensity metrics are calculated:

### 2.1 Absolute Audio Intensity

Absolute audio intensity measures the raw loudness of the audio:

```python
# From audio/analysis.py
# Combine audio features
audio_intensity = (rms_norm * 0.7 + centroid_norm * 0.3)  # Weighted combination
```

This combines:
- **RMS Energy** (70%): Root Mean Square energy, representing loudness
- **Spectral Centroid** (30%): Represents the "brightness" or frequency distribution of the audio

Both features are normalized to a 0-1 range before combination.

### 2.2 Relative Audio Intensity (Z-Score)

The relative audio intensity is calculated using a rolling window approach to identify moments that are louder than the surrounding context:

```python
# From audio/sentiment_nebius.py - extract_audio_intensity function
# Calculate relative intensity (z-score) if rolling stats provided
relative_intensity = 0.0
if rolling_stats is not None:
    # Add this value to rolling stats
    mid_time = (start_time + end_time) / 2
    rolling_stats.add_value(mid_time, raw_energy)

    # Get z-score
    z_score = rolling_stats.get_z_score(raw_energy)

    # Cap extreme values
    z_score = max(-3.0, min(3.0, z_score))

    # Convert to 0-1 range for easier integration
    # Z-score of -3 maps to 0.0, Z-score of +3 maps to 1.0
    relative_intensity = (z_score + 3.0) / 6.0
```

This approach:
1. Maintains a rolling window of audio energy values (10-minute window by default)
2. Calculates the z-score of the current segment's energy relative to this window
3. Caps extreme values to prevent outliers from dominating
4. Normalizes to a 0-1 range for easier integration

The `RollingAudioStats` class handles this calculation efficiently with optimizations for performance.

## 3. Speech Rate Factor

Speech rate is calculated as words per second and normalized for use in highlight detection:

```python
# Normalize speech rate (assuming normal speech is 2-3 words per second)
# Cap at 6 wps to avoid extreme values
speech_rate_norm = min(speech_rate / 6.0, 1.0)
```

This approach:
- Normalizes speech rate against a maximum of 6 words per second
- Higher speech rates (faster talking) get higher scores
- Lower speech rates (slower talking) get lower scores

## 4. Emotion and Sentiment Analysis

The system now uses the Nebius API with LLM models for emotion and sentiment analysis:

### 4.1 Emotion Analysis

The system uses the Nebius API to classify text into five main emotion categories:
- Excitement
- Funny
- Happiness
- Anger
- Sadness

The system prompt instructs the model to:
1. Calibrate the baseline so most segments score low (<0.3) on emotion categories unless there's clear evidence
2. Only award high emotion scores (>0.7) when the language strongly signals that emotion
3. Decide the highlight score based on whether the segment has genuine "viral potential"

### 4.2 Emotion Intensity

The emotion intensity represents the average strength of emotions expressed:

```python
# Get average emotion score (excluding neutral)
emotion_score = (
    sentiment.get('excitement', 0.0) +
    sentiment.get('funny', 0.0) +
    sentiment.get('happiness', 0.0) +
    sentiment.get('anger', 0.0) +
    sentiment.get('sadness', 0.0)
) / 5.0
```

### 4.3 Sentiment Analysis

Sentiment analysis produces a score from -1 (negative) to 1 (positive), which is included in the Nebius API response.

## 5. Highlight Score Calculation

The current implementation uses a blend of model-generated highlight scores and calculated scores based on audio features:

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

In the analysis.py file, there's additional processing for the highlight scores:

```python
# For Nebius-generated data
data['weighted_highlight_score'] = (
    data['highlight_score'] * 0.95 +    # Nebius highlight score (95%)
    audio_intensity * 0.05              # Current audio intensity (5%)
)

# For legacy data without Nebius features
data['weighted_highlight_score'] = (
    data['highlight_score'] * 0.5 +      # Base highlight score
    data['emotion_intensity'] * 0.2 +    # Emotion contribution
    abs(data['sentiment_score']) * 0.1 + # Sentiment intensity contribution
    audio_intensity * 0.2                # Audio intensity contribution
)
```

## 6. Peak Detection and Highlight Selection

After calculating highlight scores, the system identifies peak moments using signal processing techniques:

```python
# For Nebius data, use more sensitive parameters
peaks, properties = find_peaks(
    data['weighted_highlight_score'],
    distance=15,        # Reduced minimum distance between peaks
    prominence=0.25,    # Lower prominence threshold
    height=0.35         # Lower height threshold
)

# For legacy data, use the original parameters
peaks, properties = find_peaks(
    data['weighted_highlight_score'],
    distance=20,        # Minimum samples between peaks
    prominence=0.3,     # Minimum prominence of peaks
    height=0.4          # Minimum height of peaks
)
```

The system then selects the top 10 highlights based on the weighted highlight score.

## 7. Integration with Chat Analysis

The final step integrates the audio highlight score with chat analysis to create an enhanced highlight score:

```python
# From integration.py - calculate_enhanced_highlight_score function
# Base combined score (weighted average)
combined_score = base_highlight * 0.6 + chat_highlight * 0.4
```

This combines:
- Audio highlight score (60% weight)
- Chat highlight score (40% weight)

The weights can be dynamically adjusted based on the characteristics of the stream:

```python
# Determine optimal weights for this stream
audio_weight, chat_weight = determine_optimal_weights(audio_df, chat_df)
```

The `determine_optimal_weights` function analyzes chat activity and emotion intensity to adjust the relative importance of audio vs. chat:

```python
# Adjust weights based on relative strengths
# High chat activity → more weight to chat
# High audio emotion → more weight to audio
base_audio_weight = 0.6
base_chat_weight = 0.4

# Adjust for chat activity (more chat → more chat weight)
chat_activity_factor = min(1.0, max(0.0, (chat_activity - 10) / 100))

# Adjust for relative emotion intensity
if audio_emotion_intensity > 0 and chat_emotion_intensity > 0:
    emotion_ratio = audio_emotion_intensity / chat_emotion_intensity
    emotion_factor = min(1.0, max(0.0, (emotion_ratio - 0.5) / 2))
else:
    emotion_factor = 0.5

# Calculate final weights
audio_weight = base_audio_weight + (chat_activity_factor * 0.2) - (emotion_factor * 0.2)
chat_weight = base_chat_weight - (chat_activity_factor * 0.2) + (emotion_factor * 0.2)
```

## Emotional Coherence Analysis

The integration module also analyzes the coherence between streamer emotions (audio) and audience emotions (chat):

```python
# Apply relationship-specific adjustments
if relationship == "synchronized":
    # Perfect alignment between streamer and audience emotions
    # This is likely a very genuine moment
    coherence_bonus = 0.15
elif relationship == "positive_aligned":
    # Both positive but different emotions
    # Still a good highlight candidate
    coherence_bonus = 0.1
elif relationship == "audience_positive":
    # Audience positive despite streamer negative
    # Could be entertaining (audience enjoying streamer's frustration)
    coherence_bonus = 0.08
elif relationship == "audience_negative":
    # Audience negative despite streamer positive
    # Could be controversial or divisive content
    coherence_bonus = 0.05
else:
    coherence_bonus = 0.0
```

This adds a bonus to the highlight score based on the relationship between streamer and audience emotions.

## Conclusion

The audio highlight score calculation has evolved to use a more sophisticated approach that leverages LLM models through the Nebius API. The current implementation:

1. Uses sliding windows to create meaningful segments
2. Extracts audio features including absolute and relative loudness
3. Analyzes emotions and sentiment using the Nebius API
4. Calculates highlight scores using a blend of model predictions and audio features
5. Identifies peak moments using signal processing techniques
6. Integrates with chat analysis to create enhanced highlight scores
7. Analyzes emotional coherence between streamer and audience

This comprehensive approach creates a robust system for identifying noteworthy moments in Twitch streams that considers both the streamer's content and the audience's reactions.
