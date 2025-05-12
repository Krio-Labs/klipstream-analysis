# Audio Highlight Score Analysis

## Overview

This document provides a detailed analysis of how audio highlight scores are calculated in the Klipstream analysis pipeline. The highlight score is a critical component used to identify noteworthy moments in Twitch streams for video editors.

## Highlight Score Calculation Process

The audio highlight score calculation is a multi-step process that combines several factors:

1. **Sliding Window Generation**: The system first creates sliding windows of audio/transcript data
2. **Audio Feature Extraction**: Audio intensity features are extracted from the waveform
3. **Emotion & Sentiment Analysis**: Text is analyzed for emotions and sentiment
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
# From audio/sentiment.py - extract_audio_intensity function
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

The `RollingAudioStats` class handles this calculation efficiently.

## 3. Speech Rate Factor

Speech rate is calculated as words per second and then normalized:

```python
# From audio/sentiment.py
# Get speech rate factor (default to 1.0 if not provided)
speech_rate_factor = 0.0
if speech_rate is not None:
    # Convert normalized speech rate to a 0-1 score
    # A normalized rate of 1.0 (average) becomes 0.5
    # Rates below average get lower scores, rates above average get higher scores
    speech_rate_factor = min(1.0, max(0.0, (speech_rate - 0.5) * 0.5 + 0.5))
```

This transformation:
- Takes the normalized speech rate (where 1.0 is the average for the stream)
- Maps it to a 0-1 range where 0.5 is average
- Higher speech rates (faster talking) get higher scores
- Lower speech rates (slower talking) get lower scores

## 4. Emotion and Sentiment Analysis

The system analyzes the text content for emotions and sentiment:

### 4.1 Emotion Analysis

The system uses a Hugging Face model to classify text into emotion categories, which are then mapped to five main emotions:
- Excitement
- Funny
- Happiness
- Anger
- Sadness

The emotion with the highest score becomes the "mapped_emotion" for the segment.

### 4.2 Emotion Intensity

The emotion intensity represents how strongly the dominant emotion is expressed:

```python
# Calculate emotion intensity (how strong the emotion is)
emotion_intensity = scores[mapped_emotion] if mapped_emotion != 'neutral' else 0.0
```

### 4.3 Sentiment Analysis

Sentiment analysis produces a score from -1 (negative) to 1 (positive). For highlight calculation, the absolute value is used to measure intensity regardless of polarity:

```python
# Calculate sentiment intensity (how strong the sentiment is)
sentiment_intensity = abs(scores['sentiment_score'])
```

## 5. Highlight Score Calculation

The final highlight score combines all these factors with weighted importance:

```python
# From audio/sentiment.py - generate_scores_with_precomputed_results function
# Combine all factors for highlight score with rebalanced weights
scores['highlight_score'] = (
    highlight_base * 0.30 +                # Base score from emotion type (30% weight)
    emotion_intensity * 0.20 +             # Emotion intensity (20% weight)
    sentiment_intensity * 0.10 +           # Sentiment intensity (10% weight)
    absolute_intensity * 0.05 +            # Absolute audio intensity (5% weight)
    relative_intensity * 0.20 +            # Relative audio intensity (20% weight)
    speech_rate_factor * 0.15              # Speech rate factor (15% weight)
)
```

### 5.1 Highlight Base Score

The base score depends on the type of emotion detected:

```python
highlight_base = 0.0
mapped_emotion = next((e for e in ['excitement', 'funny', 'happiness', 'anger', 'sadness'] if scores[e] > 0), 'neutral')

if mapped_emotion in ['excitement', 'funny', 'happiness']:
    highlight_base = 0.5  # Positive emotions get higher base score
elif mapped_emotion in ['anger', 'sadness']:
    highlight_base = 0.3  # Negative emotions get lower base score
```

This gives a higher starting point for positive emotions (excitement, funny, happiness) compared to negative emotions (anger, sadness).

### 5.2 Synergy Bonus

An additional synergy bonus is applied when multiple factors align:

```python
# Enhanced synergy bonus calculation
# Boost score for segments with high emotion and either high relative audio intensity or high speech rate
if emotion_intensity > 0.5 and (relative_intensity > 0.7 or speech_rate_factor > 0.7):
    synergy_factor = max(relative_intensity, speech_rate_factor if speech_rate is not None else 0)
    emotion_synergy = min(emotion_intensity, synergy_factor) * 0.2

    # Add extra bonus when ALL three factors align (emotion, audio, and speech rate)
    # This rewards moments where everything comes together
    if emotion_intensity > 0.6 and relative_intensity > 0.7 and speech_rate_factor > 0.7:
        emotion_synergy += 0.1  # Additional fixed bonus for multi-factor alignment

    scores['highlight_score'] += emotion_synergy
```

This rewards segments where:
- Strong emotions coincide with either loud audio or fast speech
- An extra bonus is given when all three factors align

## 6. Integration with Chat Analysis

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

The `determine_optimal_weights` function analyzes chat activity and emotion intensity to adjust the relative importance of audio vs. chat.

## Potential Issues and Improvements

Based on the code analysis, here are potential issues that might affect the audio highlight score calculation:

1. **Relative Intensity Calculation**: The z-score approach is sensitive to the distribution of audio energy in the rolling window. If there are long quiet periods followed by normal speech, normal speech might get artificially high scores.

2. **Weight Balance**: The weights assigned to different factors (30% base emotion, 20% emotion intensity, etc.) might need adjustment based on empirical testing.

3. **Synergy Bonus Thresholds**: The thresholds for synergy bonuses (emotion > 0.5, relative_intensity > 0.7) might need tuning for different types of streams.

4. **Speech Rate Normalization**: The current approach normalizes speech rate against the stream average, which might not work well for streams with unusual speech patterns.

5. **Emotion Model Accuracy**: The accuracy of the emotion classification directly impacts the highlight base score and emotion intensity components.

## Conclusion

The audio highlight score is a sophisticated metric that combines multiple factors to identify noteworthy moments in streams. The current implementation gives the highest weight to the emotion type and intensity (50% combined), followed by audio features (25% combined) and speech rate (15%).

The integration with chat analysis further enhances this score by incorporating audience reactions, creating a more comprehensive highlight detection system that considers both streamer and audience perspectives.
