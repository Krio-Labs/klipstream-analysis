# llmware Emotion Analysis Tool

## Overview

The `llmware/slim-emotions-tool` is a specialized model for detecting emotions in text. This document outlines the complete set of emotions that this model can classify, based on extensive testing.

## Complete List of Emotions

Through systematic testing of hundreds of emotion terms in different contexts, we've determined that the `llmware/slim-emotions-tool` model can classify the following 35 distinct emotions:

```json
[
  "afraid",
  "anger",
  "angry",
  "annoyed",
  "anticipating",
  "anxious",
  "apprehensive",
  "ashamed",
  "caring",
  "confident",
  "confused",
  "content",
  "devastated",
  "disappointed",
  "disgusted",
  "embarrassed",
  "excited",
  "faithful",
  "furious",
  "grateful",
  "guilty",
  "hopeful",
  "impressed",
  "jealous",
  "joy",
  "joyful",
  "lonely",
  "neutral",
  "nostalgic",
  "prepared",
  "proud",
  "sad",
  "sentimental",
  "surprised",
  "trusting"
]
```

## Emotion Categories

These emotions can be grouped into three main categories:

### Positive Emotions
- joy
- joyful
- excited
- content
- grateful
- hopeful
- proud
- impressed
- confident
- caring
- faithful

### Negative Emotions
- sad
- afraid
- angry
- anger
- annoyed
- furious
- disappointed
- devastated
- guilty
- ashamed
- embarrassed
- disgusted
- lonely

### Complex/Neutral Emotions
- surprised
- anticipating
- apprehensive
- anxious
- confused
- sentimental
- nostalgic
- neutral
- prepared
- trusting
- jealous

## Using the Model

The `llmware/slim-emotions-tool` model can be used through the llmware Python package:

```python
from llmware.agents import LLMfx

def analyze_emotions(text):
    """
    Analyze emotions in the given text using llmware's slim-emotions-tool.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: A dictionary containing detected emotions
    """
    # Initialize the LLMfx agent
    llm_fx = LLMfx()
    
    # Load the emotions tool
    llm_fx.load_tool("emotions")
    
    # Analyze text for emotions
    response = llm_fx.emotions(text)
    
    return response

# Example usage
text = "I'm really excited about this new project!"
emotions = analyze_emotions(text)
print(f"Detected emotions: {emotions}")
```

## Emotion Mapping

The model maps various input emotions to its core set of classifications. Here are some examples:

- "happy", "cheerful", "merry", "jovial" → **joyful**
- "ecstatic", "thrilled", "exhilarated" → **excited**
- "depressed", "gloomy", "heartbroken" → **sad**
- "terrified", "scared", "panicked" → **afraid**
- "nervous", "worried", "wary" → **apprehensive**
- "annoyed", "irritated", "aggravated" → **annoyed**
- "grateful", "thankful", "appreciative" → **grateful**

## Implementation Tips

1. **Handle All 35 Emotions**: Design your system to handle all 35 emotions identified in our testing.

2. **Group Similar Emotions**: Consider grouping semantically similar emotions (like "anger"/"angry" or "joy"/"joyful") if your application doesn't need such fine-grained distinctions.

3. **Confidence Scores**: The model provides confidence scores for its classifications. You may want to set a threshold for accepting classifications.

4. **Test with Your Specific Content**: The emotions detected may vary based on your specific content and context, so test with representative samples from your domain.

## Resources

- The `llmware_emotions.json` file contains the complete list of emotions in a structured format.
- The `emotion_analyzer.py` script provides a ready-to-use class for emotion analysis.

## Testing Methodology

This list was determined through:
1. Testing hundreds of different emotion terms
2. Using multiple sentence contexts for each emotion
3. Analyzing the model's responses and confidence scores
4. Examining which input emotions map to which output classifications

The testing scripts created diverse contexts for each emotion (e.g., "I'm feeling [emotion] about the situation", "The news made me [emotion]") to ensure robust results.