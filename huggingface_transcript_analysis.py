#!/usr/bin/env python3
"""
Transcript Analysis using Hugging Face Models

This script analyzes transcript data using llmware's slim-sentiment-tool and slim-emotions-tool
from Hugging Face. It integrates with the existing pipeline by:
1. Taking the transcript paragraphs file as input
2. Performing sentiment and emotion analysis using the Hugging Face models
3. Updating the existing CSV file with the new analysis results
4. Comparing the results with the existing sentiment analysis

Usage:
python huggingface_transcript_analysis.py --input_file outputs/audio_VIDEOID_paragraphs.csv --output_file outputs/audio_VIDEOID_paragraphs_hf.csv
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import json
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_agents():
    """Set up the LLMfx agents for sentiment and emotion analysis."""
    logger.info("Setting up LLMfx agents...")

    try:
        from llmware.agents import LLMfx

        # Create sentiment agent
        logger.info("Creating sentiment agent...")
        sentiment_agent = LLMfx(verbose=False)
        sentiment_agent.load_tool("sentiment")

        # Create emotions agent
        logger.info("Creating emotions agent...")
        emotions_agent = LLMfx(verbose=False)
        emotions_agent.load_tool("emotions")

        # Test the agents with a sample text
        test_text = "I am extremely happy and excited about this amazing news!"
        logger.info(f"Testing sentiment agent with text: '{test_text}'")

        sentiment_result = sentiment_agent.sentiment(test_text)
        logger.info(f"Sentiment agent test result: {sentiment_result}")

        emotions_result = emotions_agent.emotions(test_text)
        logger.info(f"Emotions agent test result: {emotions_result}")

        return sentiment_agent, emotions_agent

    except ImportError as e:
        logger.error(f"Failed to import llmware: {str(e)}")
        logger.error("Make sure it's installed: pip install llmware")
        raise
    except Exception as e:
        logger.error(f"Error setting up agents: {str(e)}")
        raise

def map_sentiment_score_to_label(score):
    """Map numerical sentiment score to categorical label."""
    if score > 0.3:
        return "positive"
    elif score < -0.3:
        return "negative"
    else:
        return "neutral"

def analyze_transcript(transcript_df, sentiment_agent, emotions_agent, text_column='text'):
    """
    Analyze transcript data using sentiment and emotion agents.

    Args:
        transcript_df: DataFrame containing transcript data
        sentiment_agent: The LLMfx agent for sentiment analysis
        emotions_agent: The LLMfx agent for emotion analysis
        text_column: The column name containing the text to analyze

    Returns:
        DataFrame with original data plus sentiment and emotion analysis results
    """
    logger.info(f"Analyzing {len(transcript_df)} transcript entries...")

    # Create new columns for results
    transcript_df['hf_sentiment'] = None
    transcript_df['hf_emotion'] = None
    transcript_df['hf_sentiment_raw'] = None
    transcript_df['hf_emotion_raw'] = None
    transcript_df['hf_confidence'] = None

    # Process each transcript entry
    for idx, row in tqdm(transcript_df.iterrows(), total=len(transcript_df)):
        text = row[text_column]

        if pd.isna(text) or text.strip() == "":
            transcript_df.at[idx, 'hf_sentiment'] = 'neutral'
            transcript_df.at[idx, 'hf_emotion'] = 'neutral'
            transcript_df.at[idx, 'hf_confidence'] = 0.0
            continue

        try:
            # Perform sentiment analysis
            sentiment_result = sentiment_agent.sentiment(text)
            transcript_df.at[idx, 'hf_sentiment_raw'] = str(sentiment_result)

            # Log the first few responses for debugging
            if idx < 3:
                logger.info(f"Sample sentiment result for entry {idx}: {sentiment_result}")

            # Extract sentiment from the result
            if isinstance(sentiment_result, dict):
                # Get the sentiment from the llm_response
                if 'llm_response' in sentiment_result and 'sentiment' in sentiment_result['llm_response']:
                    sentiment_value = sentiment_result['llm_response']['sentiment']
                    if isinstance(sentiment_value, list) and len(sentiment_value) > 0:
                        sentiment = sentiment_value[0]
                    else:
                        sentiment = str(sentiment_value)
                else:
                    sentiment = 'neutral'

                # Get the confidence score
                confidence = sentiment_result.get('confidence_score', 0.0)
                transcript_df.at[idx, 'hf_confidence'] = confidence
            else:
                sentiment = 'neutral'
                transcript_df.at[idx, 'hf_confidence'] = 0.0

            # Perform emotion analysis
            emotion_result = emotions_agent.emotions(text)
            transcript_df.at[idx, 'hf_emotion_raw'] = str(emotion_result)

            # Log the first few responses for debugging
            if idx < 3:
                logger.info(f"Sample emotion result for entry {idx}: {emotion_result}")

            # Extract emotion from the result
            if isinstance(emotion_result, dict):
                # Get the emotion from the llm_response
                if 'llm_response' in emotion_result and 'emotions' in emotion_result['llm_response']:
                    emotion_value = emotion_result['llm_response']['emotions']
                    if isinstance(emotion_value, list) and len(emotion_value) > 0:
                        emotion = emotion_value[0]
                    else:
                        emotion = str(emotion_value)
                else:
                    emotion = 'neutral'
            else:
                emotion = 'neutral'

            # Store results
            transcript_df.at[idx, 'hf_sentiment'] = sentiment
            transcript_df.at[idx, 'hf_emotion'] = emotion

            # Add a small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing entry {idx}: {e}")
            transcript_df.at[idx, 'hf_sentiment'] = 'error'
            transcript_df.at[idx, 'hf_emotion'] = 'error'
            transcript_df.at[idx, 'hf_confidence'] = 0.0

    return transcript_df

def compare_sentiment_analysis(df):
    """
    Compare the existing sentiment analysis with the Hugging Face sentiment analysis.

    Args:
        df: DataFrame with both sentiment analyses

    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Comparing sentiment analyses...")

    # Map existing sentiment scores to categorical labels
    if 'sentiment_score' in df.columns:
        df['existing_sentiment_label'] = df['sentiment_score'].apply(map_sentiment_score_to_label)
    else:
        logger.warning("No 'sentiment_score' column found. Skipping comparison.")
        return df

    # Calculate agreement
    agreement = (df['existing_sentiment_label'] == df['hf_sentiment']).mean() * 100
    logger.info(f"Agreement between existing and Hugging Face sentiment analysis: {agreement:.2f}%")

    # Create a confusion matrix
    confusion = pd.crosstab(
        df['existing_sentiment_label'],
        df['hf_sentiment'],
        rownames=['Existing'],
        colnames=['Hugging Face']
    )
    logger.info(f"Confusion matrix:\n{confusion}")

    return df

def analyze_emotions(df):
    """
    Analyze the distribution of emotions in the transcript.

    Args:
        df: DataFrame with emotion analysis

    Returns:
        DataFrame with emotion analysis metrics
    """
    logger.info("Analyzing emotions...")

    # Count emotions
    emotion_counts = df['hf_emotion'].value_counts()
    logger.info(f"Emotion distribution:\n{emotion_counts}")

    # Calculate percentage of each emotion
    emotion_percentages = (emotion_counts / len(df)) * 100
    logger.info(f"Emotion percentages:\n{emotion_percentages}")

    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze transcript data using Hugging Face models')
    parser.add_argument('--input_file', required=True, help='Path to the input CSV file containing transcript data')
    parser.add_argument('--output_file', required=True, help='Path to save the output CSV file with analysis results')
    parser.add_argument('--text_column', default='text', help='Column name containing the text to analyze (default: text)')
    parser.add_argument('--batch_size', type=int, default=None, help='Number of entries to process (for testing)')

    args = parser.parse_args()

    # Load transcript data
    logger.info(f"Loading transcript data from {args.input_file}...")
    try:
        transcript_df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(transcript_df)} transcript entries")

        # For testing, limit to batch_size entries
        if args.batch_size:
            transcript_df = transcript_df.head(args.batch_size)
            logger.info(f"Limited to {len(transcript_df)} entries for testing")
    except Exception as e:
        logger.error(f"Error loading transcript data: {e}")
        return

    # Set up agents
    try:
        sentiment_agent, emotions_agent = setup_agents()
    except Exception as e:
        logger.error(f"Error setting up agents: {e}")
        return

    # Process the DataFrame
    result_df = analyze_transcript(
        transcript_df,
        sentiment_agent,
        emotions_agent,
        text_column=args.text_column
    )

    # Compare sentiment analyses
    result_df = compare_sentiment_analysis(result_df)

    # Analyze emotions
    result_df = analyze_emotions(result_df)

    # Save results
    logger.info(f"Saving analysis results to {args.output_file}...")
    result_df.to_csv(args.output_file, index=False)
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
