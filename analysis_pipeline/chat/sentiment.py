"""
Chat Sentiment Analysis Module

This module handles sentiment analysis for Twitch chat messages.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle
import psutil
from functools import partial

from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("chat_sentiment", "chat_sentiment.log")

# Define emotion clusters mapping
emotion_clusters = {
    'excitement': ['Excitement'],
    'funny': ['Humor'],
    'happiness': ['Joy'],
    'anger': ['Anger'],
    'sadness': ['Sadness'],
    'neutral': ['Neutral']
}

def load_models():
    """Load both emotion and highlight classifier models."""
    try:
        with open('models/emotion_classifier_pipe_lr.pkl', 'rb') as f:
            emotion_model = pickle.load(f)
        with open('models/highlight_classifier_pipe_lr.pkl', 'rb') as f:
            highlight_model = pickle.load(f)
        return emotion_model, highlight_model
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def process_chunk(messages, models=None):
    """Process a chunk of messages using both classifiers"""
    if models is None:
        models = load_models()
    emotion_model, highlight_model = models

    try:
        # Get predictions from both models
        emotion_predictions = emotion_model.predict(messages)
        emotion_probabilities = emotion_model.predict_proba(messages)
        highlight_scores = highlight_model.predict(messages)

        results = []
        for message, _, probs, highlight_score in zip(
            messages, emotion_predictions, emotion_probabilities, highlight_scores
        ):
            # Create base result dictionary with highlight score
            result = {
                'message': message,
                'sentiment_score': 0.0,
                'highlight_score': float(highlight_score),  # Add highlight score
                'excitement': 0.0,
                'funny': 0.0,
                'happiness': 0.0,
                'anger': 0.0,
                'sadness': 0.0,
                'neutral': 0.0
            }

            # Get the probability for each emotion class
            emotion_probs = dict(zip(emotion_model.classes_, probs))

            # Calculate cluster scores
            for cluster_name, emotions in emotion_clusters.items():
                cluster_score = sum(emotion_probs.get(emotion, 0.0) for emotion in emotions)
                result[cluster_name] = round(cluster_score, 3)

            # Calculate sentiment score
            pos_emotions = ['Excitement', 'Joy', 'Humor']
            neg_emotions = ['Anger', 'Sadness']

            pos_score = sum(emotion_probs.get(emotion, 0.0) for emotion in pos_emotions)
            neg_score = sum(emotion_probs.get(emotion, 0.0) for emotion in neg_emotions)

            sentiment_score = pos_score - neg_score
            sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp between -1 and 1
            result['sentiment_score'] = round(sentiment_score, 3)

            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return [{'message': m, 'sentiment_score': None, 'highlight_score': None,
                'excitement': None, 'funny': None, 'happiness': None,
                'anger': None, 'sadness': None, 'neutral': None} for m in messages]

def analyze_chat_sentiment(video_id, output_dir=None):
    """
    Analyze sentiment of chat messages for a Twitch VOD

    Args:
        video_id (str): The ID of the video to analyze
        output_dir (str, optional): Directory to save output files

    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results, or None if analysis failed
    """
    try:
        # Define output directory
        if output_dir is None:
            output_dir = Path('Output/Analysis/Chat')
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Determine input file path
        input_file = output_dir / f"{video_id}_processed_chat.csv"

        # Check if file exists
        if not os.path.exists(input_file):
            # Try to find the file in the raw directory
            raw_input_file = Path(f'Output/Raw/Chat/{video_id}_chat.csv')
            if os.path.exists(raw_input_file):
                logger.info(f"Processed chat file not found, using raw chat file: {raw_input_file}")
                input_file = raw_input_file
            else:
                logger.error(f"Required file not found: {input_file}")
                logger.info("Please run the chat processor first to generate the required processed chat file.")
                return None

        # Load both models once
        logger.info("Loading sentiment and highlight models")
        models = load_models()

        # Output file path
        output_file = output_dir / f"{video_id}_chat_sentiment.csv"

        # Track total processed messages

        # Determine chunk size based on available memory
        mem = psutil.virtual_memory()
        available_mem = mem.available / (1024 * 1024 * 1024)  # Convert to GB
        chunk_size = 10000 if available_mem > 8 else 5000

        # Process file in chunks to reduce memory usage
        logger.info(f"Processing file in chunks of {chunk_size} rows")
        total_processed = 0

        for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
            logger.info(f"Processing chunk {chunk_num + 1}")

            # Check required columns
            required_columns = ['time', 'message']
            missing_columns = [col for col in required_columns if col not in chunk.columns]
            if missing_columns:
                logger.error(f"Missing columns in chat data: {missing_columns}")
                continue

            # Prepare messages for analysis
            messages = chunk['message'].fillna('').tolist()

            # Calculate optimal batch size based on available memory
            batch_size = min(256 if available_mem > 16 else 128, len(messages))

            # Create batches for processing
            message_batches = np.array_split(messages, max(1, len(messages) // batch_size))
            num_batches = len(message_batches)

            logger.info(f"Processing {len(messages)} messages in {num_batches} batches")

            # Process batches using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                process_fn = partial(process_chunk, models=models)
                batch_results = list(executor.map(process_fn, message_batches))

                # Flatten results
                all_results = [item for batch_result in batch_results for item in batch_result]

            # Create DataFrame from results
            results_df = pd.DataFrame(all_results)

            if not results_df.empty:
                # Add index for merging
                results_df.index = range(len(results_df))
                chunk.index = range(len(chunk))

                # Merge the dataframes
                chunk_final_df = pd.concat([
                    chunk[['time', 'username', 'message']],
                    results_df.drop('message', axis=1)
                ], axis=1)

                # Round scores
                score_columns = [col for col in chunk_final_df.columns
                               if col not in ['time', 'username', 'message']]
                chunk_final_df[score_columns] = chunk_final_df[score_columns].round(3)

                # Append to main DataFrame or write directly to file
                if chunk_num == 0:
                    # First chunk - write with header
                    chunk_final_df.to_csv(output_file, index=False, mode='w')
                else:
                    # Subsequent chunks - append without header
                    chunk_final_df.to_csv(output_file, index=False, mode='a', header=False)

                total_processed += len(chunk_final_df)
                logger.info(f"Processed {total_processed} messages so far")
            else:
                logger.error(f"No results were generated for chunk {chunk_num + 1}")

        if total_processed > 0:
            logger.info(f"Saving sentiment analysis results to {output_file}")
            logger.info(f"Total messages processed: {total_processed}")

            # Load the final result to return
            final_df = pd.read_csv(output_file)
            return final_df
        else:
            logger.error("No results were generated from the analysis")
            return None

    except Exception as e:
        logger.error(f"Error analyzing chat sentiment: {str(e)}")
        return None
