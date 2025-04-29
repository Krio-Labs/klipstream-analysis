#!/usr/bin/env python3
"""
Test script for transcript analysis using llmware's slim-sentiment-tool and slim-emotions-tool.

This script:
1. Loads transcript data from a CSV file
2. Performs sentiment analysis using slim-sentiment-tool
3. Performs emotion analysis using slim-emotions-tool
4. Combines the results and saves to a new CSV file

Requirements:
- llmware
- pandas
- huggingface_hub
- tqdm

Usage:
python test_transcript_analysis.py --input_file path/to/transcript.csv --output_file path/to/output.csv
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import json
import logging
from huggingface_hub import snapshot_download

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_models():
    """Download and set up the models from Hugging Face."""
    logger.info("Setting up models...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download models from Hugging Face
    sentiment_model_path = os.path.join("models", "slim-sentiment-tool")
    emotions_model_path = os.path.join("models", "slim-emotions-tool")
    
    if not os.path.exists(sentiment_model_path):
        logger.info("Downloading slim-sentiment-tool...")
        snapshot_download(
            "llmware/slim-sentiment-tool", 
            local_dir=sentiment_model_path, 
            local_dir_use_symlinks=False
        )
    
    if not os.path.exists(emotions_model_path):
        logger.info("Downloading slim-emotions-tool...")
        snapshot_download(
            "llmware/slim-emotions-tool", 
            local_dir=emotions_model_path, 
            local_dir_use_symlinks=False
        )
    
    logger.info("Models downloaded successfully")
    
    # Import llmware after downloading models to ensure proper setup
    try:
        from llmware.models import ModelCatalog
        
        # Load the models
        logger.info("Loading sentiment model...")
        sentiment_model = ModelCatalog().load_model("slim-sentiment-tool")
        
        logger.info("Loading emotions model...")
        emotions_model = ModelCatalog().load_model("slim-emotions-tool")
        
        return sentiment_model, emotions_model
    except ImportError:
        logger.error("Failed to import llmware. Make sure it's installed: pip install llmware")
        raise

def analyze_transcript(transcript_df, sentiment_model, emotions_model, text_column='text'):
    """
    Analyze transcript data using sentiment and emotion models.
    
    Args:
        transcript_df: DataFrame containing transcript data
        sentiment_model: The sentiment analysis model
        emotions_model: The emotion analysis model
        text_column: The column name containing the text to analyze
        
    Returns:
        DataFrame with original data plus sentiment and emotion analysis results
    """
    logger.info(f"Analyzing {len(transcript_df)} transcript entries...")
    
    # Create new columns for results
    transcript_df['sentiment'] = None
    transcript_df['emotions'] = None
    
    # Process each transcript entry
    for idx, row in tqdm(transcript_df.iterrows(), total=len(transcript_df)):
        text = row[text_column]
        
        if pd.isna(text) or text.strip() == "":
            continue
            
        try:
            # Perform sentiment analysis
            sentiment_response = sentiment_model.function_call(text)
            if isinstance(sentiment_response, str):
                sentiment_data = json.loads(sentiment_response.replace("'", "\""))
                sentiment = sentiment_data.get('sentiment', ['neutral'])[0]
            else:
                sentiment = sentiment_response.get('sentiment', ['neutral'])[0]
                
            # Perform emotion analysis
            emotion_response = emotions_model.function_call(text)
            if isinstance(emotion_response, str):
                emotion_data = json.loads(emotion_response.replace("'", "\""))
                emotions = emotion_data.get('emotions', ['neutral'])[0]
            else:
                emotions = emotion_response.get('emotions', ['neutral'])[0]
                
            # Store results
            transcript_df.at[idx, 'sentiment'] = sentiment
            transcript_df.at[idx, 'emotions'] = emotions
            
        except Exception as e:
            logger.error(f"Error processing entry {idx}: {e}")
            transcript_df.at[idx, 'sentiment'] = 'error'
            transcript_df.at[idx, 'emotions'] = 'error'
    
    return transcript_df

def main():
    parser = argparse.ArgumentParser(description='Analyze transcript data using sentiment and emotion models')
    parser.add_argument('--input_file', required=True, help='Path to the input CSV file containing transcript data')
    parser.add_argument('--output_file', required=True, help='Path to save the output CSV file with analysis results')
    parser.add_argument('--text_column', default='text', help='Column name containing the text to analyze (default: text)')
    
    args = parser.parse_args()
    
    # Load transcript data
    logger.info(f"Loading transcript data from {args.input_file}...")
    try:
        transcript_df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(transcript_df)} transcript entries")
    except Exception as e:
        logger.error(f"Error loading transcript data: {e}")
        return
    
    # Set up models
    try:
        sentiment_model, emotions_model = setup_models()
    except Exception as e:
        logger.error(f"Error setting up models: {e}")
        return
    
    # Analyze transcript
    result_df = analyze_transcript(
        transcript_df, 
        sentiment_model, 
        emotions_model,
        text_column=args.text_column
    )
    
    # Save results
    logger.info(f"Saving analysis results to {args.output_file}...")
    result_df.to_csv(args.output_file, index=False)
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
