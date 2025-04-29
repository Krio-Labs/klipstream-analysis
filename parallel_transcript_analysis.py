#!/usr/bin/env python3
"""
Parallel Transcript Analysis using Hugging Face Models

This script analyzes transcript data in parallel using llmware's sentiment and emotion tools.
It splits the data into chunks and processes each chunk in a separate process.

Usage:
python parallel_transcript_analysis.py --input_file outputs/audio_VIDEOID_paragraphs.csv --output_file outputs/audio_VIDEOID_paragraphs_hf.csv --num_processes 4
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import logging
import time
import multiprocessing
from functools import partial
import tempfile

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
    try:
        from llmware.agents import LLMfx
        
        # Create sentiment agent
        sentiment_agent = LLMfx(verbose=False)
        sentiment_agent.load_tool("sentiment")
        
        # Create emotions agent
        emotions_agent = LLMfx(verbose=False)
        emotions_agent.load_tool("emotions")
        
        return sentiment_agent, emotions_agent
    
    except ImportError as e:
        logger.error(f"Failed to import llmware: {str(e)}")
        logger.error("Make sure it's installed: pip install llmware")
        raise
    except Exception as e:
        logger.error(f"Error setting up agents: {str(e)}")
        raise

def analyze_text(text, sentiment_agent, emotions_agent):
    """
    Analyze a single text entry with sentiment and emotion agents.
    
    Args:
        text: The text to analyze
        sentiment_agent: The LLMfx agent for sentiment analysis
        emotions_agent: The LLMfx agent for emotion analysis
        
    Returns:
        Dictionary with analysis results
    """
    if pd.isna(text) or text.strip() == "":
        return {
            'hf_sentiment': 'neutral',
            'hf_emotion': 'neutral',
            'hf_confidence': 0.0
        }
        
    try:
        # Perform sentiment analysis
        sentiment_result = sentiment_agent.sentiment(text)
        
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
            if isinstance(confidence, np.float64):
                confidence = float(confidence)
        else:
            sentiment = 'neutral'
            confidence = 0.0
            
        # Perform emotion analysis
        emotion_result = emotions_agent.emotions(text)
        
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
            
        return {
            'hf_sentiment': sentiment,
            'hf_emotion': emotion,
            'hf_confidence': confidence,
            'hf_sentiment_raw': str(sentiment_result),
            'hf_emotion_raw': str(emotion_result)
        }
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return {
            'hf_sentiment': 'error',
            'hf_emotion': 'error',
            'hf_confidence': 0.0
        }

def process_chunk(chunk_df, chunk_id, temp_dir, text_column='text'):
    """
    Process a chunk of transcript data.
    
    Args:
        chunk_df: DataFrame containing a chunk of transcript data
        chunk_id: ID of the chunk (for logging)
        temp_dir: Directory to save temporary results
        text_column: Column name containing the text to analyze
        
    Returns:
        Path to the saved chunk results
    """
    logger.info(f"Processing chunk {chunk_id} with {len(chunk_df)} entries")
    
    # Set up agents for this process
    sentiment_agent, emotions_agent = setup_agents()
    
    # Process each entry in the chunk
    results = []
    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Chunk {chunk_id}"):
        text = row[text_column]
        
        # Analyze the text
        analysis_result = analyze_text(text, sentiment_agent, emotions_agent)
        
        # Add the index to the result
        analysis_result['index'] = idx
        results.append(analysis_result)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.05)
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    # Save the results to a temporary file
    output_file = os.path.join(temp_dir, f"chunk_{chunk_id}.csv")
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"Chunk {chunk_id} processing complete. Results saved to {output_file}")
    
    return output_file

def map_sentiment_score_to_label(score):
    """Map numerical sentiment score to categorical label."""
    if score > 0.3:
        return "positive"
    elif score < -0.3:
        return "negative"
    else:
        return "neutral"

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
    parser = argparse.ArgumentParser(description='Analyze transcript data in parallel using Hugging Face models')
    parser.add_argument('--input_file', required=True, help='Path to the input CSV file containing transcript data')
    parser.add_argument('--output_file', required=True, help='Path to save the output CSV file with analysis results')
    parser.add_argument('--text_column', default='text', help='Column name containing the text to analyze (default: text)')
    parser.add_argument('--batch_size', type=int, default=None, help='Number of entries to process (for testing)')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--chunk_size', type=int, default=None, help='Size of chunks for parallel processing (default: auto)')
    
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
    
    # Determine number of processes
    if args.num_processes:
        num_processes = args.num_processes
    else:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"Using {num_processes} processes for parallel processing")
    
    # Determine chunk size
    total_entries = len(transcript_df)
    if args.chunk_size:
        chunk_size = args.chunk_size
    else:
        # Aim for at least 5 entries per chunk, but no more than 100
        chunk_size = min(max(5, total_entries // num_processes), 100)
    
    logger.info(f"Using chunk size of {chunk_size} entries")
    
    # Split the data into chunks
    chunks = []
    for i in range(0, total_entries, chunk_size):
        chunk = transcript_df.iloc[i:i+chunk_size].copy()
        chunks.append(chunk)
    
    logger.info(f"Split data into {len(chunks)} chunks")
    
    # Create a temporary directory for chunk results
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory for chunk results: {temp_dir}")
        
        # Process chunks in parallel
        chunk_results = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Create a partial function with fixed arguments
            process_chunk_partial = partial(
                process_chunk,
                temp_dir=temp_dir,
                text_column=args.text_column
            )
            
            # Process chunks in parallel
            chunk_results = list(
                tqdm(
                    pool.starmap(
                        process_chunk_partial,
                        [(chunk, i) for i, chunk in enumerate(chunks)]
                    ),
                    total=len(chunks),
                    desc="Processing chunks"
                )
            )
        
        logger.info(f"All chunks processed. Combining results...")
        
        # Load and combine chunk results
        result_dfs = []
        for result_file in chunk_results:
            chunk_df = pd.read_csv(result_file)
            result_dfs.append(chunk_df)
        
        # Combine all results
        combined_results = pd.concat(result_dfs, ignore_index=True)
        
        # Sort by the original index
        combined_results = combined_results.sort_values('index')
        
        # Set the index to match the original DataFrame
        combined_results.set_index('index', inplace=True)
        
        # Merge with the original DataFrame
        result_df = transcript_df.copy()
        for col in combined_results.columns:
            result_df[col] = combined_results[col]
    
    logger.info(f"Combined results from {len(chunks)} chunks")
    
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
