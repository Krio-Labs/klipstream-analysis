import pandas as pd
from transformers import pipeline
import torch
import platform
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import psutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from collections import defaultdict
import os
import json
import google.generativeai as genai
import random
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the matrix globally in each process
def init_pool_processes(m):
    global matrix
    matrix = m

def addMatrixRow(rowNum):
    # Access the global matrix
    return sum(matrix[rowNum])

def genMatrix(row, col):
    return [[random.randint(0, 1) for _ in range(col)] for _ in range(row)]

def compute_chunksize(pool_size, iterable_size):
    chunksize, remainder = divmod(iterable_size, 4 * pool_size)
    if remainder:
        chunksize += 1
    return chunksize

def init_worker(api_key):
    """Initialize worker process with Gemini model"""
    try:
        global model
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        
        # Add delay between worker initializations to avoid GRPC issues
        time.sleep(1)
    except Exception as e:
        logging.error(f"Error initializing worker: {str(e)}")
        raise

def process_chunk(paragraphs, start_times, end_times, batch_delay=0.1):
    """Process a chunk of paragraphs using Gemini 1.5 Flash-8B"""
    results = []
    try:
        total = len(paragraphs)
        for idx, (text, start_time, end_time) in enumerate(zip(paragraphs, start_times, end_times), 1):
            if not text.strip():
                logging.info(f"Skipping empty text [{idx}/{total}]")
                continue
            
            try:
                time.sleep(batch_delay)
                logging.info(f"Processing text [{idx}/{total}]: {text[:50]}...")
                
                # Create prompt and get response
                prompt = f"""
                You are an expert at analyzing text and identifying emotions for twitch streams. Analyze the following text and respond ONLY with a JSON object containing emotion probabilities.
                When you analyze the text, please consider nuances of twitch lingo.
                The highlight_score ranges from 0 to 1 and indicates the likelihood of the text being something worth marking into a social media clip.
                The sentiment_score ranges from -1 (most negative) to 1 (most positive).
                
                Text: "{text}"

                Respond with ONLY this JSON format, no other text:
                {{
                    "excitement": 0.0,
                    "funny": 0.0,
                    "happiness": 0.0,
                    "anger": 0.0,
                    "sadness": 0.0,
                    "neutral": 0.0,
                    "sentiment_score": 0.0,
                    "highlight_score": 0.0
                }}
                """
                
                # Get response from Gemini
                logging.debug(f"Sending request to Gemini API [{idx}/{total}]")
                response = model.generate_content(prompt)
                
                # Process response
                response_text = response.text.strip()
                if response_text.startswith("```json") and response_text.endswith("```"):
                    response_text = response_text[7:-3].strip()
                
                scores = json.loads(response_text)
                
                # Add the text and timestamps to the scores dictionary
                scores.update({
                    'text': text,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                logging.info(f"Successfully processed text [{idx}/{total}] with sentiment: {scores['sentiment_score']:.2f}")
                results.append(scores)
                
            except Exception as e:
                logging.error(f"Error processing text [{idx}/{total}]: {str(e)}")
                continue
                
        logging.info(f"Completed chunk processing: {len(results)}/{total} texts successful")
        return results
        
    except Exception as e:
        logging.error(f"Error in process_chunk: {str(e)}")
        return []

def clean_up_workers():
    """Kill any remaining worker processes and cleanup resources."""
    try:
        # Kill child processes
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Clean up multiprocessing resources - Python 3.12 compatible
        import multiprocessing.resource_tracker
        try:
            # Get the resource tracker singleton
            tracker = multiprocessing.resource_tracker._resource_tracker
            if tracker is not None:
                tracker.clear()
        except Exception as e:
            logging.debug(f"Resource tracker cleanup skipped: {str(e)}")

    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

def main(input_file='outputs/audio_paragraphs.csv', api_key=None):
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable must be set")

    try:
        # Read the existing CSV file
        logging.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        paragraphs = df['text'].tolist()
        start_times = df['start_time'].tolist()
        end_times = df['end_time'].tolist()
        
        total_paragraphs = len(paragraphs)
        logging.info(f"Found {total_paragraphs} paragraphs to process")
        
        # Create batches for all three lists
        batch_size = 15
        paragraph_batches = [paragraphs[i:i + batch_size] for i in range(0, len(paragraphs), batch_size)]
        start_time_batches = [start_times[i:i + batch_size] for i in range(0, len(start_times), batch_size)]
        end_time_batches = [end_times[i:i + batch_size] for i in range(0, len(end_times), batch_size)]
        num_batches = len(paragraph_batches)
        logging.info(f"Split into {num_batches} batches of size {batch_size}")
        
        # Configure Gemini in the main process
        genai.configure(api_key=api_key)
        
        # Process in parallel
        results = []
        num_workers = max(cpu_count()-1, 8)
        logging.info(f"Starting processing with {num_workers} workers")
        
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(api_key,)
        ) as executor:
            batch_results = []
            for i, batch_result in enumerate(executor.map(
                process_chunk,
                paragraph_batches,
                start_time_batches,
                end_time_batches
            ), 1):
                batch_results.append(batch_result)
                logging.info(f"Completed batch {i}/{num_batches} ({(i/num_batches)*100:.1f}%)")
            
            results = [item for sublist in batch_results if sublist for item in sublist]
            
        logging.info(f"Processing complete. Got results for {len(results)}/{total_paragraphs} paragraphs")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add new columns to original DataFrame
        emotion_columns = [col for col in results_df.columns if col not in ['start_time', 'end_time', 'text']]
        for col in emotion_columns:
            df[col] = results_df[col]
        
        # Save back to the same file
        logging.info(f"Updating input file with new columns: {input_file}")
        df.to_csv(input_file, index=False)
        logging.info("Save complete")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
    finally:
        clean_up_workers()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be set")
        
    main(api_key=api_key)
