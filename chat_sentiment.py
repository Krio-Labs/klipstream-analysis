import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil
from functools import partial
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.error(f"Error loading models: {str(e)}")
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
        for message, prediction, probs, highlight_score in zip(
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
        logging.error(f"Error processing batch: {str(e)}")
        return [{'message': m, 'sentiment_score': None, 'highlight_score': None,
                'excitement': None, 'funny': None, 'happiness': None, 
                'anger': None, 'sadness': None, 'neutral': None} for m in messages]

def analyze_chat_sentiment(video_id):
    """Analyze sentiment for a specific video's chat data."""
    try:
        # Construct input and output file paths
        input_file = f'data/{video_id}_chat_preprocessed.csv'
        output_file = f'outputs/{video_id}_chat_sentiment.csv'

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Chat file not found: {input_file}")

        # Load both models once
        models = load_models()
        
        # Initialize an empty DataFrame to store all results
        all_final_df = pd.DataFrame()
        
        # Load data in chunks to reduce memory usage
        chunk_size = 10000
        for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
            
            logging.info(f"Processing chunk {chunk_num + 1}")
            messages = chunk['message'].values
            
            # Calculate optimal batch size based on available memory
            mem = psutil.virtual_memory()
            available_mem = mem.available / (1024 * 1024 * 1024)  # Convert to GB
            batch_size = min(256 if available_mem > 16 else 128, len(messages))
            
            # Create batches for processing
            message_batches = np.array_split(messages, max(1, len(messages) // batch_size))
            
            # Process batches using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                process_fn = partial(process_chunk, models=models)
                results = list(executor.map(process_fn, message_batches))
            
            # Flatten results
            all_results = [item for batch_result in results for item in batch_result]
            
            # Create DataFrame from results
            output_df = pd.DataFrame(all_results)
            
            if not output_df.empty:
                # Add index for merging
                output_df.index = range(len(output_df))
                chunk.index = range(len(chunk))
                
                # Merge the dataframes
                chunk_final_df = pd.concat([
                    chunk[['time', 'username', 'message']],
                    output_df.drop('message', axis=1)
                ], axis=1)
                
                # Round scores
                score_columns = [col for col in chunk_final_df.columns 
                               if col not in ['time', 'username', 'message']]
                chunk_final_df[score_columns] = chunk_final_df[score_columns].round(3)
                
                # Append to main DataFrame
                all_final_df = pd.concat([all_final_df, chunk_final_df], ignore_index=True)
                
                logging.info(f"Processed {len(all_final_df)} messages so far")
            else:
                logging.error(f"No results were generated for chunk {chunk_num + 1}")
        
        if not all_final_df.empty:
            # Create outputs directory if it doesn't exist
            os.makedirs('outputs', exist_ok=True)
            
            # Save complete results
            all_final_df.to_csv(output_file, index=False)
            logging.info(f"Analysis completed. Total messages processed: {len(all_final_df)}")
            logging.info(f"Results saved to {output_file}")
            
            # Delete the input file
            try:
                os.remove(input_file)
                logging.info(f"Deleted input file: {input_file}")
            except Exception as e:
                logging.warning(f"Could not delete input file {input_file}: {str(e)}")
            
            return output_file
        else:
            logging.error("No results were generated from the analysis")
            return None
            
    except Exception as e:
        logging.error(f"Error in analyze_chat_sentiment: {str(e)}")
        raise

def main():
    """Main function that can be called with a video ID."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python chat_sentiment.py <video_id>")
        sys.exit(1)
        
    video_id = sys.argv[1]
    try:
        output_file = analyze_chat_sentiment(video_id)
        if output_file:
            print(f"Successfully analyzed sentiment for video {video_id}")
            print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error analyzing sentiment for video {video_id}: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
