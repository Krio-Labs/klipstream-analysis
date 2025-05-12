"""
Chat Processor Module

This module handles the processing of Twitch chat data.
"""

import pandas as pd
import numpy as np
import re
import os
import csv
from pathlib import Path
import emoji

from utils.logging_setup import setup_logger

# Set up logger
logger = setup_logger("chat_processor", "chat_processor.log")

# Directory and file setup
script_dir = Path(__file__).parent
dict_dir = script_dir / 'dictionary'

# Create directories if they don't exist
os.makedirs(dict_dir, exist_ok=True)

# Define dictionary files
emote_dict_file = dict_dir / "emote_dictionary.csv"
emoji_dict_file = dict_dir / "emoji_dictionary.csv"
slang_dict_file = dict_dir / "slang_dictionary.csv"

# Define filtering patterns
NUMERIC_PATTERN = re.compile(r'^\d+$')
COMMAND_PATTERN = re.compile(r'^[/!].*')
URL_PATTERN = re.compile(r'http[s]?://')
GREETING_PATTERN = re.compile(r'^(hi|hey|hello|sup)\s+\w+$')

ADVERTISEMENT_KEYWORDS = ['buy', 'subscribe', 'follow']
GIFTING_KEYWORDS = ['gift', 'sub', 'donate']
GENERIC_EXPRESSIONS = ['nice', 'good game', 'lol', 'gg']

# Constants and keyword maps for filtering
BOT_NAMES = {
    "nightbot": True,
    "streamelements": True,
    "moobot": True
}

SUBSCRIPTION_KEYWORDS = ADVERTISEMENT_KEYWORDS + GIFTING_KEYWORDS + GENERIC_EXPRESSIONS

def create_default_dictionaries():
    """Create default dictionary files if they don't exist"""
    # Create emote dictionary
    if not os.path.exists(emote_dict_file):
        logger.info(f"Creating default emote dictionary at {emote_dict_file}")
        with open(emote_dict_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['emote', 'meaning'])
            writer.writerows([
                ['Kappa', 'sarcasm'],
                ['PogChamp', 'excitement'],
                ['LUL', 'laughter'],
                ['BibleThump', 'sadness'],
                ['ResidentSleeper', 'boredom'],
                ['TriHard', 'trying hard'],
                ['4Head', 'laughter'],
                ['DansGame', 'disgust'],
                ['Kreygasm', 'excitement'],
                ['SwiftRage', 'anger']
            ])

    # Create emoji dictionary
    if not os.path.exists(emoji_dict_file):
        logger.info(f"Creating default emoji dictionary at {emoji_dict_file}")
        with open(emoji_dict_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['emoji', 'meaning'])
            writer.writerows([
                ['😂', 'laughter'],
                ['😊', 'happiness'],
                ['😍', 'love'],
                ['🔥', 'fire'],
                ['👍', 'approval'],
                ['❤️', 'love'],
                ['😭', 'crying'],
                ['😢', 'sadness'],
                ['😡', 'anger'],
                ['🤔', 'thinking']
            ])

    # Create slang dictionary
    if not os.path.exists(slang_dict_file):
        logger.info(f"Creating default slang dictionary at {slang_dict_file}")
        with open(slang_dict_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['slang', 'meaning'])
            writer.writerows([
                ['pog', 'excitement'],
                ['poggers', 'excitement'],
                ['monkas', 'anxiety'],
                ['pepega', 'silly'],
                ['omegalul', 'extreme laughter'],
                ['pepelaugh', 'laughing at someone'],
                ['sadge', 'sadness'],
                ['kekw', 'laughter'],
                ['weirdchamp', 'disapproval'],
                ['peeposad', 'sadness']
            ])

def load_dictionaries():
    """Load dictionaries from CSV files"""
    try:
        # Create default dictionaries if they don't exist
        create_default_dictionaries()

        # Load dictionaries
        emote_df = pd.read_csv(emote_dict_file)
        emoji_df = pd.read_csv(emoji_dict_file)
        slang_df = pd.read_csv(slang_dict_file)

        # Convert to dictionaries
        emote_dict = pd.Series(emote_df['meaning'].values, index=emote_df['emote']).to_dict()
        emoji_dict = pd.Series(emoji_df['meaning'].values, index=emoji_df['emoji']).to_dict()
        slang_dict = pd.Series(slang_df['meaning'].values, index=slang_df['slang']).to_dict()

        return emote_dict, emoji_dict, slang_dict

    except Exception as e:
        logger.error(f"Error loading dictionaries: {str(e)}")
        # Return empty dictionaries as fallback
        return {}, {}, {}

def filters(message_data):
    """
    Filter chat messages based on various criteria.

    Args:
        message_data (dict): Dictionary containing message data

    Returns:
        bool: True if message passes filters, False otherwise
    """
    username = message_data['username']
    message = message_data['message']

    # Early returns for basic checks
    if pd.isna(message) or not isinstance(message, str) or not message.strip():
        return False

    if username.lower() in BOT_NAMES:
        return False

    if '@' in message:
        return False

    message_lower = message.lower()

    # Check against keyword maps
    if any(keyword in message_lower for keyword in SUBSCRIPTION_KEYWORDS):
        return False

    # Check against patterns
    if any([
        NUMERIC_PATTERN.match(message),
        COMMAND_PATTERN.match(message),
        GREETING_PATTERN.match(message_lower),
        URL_PATTERN.search(message)
    ]):
        return False

    return True

def clean_message(message):
    """
    Clean and normalize chat messages

    Args:
        message (str): The chat message to clean

    Returns:
        str: The cleaned message
    """
    if not isinstance(message, str):
        return ""

    # Convert to lowercase
    message = message.lower()

    # Remove URLs
    message = re.sub(r'https?://\S+', '', message)

    # Remove special characters but keep emojis
    message = re.sub(r'[^\w\s\U00010000-\U0010ffff]', '', message)

    # Remove extra whitespace
    message = re.sub(r'\s+', ' ', message).strip()

    return message

def extract_emojis(message):
    """
    Extract emojis from a message

    Args:
        message (str): The chat message to extract emojis from

    Returns:
        list: List of emojis found in the message
    """
    if not isinstance(message, str):
        return []

    return [c for c in message if c in emoji.EMOJI_DATA]

def preprocess_message(message, emote_dict, emoji_dict, slang_dict):
    """
    Simplified preprocessing of Twitch chat messages.

    Args:
        message (str): The message to preprocess
        emote_dict (dict): Dictionary of emotes and their meanings
        emoji_dict (dict): Dictionary of emojis and their meanings
        slang_dict (dict): Dictionary of slang terms and their meanings

    Returns:
        str: The preprocessed message
    """
    # Handle non-string messages
    if not isinstance(message, str):
        return ""

    # Normalize excessive whitespace
    message = re.sub(r'\s+', ' ', message).strip()

    # Just return the cleaned message for now to avoid performance issues
    return message

def process_chat_data(video_id, input_file=None, output_dir=None):
    """
    Process chat data for a Twitch VOD

    Args:
        video_id (str): The ID of the video to process
        input_file (str, optional): Path to the input chat CSV file
        output_dir (str, optional): Directory to save output files

    Returns:
        pd.DataFrame: Processed chat data, or None if processing failed
    """
    try:
        # Define output directory
        if output_dir is None:
            output_dir = Path('output/Analysis/Chat')
        else:
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Determine input file path
        if input_file is None:
            input_file = Path(f'output/Raw/Chat/{video_id}_chat.csv')
        else:
            input_file = Path(input_file)

        # Check if file exists
        if not os.path.exists(input_file):
            logger.error(f"Required file not found: {input_file}")
            logger.info("Please run the full pipeline first to generate the required chat file.")
            return None

        # Load chat data
        logger.info(f"Loading chat data from {input_file}")
        chat_data = pd.read_csv(input_file)

        # Check required columns
        required_columns = ['time', 'username', 'message']
        missing_columns = [col for col in required_columns if col not in chat_data.columns]
        if missing_columns:
            logger.error(f"Missing columns in chat data: {missing_columns}")
            return None

        # Drop NaN values
        chat_data = chat_data.dropna(subset=['message', 'username'])

        # Apply filters
        logger.info("Cleaning and normalizing messages")
        chat_data = chat_data[chat_data.apply(filters, axis=1)].copy()

        # Add lowercase message column for later use
        chat_data['message_lower'] = chat_data['message'].str.lower()

        # Skip duplicate removal for now as it's causing performance issues
        logger.info("Skipping duplicate removal to improve performance")

        # Load dictionaries
        emote_dict, emoji_dict, slang_dict = load_dictionaries()

        # Preprocess messages
        chat_data['cleaned_message'] = chat_data['message'].apply(
            lambda msg: preprocess_message(msg, emote_dict, emoji_dict, slang_dict)
        )

        # Extract emojis
        logger.info("Extracting emojis")
        chat_data['emojis'] = chat_data['message'].apply(extract_emojis)
        chat_data['emoji_count'] = chat_data['emojis'].apply(len)

        # Calculate message length
        chat_data['message_length'] = chat_data['cleaned_message'].apply(len)

        # Convert time to numeric if it's not already
        if chat_data['time'].dtype != 'float64' and chat_data['time'].dtype != 'int64':
            chat_data['time'] = pd.to_numeric(chat_data['time'], errors='coerce')

        # Sort by time
        chat_data = chat_data.sort_values('time')

        # Calculate time intervals
        chat_data['time_diff'] = chat_data['time'].diff()

        # Calculate message rate (messages per minute)
        window_size = 60  # 1 minute window
        chat_data['message_rate'] = chat_data['time'].rolling(window=window_size).count()

        # Save processed data
        output_file = output_dir / f"{video_id}_processed_chat.csv"
        logger.info(f"Saving processed chat data to {output_file}")
        chat_data.to_csv(output_file, index=False)

        return chat_data

    except Exception as e:
        logger.error(f"Error processing chat data: {str(e)}")
        return None
