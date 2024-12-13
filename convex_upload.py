from typing import Optional, TypedDict, Literal
import aiohttp
import json
import os
import asyncio
from typing import List
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SavedFileData(TypedDict):
    storageId: str
    url: str

class UpdateVideo(TypedDict, total=False):  
    id: str  # type: ignore  
    status: str
    chatId: str
    chatUrl: str
    transcriptId: str
    transcriptUrl: str
    audiowaveId: str
    audiowaveUrl: str
    chatAnalysisId: str
    chatAnalysisUrl: str
    transcriptAnalysisId: str
    transcriptAnalysisUrl: str
    transcriptWordId: str
    transcriptWordUrl: str

FileType = Literal['audiowave', 'transcript', 'chat', 'chatAnalysis', 'transcriptAnalysis', 'transcriptWord']

type_to_fields = {
    'audiowave': ('audiowaveId', 'audiowaveUrl'),
    'transcript': ('transcriptId', 'transcriptUrl'),
    'chat': ('chatId', 'chatUrl'),
    'chatAnalysis': ('chatAnalysisId', 'chatAnalysisUrl'),
    'transcriptAnalysis': ('transcriptAnalysisId', 'transcriptAnalysisUrl'),
    'transcriptWord': ('transcriptWordId', 'transcriptWordUrl')
}

async def upload_file_to_convex(contents: str) -> SavedFileData:
    """Renamed from upload_file to avoid naming conflict"""
    logger.debug("Attempting to upload file to Convex")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://curious-lark-519.convex.site/upload-file",
                headers={
                    "Authorization": "Basic e1a36cfcdf58e3282d4aee7a22adebbe0d2c349f",
                    "Content-Type": "text/plain",
                },
                data=contents
            ) as response:
                if not response.ok:
                    logger.error(f"Upload failed with status {response.status}: {await response.text()}")
                    raise Exception(f"Upload failed with status {response.status}")
                return await response.json()
        except Exception as e:
            logger.error(f"Error in upload_file_to_convex: {str(e)}")
            raise

async def update_video(contents: UpdateVideo) -> bool:
    logger.debug(f"Attempting to update video with data: {contents}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://curious-lark-519.convex.site/update-video",
                headers={
                    "Authorization": "Basic e1a36cfcdf58e3282d4aee7a22adebbe0d2c349f",
                    "Content-Type": "application/json",
                },
                data=json.dumps(contents)
            ) as response:
                response_text = await response.text()
                if not response.ok:
                    logger.error(f"Update video failed with status {response.status}")
                    logger.error(f"Response body: {response_text}")
                    logger.error(f"Request data: {json.dumps(contents, indent=2)}")
                    return False
                logger.debug(f"Update successful. Response: {response_text}")
                return True
        except Exception as e:
            logger.error(f"Error in update_video: {str(e)}")
            return False

async def send_file(video_id: str, contents: str, file_type: FileType) -> bool:
    logger.debug(f"Sending file of type {file_type} for video {video_id}")
    if not video_id:
        raise ValueError("video_id cannot be empty")
    if not contents:
        raise ValueError("contents cannot be empty")
    if file_type not in type_to_fields:
        raise ValueError(f"Invalid file_type: {file_type}")    
    try:
        upload_result = await upload_file_to_convex(contents)
        logger.debug(f"Upload successful, got result: {upload_result}")
        
        update_data: UpdateVideo = {"id": video_id}    
        id_field, url_field = type_to_fields[file_type]        
        update_data[id_field] = upload_result['storageId']
        update_data[url_field] = upload_result['url']        
        
        logger.debug(f"Preparing to update video with data: {json.dumps(update_data, indent=2)}")
        success = await update_video(update_data)
        if success:
            logger.debug(f"Successfully updated video with {file_type}")
        else:
            logger.error(f"Failed to update video with {file_type}")
        return success
    except Exception as e:
        logger.error(f"Error in send_file: {str(e)}", exc_info=True)
        return False

# Define the directories
OUTPUTS_DIR = 'outputs'
DATA_DIR = 'data'

async def get_file_type(filename: str, video_id: str) -> str | None:
    """Determine the file type based on filename pattern"""
    if filename == f"{video_id}_chat.csv":
        return 'chat'
    elif filename == f"{video_id}_chat_analysis.csv":
        return 'chatAnalysis'
    elif filename == f"audio_{video_id}_waveform.json":
        return 'audiowave'
    elif filename == f"audio_{video_id}_paragraphs.csv":
        return 'transcript'
    elif filename == f"audio_{video_id}_words.csv":
        return 'transcriptWord'
    return None

async def upload_files(video_id: str) -> None:
    # Upload chat file from data directory
    chat_filename = f"{video_id}_chat.csv"
    chat_path = os.path.join(DATA_DIR, chat_filename)
    
    logger.debug(f"Checking for chat file at: {chat_path}")
    if os.path.exists(chat_path):
        try:
            with open(chat_path, 'r', encoding='utf-8') as file:
                contents = file.read()
                success = await send_file(video_id, contents, 'chat')
                if success:
                    logger.info(f"Successfully uploaded {chat_filename}")
                else:
                    logger.error(f"Failed to upload {chat_filename}")
        except Exception as e:
            logger.error(f"Error reading {chat_path}: {str(e)}")
    else:
        logger.warning(f"Chat file not found at {chat_path}")

    # Upload files from outputs directory
    logger.debug(f"Checking outputs directory: {OUTPUTS_DIR}")
    for filename in os.listdir(OUTPUTS_DIR):
        file_type = await get_file_type(filename, video_id)
        if file_type:
            file_path = os.path.join(OUTPUTS_DIR, filename)
            logger.debug(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    contents = file.read()
                    success = await send_file(video_id, contents, file_type)
                    if success:
                        logger.info(f"Successfully uploaded {filename}")
                    else:
                        logger.error(f"Failed to upload {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload video files to Convex')
    parser.add_argument('video_id', type=str, help='The ID of the video to process')
    
    args = parser.parse_args()
    
    if not args.video_id:
        print("Error: video_id is required")
        parser.print_help()
        exit(1)
        
    print(f"Processing files for video ID: {args.video_id}")
    asyncio.run(upload_files(args.video_id))