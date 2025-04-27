from typing import Optional, TypedDict, Union, List, Tuple
import aiohttp
import json
import os
import asyncio
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

class FileObject(TypedDict):
    name: str
    storageId: str
    url: str

class UpdateVideo(TypedDict, total=False):
    id: str
    status: str
    files: List[FileObject]

async def upload_file_to_convex(contents: str) -> SavedFileData:
    logger.debug("Attempting to upload file to Convex")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://laudable-horse-446.convex.site/upload-file",
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
                "https://laudable-horse-446.convex.site/update-video",
                headers={
                    "Authorization": "Basic e1a36cfcdf58e3282d4aee7a22adebbe0d2c349f",
                    "Content-Type": "application/json",
                },
                data=json.dumps(contents)
            ) as response:
                if not response.ok:
                    logger.error(f"Update video failed with status {response.status}")
                return response.ok
        except Exception as e:
            logger.error(f"Error in update_video: {str(e)}")
            return False

async def send_files(video_id: str, files: List[Tuple[str, str]]) -> bool:
    success = True
    for contents, filename in files:
        logger.debug(f"Sending file {filename} for video {video_id}")
        try:
            upload_result = await upload_file_to_convex(contents)
            logger.debug(f"Upload successful, got result: {upload_result}")

            update_data: UpdateVideo = {"id": video_id, "status": "uploaded", "files": [{"name": filename, "storageId": upload_result['storageId'], "url": upload_result['url']}]}

            logger.debug(f"Preparing to update video with data: {json.dumps(update_data, indent=2)}")
            success = await update_video(update_data)
            if success:
                logger.debug(f"Successfully updated video with {filename}")
            else:
                logger.error(f"Failed to update video with {filename}")
        except Exception as e:
            logger.error(f"Error in send_files: {str(e)}", exc_info=True)
            success = False
    return success

# Define the directories - use /tmp for Cloud Functions
OUTPUTS_DIR = os.environ.get('OUTPUTS_DIR', '/tmp/outputs')
DATA_DIR = os.environ.get('DATA_DIR', '/tmp/data')

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
    files_to_upload = []

    # Check chat file from data directory
    chat_filename = f"{video_id}_chat.csv"
    chat_path = os.path.join(DATA_DIR, chat_filename)

    logger.debug(f"Checking for chat file at: {chat_path}")
    if os.path.exists(chat_path):
        try:
            with open(chat_path, 'r', encoding='utf-8') as file:
                contents = file.read()
                files_to_upload.append((contents, chat_filename))
        except Exception as e:
            logger.error(f"Error reading {chat_path}: {str(e)}")

    # Check files from outputs directory
    logger.debug(f"Checking outputs directory: {OUTPUTS_DIR}")
    for filename in os.listdir(OUTPUTS_DIR):
        file_type = await get_file_type(filename, video_id)
        if file_type:
            file_path = os.path.join(OUTPUTS_DIR, filename)
            logger.debug(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    contents = file.read()
                    files_to_upload.append((contents, filename))
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")

    if files_to_upload:
        success = await send_files(video_id, files_to_upload)
        if success:
            logger.info(f"Successfully uploaded all files for video {video_id}")
        else:
            logger.error(f"Failed to upload files for video {video_id}")

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