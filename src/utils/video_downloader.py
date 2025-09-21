# src/utils/video_downloader.py

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def download_video(url: str, destination_folder: Path) -> Optional[Path]:
    """
    Downloads a video from a URL to a specified local folder using a streaming request.

    This function is designed to handle potentially large files efficiently by not
    loading the entire content into memory at once. It includes robust error
    handling for network issues and file I/O problems.

    Args:
        url: The URL of the video file to download.
        destination_folder: The local directory where the file will be saved.

    Returns:
        The Path object of the downloaded file on success, or None on failure.
    """
    try:
        # Use a streaming request to handle large files without high memory usage.
        # A timeout is set to prevent the request from hanging indefinitely.
        logger.info(f"Downloading video from {url}...")
        with requests.get(url, stream=True, timeout=30) as response:
            # Raise an HTTPError for bad responses (4xx or 5xx).
            response.raise_for_status()

            # Safely extract a filename from the URL's path component.
            parsed_path = urlparse(url).path
            filename = Path(parsed_path).name
            if not filename:
                filename = "downloaded_video.tmp"  # Provide a fallback filename.

            destination_path = destination_folder / filename

            # Write the content to the file in chunks to maintain low memory usage.
            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded video to {destination_path}")
            return destination_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download video from URL '{url}'. Error: {e}")
        return None
    except IOError as e:
        logger.error(f"Failed to write video to disk at '{destination_folder}'. Error: {e}")
        return None
