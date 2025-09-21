# main.py

import argparse
import json
import logging
from collections.abc import MutableMapping
from pathlib import Path
import tempfile

import cv2

# --- Import our core components ---
from src.pipeline import ReframerPipeline
from src.services.output_formatter_service import OutputFormatterService
from src.utils.caching import FileSystemCacheManager
from src.utils.config import load_config
from src.utils.video_downloader import download_video

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def deep_merge(d: dict, u: dict) -> dict:
    """
    Performs a deep merge of dictionary 'u' into dictionary 'd'.
    Modifies 'd' in place.
    """
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            d[k] = deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main():
    """
    Main command-line execution function for the video reframing pipeline.
    
    This script provides a command-line interface to run the entire pipeline
    on a single video, either from a local path or a remote URL.
    """
    # 1. --- Argument Parsing ---
    # Sets up the CLI to accept a video source and optional config overrides.
    parser = argparse.ArgumentParser(description="Run the full video reframing pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="URL of the video file to process.")
    group.add_argument("--path", type=str, help="Local path to the video file to process.")
    parser.add_argument("--config", type=str, help="A JSON string to override default config values.")
    args = parser.parse_args()

    # 2. --- Configuration Handling ---
    # Loads the base config and merges any command-line overrides for this run.
    try:
        config = load_config()
        if args.config:
            logger.info("Applying command-line config overrides...")
            overrides = json.loads(args.config)
            config = deep_merge(config, overrides)
    except (FileNotFoundError, TypeError, json.JSONDecodeError) as e:
        logger.critical(f"Failed to handle configuration: {e}")
        return

    # 3. --- Input Video Handling ---
    # Uses a temporary directory to store downloaded videos, ensuring cleanup.
    temp_dir = tempfile.TemporaryDirectory()
    try:
        if args.url:
            logger.info(f"Downloading video from URL: {args.url}")
            local_video_path = download_video(args.url, Path(temp_dir.name))
        else:
            local_video_path = Path(args.path)

        if not local_video_path or not local_video_path.exists():
            logger.critical("Input video file could not be found or downloaded.")
            return

        # 4. --- Pipeline Execution ---
        # For local runs, we always use the FileSystemCacheManager to speed up iteration.
        logger.info("Initializing pipeline with FileSystemCacheManager...")
        cache_manager = FileSystemCacheManager()
        pipeline = ReframerPipeline(
            media_path=local_video_path,
            config=config,
            cache_manager=cache_manager
        )
        internal_results = pipeline.run()
        logger.info("Pipeline execution complete.")

        # 5. --- Final Output Formatting ---
        # Gathers video metadata and formats the internal results for the client.
        cap = cv2.VideoCapture(str(pipeline.internal_media_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        formatter = OutputFormatterService(fps=fps, video_width=width, video_height=height)
        final_results = formatter.run(internal_results)

        # 6. --- Print Results ---
        # Prints the final, formatted JSON to the console.
        print(json.dumps(final_results, indent=2))

    finally:
        # This block ensures the temporary directory is always cleaned up.
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
