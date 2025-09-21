# src/pipeline_server.py

import logging
from pathlib import Path
import tempfile

import cv2
from flask import Flask, request, jsonify

# --- Import our core components ---
from .pipeline import ReframerPipeline
from .services.output_formatter_service import OutputFormatterService
from .utils.caching import NoOpCacheManager
from .utils.config import load_config
from .utils.video_downloader import download_video

# --- Initialize Flask App and Load Base Config ---
app = Flask(__name__)
logger = logging.getLogger(__name__)

try:
    BASE_CONFIG = load_config()
except (FileNotFoundError, TypeError) as e:
    logger.critical(f"Failed to load base configuration: {e}. Server cannot start.")
    BASE_CONFIG = None


@app.route("/process", methods=["POST"])
def process_video():
    """
    Main API endpoint to process a video for intelligent reframing.

    This endpoint orchestrates the entire pipeline:
    1.  Validates the incoming JSON request.
    2.  Merges client-provided configuration overrides with the server's base config.
    3.  Downloads the video from the provided URL to a temporary location.
    4.  Instantiates and runs the core `ReframerPipeline`.
    5.  Formats the pipeline's internal results into a client-friendly JSON response.
    """
    if BASE_CONFIG is None:
        return jsonify({"error": "Server is not configured correctly."}), 500

    # 1. Get and validate the client's request data
    client_data = request.get_json()
    if not client_data or "video_url" not in client_data:
        return jsonify({"error": "Request must include 'video_url' in JSON body."}), 400
    
    video_url = client_data["video_url"]
    logger.info(f"Received request to process video: {video_url}")

    # 2. Handle Configuration (Default + Client Overrides)
    from copy import deepcopy
    config = deepcopy(BASE_CONFIG)
    if "scouting" in client_data:
        config["scouting"].update(client_data["scouting"])
    if "output" in client_data:
        config["output"].update(client_data["output"])

    # Use a temporary directory that is automatically cleaned up after the request.
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 3. Download the video
            local_video_path = download_video(video_url, Path(temp_dir))
            if local_video_path is None:
                return jsonify({"error": "Failed to download video from URL."}), 400

            # 4. Run the core pipeline
            # The deployed server uses the NoOpCacheManager to disable caching.
            cache_manager = NoOpCacheManager()
            pipeline = ReframerPipeline(
                media_path=local_video_path,
                config=config,
                cache_manager=cache_manager
            )
            internal_results = pipeline.run()

            # 5. Format the output for the client
            # The formatter requires video metadata to correctly scale coordinates.
            cap = cv2.VideoCapture(str(local_video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            formatter = OutputFormatterService(fps=fps, video_width=width, video_height=height)
            final_results = formatter.run(internal_results)
            
            return jsonify(final_results)

        except Exception as e:
            logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
            return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    # This block allows the Flask server to be run directly for local testing.
    # In a production environment, a WSGI server like Gunicorn is used.
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5001, debug=True)
