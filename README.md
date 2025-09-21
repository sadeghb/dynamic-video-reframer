# Dynamic Video Reframing Pipeline

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-007F7F?style=for-the-badge&logo=google)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)

## Overview

This repository contains an intelligent computer vision pipeline that acts as a "virtual camera operator". It was designed to solve the tedious but critical problem of reformatting landscape (16:9) video for vertical (9:16) platforms by automatically generating the necessary tracking data to keep on-screen subjects perfectly framed.

The system ingests a landscape video and outputs a structured JSON file containing precise, frame-by-frame coordinate data of key subjects, ready to be consumed by a downstream rendering service.

---

## ✨ Key Features

* **Custom Multi-Object Tracking Algorithm**: The core of the system is a custom, heuristic-based algorithm that links sparse detections into persistent tracks. Its key innovation is a dynamic, size-relative distance threshold that robustly tracks subjects as they move closer to or farther from the camera.
* **Dual-Output Tracks**: For each track, the pipeline generates both a static "Fixed Box" (the union of all detections) and a dense, frame-by-frame "Dynamic Track" using linear interpolation and extrapolation for smooth camera motion.
* **Multi-Stage CV Pipeline**: A "divide and conquer" architecture that uses industry-standard libraries like `PySceneDetect` for scene segmentation and Google's `MediaPipe` for high-performance object detection.
* **Multiple Interfaces**: The pipeline can be run as a REST API (via Flask), a command-line interface (CLI), or tested with a `pytest` suite.
* **Flexible Dual-Deployment Strategy**: The application is designed for two deployment models: as a standard, containerized web service using Gunicorn, or as a serverless function on AWS Lambda.
* **Integrated Caching System**: Automatically caches the results of computationally expensive steps to the local filesystem, dramatically speeding up development and iteration.

---

## ⚙️ The Computer Vision Pipeline

The pipeline is architected as a sequence of modular, specialized stages:

1.  **Scene Segmentation**: The video is first processed with `PySceneDetect` to identify all camera cuts. This ensures that subject tracking is logically constrained to a single, continuous scene.
2.  **Sparse Keyframe Sampling**: To avoid the immense cost of processing every frame, the pipeline intelligently samples a configurable number of keyframes within each scene for analysis.
3.  **Multi-Object Detection**: Google's `MediaPipe` framework is used on the sampled keyframes to detect all instances of faces and persons.
4.  **Intelligent Multi-Object Tracking**: The custom-built tracking algorithm links the sparse detections across keyframes into persistent tracks and filters out short-lived, "noisy" tracks.
5.  **Track Generation & Output**: The final stage processes the persistent tracks to generate the structured JSON output with both "fixed" and "dynamic" track data.

---

## 🛠️ Tech Stack

* **Core Libraries**: Python 3.12, OpenCV, MediaPipe, SciPy, NumPy
* **Web Service**: Flask, Gunicorn
* **Deployment**: Docker
* **Tooling**: Pytest, Pydantic, PyYAML

---

## 🚀 Usage & Local Development

This project is structured as an installable Python package.

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sadeghb/dynamic-video-reframer.git](https://github.com/sadeghb/dynamic-video-reframer.git)
    cd dynamic-video-reframer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode:**
    This crucial step makes the `src` package importable throughout the project.
    ```bash
    pip install -e .
    ```

### Running the CLI (`main.py`)

The `main.py` script is the entry point for local pipeline runs.

**Run with a local file:**
```bash
python main.py --path path/to/your/video.mp4
````

**Run with a URL:**

```bash
python main.py --url [http://example.com/video.mp4](http://example.com/video.mp4)
```

### Running the Web Service (`pipeline_server.py`)

**1. Start the local server:**

```bash
# The server will start on [http://0.0.0.0:5001](http://0.0.0.0:5001)
python src/pipeline_server.py
```

**2. Send an API Request (using `curl`):**
Open a new terminal to send a request to your local server.

```bash
curl -X POST \
  [http://127.0.0.1:5001/process](http://127.0.0.1:5001/process) \
  -H 'Content-Type: application/json' \
  -d '{
    "video_url": "[http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4)"
  }'
```

-----

## 🐳 Deployment

This application is designed with a flexible, dual-deployment strategy.

### 1\. Containerized Web Service

The included `Dockerfile` packages the application with a Gunicorn production server, ready for deployment on any container hosting platform (e.g., AWS ECS, Google Cloud Run).

**Build the image:**

```bash
docker build -t reframer-pipeline .
```

**Run the container:**

```bash
docker run -p 8080:8080 reframer-pipeline
```

### 2\. Serverless Function

The `lambda_function.py` file serves as an adapter, allowing the same Flask application to be deployed as a cost-effective, auto-scaling serverless function on AWS Lambda.