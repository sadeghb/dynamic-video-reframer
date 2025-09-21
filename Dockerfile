# Dockerfile

# --- Stage 1: Build Stage ---
# Use a full Python image to build our dependencies
FROM python:3.12-slim as builder

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy only the files needed to install dependencies
COPY requirements.txt pyproject.toml ./

# Install dependencies
# Using --no-cache-dir to keep the layer small
RUN pip install --no-cache-dir -r requirements.txt

# Install our project in editable mode
RUN pip install -e .


# --- Stage 2: Final Stage ---
# Use a minimal base image for the final container
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy system dependencies from the builder stage
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy our application source code
COPY src/ ./src
COPY config/ ./config
COPY models/ ./models

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the application using Gunicorn
# This is the production-ready command to start the web server.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "src.pipeline_server:app"]