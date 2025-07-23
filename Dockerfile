FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES=0

# Copy and install requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Create optimized temp directory
RUN mkdir -p /tmp/geopixel_cache && chmod 777 /tmp/geopixel_cache

# Expose the port for the API server
EXPOSE 5000

# Revert to Flask dev server for better GPU performance with this workload
CMD ["python", "api_server.py"]