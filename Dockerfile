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

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Copy and install requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .
# Expose the port for the API server
EXPOSE 5000

# Command to run when container starts
# Run the API server by default
CMD ["python", "api_server.py"]