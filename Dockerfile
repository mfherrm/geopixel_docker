FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY GeoPixel/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy GeoPixel code
COPY GeoPixel/ /app/GeoPixel/

# Create directory for output
RUN mkdir -p /app/GeoPixel/vis_output

# Set the entrypoint
ENTRYPOINT ["python", "GeoPixel/chat.py"]