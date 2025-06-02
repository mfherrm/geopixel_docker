#!/bin/bash

# Create directories if they don't exist
mkdir -p input_images output

# Build the Docker image if it doesn't exist
if [[ "$(docker images -q geopixel:latest 2> /dev/null)" == "" ]]; then
  echo "Building GeoPixel Docker image..."
  docker-compose build
fi

# Run the container
echo "Starting GeoPixel container..."
docker-compose run geopixel

echo "GeoPixel container has been stopped."
echo "Check the 'output' directory for visualization results."