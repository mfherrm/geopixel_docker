@echo off
echo Creating directories if they don't exist...
if not exist input_images mkdir input_images
if not exist output mkdir output

echo Checking if Docker image exists...
docker images -q geopixel:latest > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Building GeoPixel Docker image...
  docker-compose build
)

echo Starting GeoPixel container...
docker-compose run geopixel

echo GeoPixel container has been stopped.
echo Check the 'output' directory for visualization results.
pause