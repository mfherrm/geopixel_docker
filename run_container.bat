@echo off
echo Building and starting Docker container...
docker-compose up -d

echo GeoPixel API server is now running!
echo The API is accessible at http://localhost:5000
echo.
echo You can use the following endpoints:
echo - GET /health - Check if the model is loaded and ready
echo - POST /process - Process an image with a query
echo.
echo Example curl command to process an image:
echo curl -X POST -F "image=@path/to/your/image.jpg" -F "query=Describe this image" http://localhost:5000/process
echo.
echo When done, you can stop the container with:
echo docker-compose down