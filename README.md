# PyTorch CUDA Docker Container

This repository contains a Docker setup for PyTorch with CUDA support, based on the `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` image.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)
- [Docker Compose](https://docs.docker.com/compose/install/) (optional)

## Features

- PyTorch 2.3.1
- CUDA 12.1
- cuDNN 8
- Basic development utilities (git, curl, wget)

## Usage Examples

### Running GeoPixel API Server

This repository is configured to run the GeoPixel model as an API server, which allows you to analyze images and get segmentation masks without interactive input. This is particularly useful for running on platforms like RunPod.

To run the GeoPixel API server:

1. The API server will be accessible at http://localhost:5000 with the following endpoints:
   - `GET /health` - Check if the model is loaded and ready
   - `POST /process` - Process an image with a query

4. Example of using the API with curl:
   ```bash
   curl -X POST -F "image=@path/to/your/image.jpg" -F "query=Describe this image" http://localhost:5000/process
   ```

5. When you're done, stop the container:
   ```bash
   docker-compose down
   ```

### Running on RunPod

To run this container on RunPod:

1. Create a new pod with the GPU of your choice
2. Use a custom Docker image by specifying the GitHub repository URL
3. The API server will automatically start when the pod is ready
4. You can access the API through the RunPod HTTP endpoints feature

Example RunPod HTTP endpoint configuration:
- Port: 5000
- Path: /
- Target Port: 5000

This will allow you to send HTTP requests to the GeoPixel API from anywhere.

### Testing Your PyTorch Setup

You can verify your PyTorch CUDA setup with these commands:

### Basic PyTorch Commands

Once inside the container, you can run Python with PyTorch:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

To verify GPU access:

```bash
python -c "import torch; print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## Customizing

To install additional Python packages, uncomment and modify the requirements.txt section in the Dockerfile:

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

Then create a `requirements.txt` file with your desired packages.