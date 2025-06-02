# GeoPixel Docker

This repository contains Docker configuration for running [GeoPixel](https://github.com/MBZUAI/GeoPixel), a vision-language model for geospatial image understanding with segmentation capabilities.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/geopixel_docker.git
   cd geopixel_docker
   ```

2. Create directories for input images and output:
   ```bash
   mkdir -p input_images output
   ```

3. Build the Docker image:
   ```bash
   docker-compose build
   ```

## Usage

### Quick Start

#### Windows
Simply double-click the `run_geopixel.bat` file to:
- Create necessary directories
- Build the Docker image (if needed)
- Start the GeoPixel container

#### Linux/macOS
Run the shell script:
```bash
./run_geopixel.sh
```

### Manual Usage

1. Place your input images in the `input_images` directory.

2. Run the container:
   ```bash
   docker-compose run geopixel
   ```

3. When prompted:
   - Enter your query (e.g., "Can you provide a thorough description of this image?")
   - Enter the image path (e.g., `/app/input_images/your_image.jpg`)

4. The segmentation results will be saved in the `output` directory.

## Example

```bash
# Start the container
docker-compose run geopixel

# When prompted:
Please input your query: Can you provide a thorough description of this image? Please output with interleaved segmentation masks for the corresponding phrases.
Please input the image path: /app/input_images/satellite_image.jpg
```

## Notes

- The model will be downloaded from Hugging Face the first time you run the container.
- The default model is `MBZUAI/GeoPixel-7B`, which requires approximately 14GB of VRAM.
- Processed images with segmentation masks will be saved in the `output` directory.

## Custom Arguments

You can pass custom arguments to the GeoPixel chat.py script by modifying the command in docker-compose.yml:

```yaml
command: ["--version", "MBZUAI/GeoPixel-7B", "--vis_save_path", "/app/GeoPixel/vis_output"]
```

Available arguments:
- `--version`: Model version (default: "MBZUAI/GeoPixel-7B")
- `--vis_save_path`: Path to save visualization outputs (default: "./vis_output")