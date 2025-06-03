import requests
import json
import base64
import os
import sys
import time
from PIL import Image
from io import BytesIO

def check_health(api_url):
    """
    Check if the API server is healthy
    
    Args:
        api_url (str): Base URL of the API
        
    Returns:
        bool: True if healthy, False otherwise
    """
    health_url = api_url.rstrip('/process') + '/health'
    try:
        print(f"Checking API health at {health_url}")
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"Health check result: {result}")
            return True
        else:
            print(f"Health check failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error checking API health: {str(e)}")
        return False

def process_image(image_path, query, api_url, max_retries=3):
    """
    Send an image to the GeoPixel API and get the description
    
    Args:
        image_path (str): Path to the image file
        query (str): Query to send with the image
        api_url (str): URL of the API endpoint
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        dict: The API response
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Check image file size
    file_size = os.path.getsize(image_path)
    print(f"Image file size: {file_size} bytes")
    if file_size == 0:
        print("Error: Image file is empty")
        return None
    
    # Try to open the image to verify it's valid
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image dimensions: {width}x{height}")
            print(f"Image format: {img.format}")
    except Exception as e:
        print(f"Warning: Could not verify image with PIL: {str(e)}")
    
    # Prepare the request
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print(f"Sending request to {api_url}")
            print(f"Image: {image_path}")
            print(f"Query: {query}")
            
            # Open the file for each attempt to avoid closed file errors
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {'query': query}
                
                # Send the POST request with a longer timeout
                response = requests.post(api_url, files=files, data=data, timeout=300)
            
            # Check if the request was successful
            if response.status_code == 200:
                try:
                    result = response.json()
                    return result
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse JSON response: {str(e)}")
                    print(f"Response text: {response.text[:500]}...")
            else:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                
                # If server error, wait before retrying
                if response.status_code >= 500:
                    if attempt < max_retries - 1:
                        retry_delay = (attempt + 1) * 5  # Increasing delay
                        print(f"Server error, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    continue
                else:
                    # Client error, don't retry
                    break
        except requests.exceptions.Timeout:
            print("Error: Request timed out")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)
            continue
        except Exception as e:
            print(f"Error sending request: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)
            continue
    
    return None

def save_masked_image(base64_image, output_path):
    """
    Save a base64-encoded image to a file
    
    Args:
        base64_image (str): Base64-encoded image data
        output_path (str): Path to save the image
    """
    try:
        # Decode the base64 image
        image_data = base64.b64decode(base64_image)
        
        # Create an image from the binary data
        image = Image.open(BytesIO(image_data))
        
        # Save the image
        image.save(output_path)
        print(f"Masked image saved to: {output_path}")
        
        return True
    except Exception as e:
        print(f"Error saving masked image: {str(e)}")
        return False

def main():
    # Configuration
    api_base_url = "https://bpds0xic8d0b7g-5000.proxy.runpod.net/"
    api_process_url = f"{api_base_url}/process"
    image_path = "GeoPixel/images/example1.png"
    query = "Please give me a segmentation mask for grey car."
    output_dir = "output"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        query = sys.argv[2]
    
    print(f"GeoPixel API Client")
    print(f"==================")
    print(f"API URL: {api_base_url}")
    print(f"Image: {image_path}")
    print(f"Query: {query}")
    print(f"==================\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check API health first
    if not check_health(api_base_url):
        print("\nWARNING: API health check failed. The server might not be ready.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting.")
            return
    
    # Process the image
    print("\nProcessing image...")
    result = process_image(image_path, query, api_process_url)
    
    if result:
        # Check for errors in the result
        if "error" in result:
            print(f"\nError from API: {result['error']}")
            return
        
        # Print the text response
        print("\n--- Text Response ---")
        print(result.get("text_response", "No text response"))
        
        # Print any error messages
        for key in result:
            if key.endswith('_error'):
                print(f"\n--- Error: {key} ---")
                print(result[key])
        
        # Print the reconstructed text if available
        if "reconstructed_text" in result:
            print("\n--- Reconstructed Text ---")
            print(result["reconstructed_text"])
        
        # Save the masked image if available
        if "masked_image_base64" in result:
            print("\n--- Masked Image ---")
            output_path = os.path.join(output_dir, f"masked_{os.path.basename(image_path)}")
            save_masked_image(result["masked_image_base64"], output_path)
        elif "masked_image_path" in result:
            print(f"\nMasked image saved on server at: {result['masked_image_path']}")
        
        print("\nProcessing complete!")
    else:
        print("\nFailed to process the image. Please check the API server logs for more details.")

if __name__ == "__main__":
    main()