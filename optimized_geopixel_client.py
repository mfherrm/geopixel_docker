import requests
import json
import base64
import os
import sys
import time
import numpy as np
from PIL import Image
from io import BytesIO
import concurrent.futures
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import threading

class OptimizedGeoPixelClient:
    """Optimized client for GeoPixel API with batch processing and connection pooling"""
    
    def __init__(self, api_base_url, max_retries=3, pool_connections=10, pool_maxsize=20, max_workers=6):
        self.api_base_url = api_base_url.rstrip('/')
        self.api_process_url = f"{self.api_base_url}/process"  # Unified endpoint for all requests
        self.health_url = f"{self.api_base_url}/health"
        self.max_workers = max_workers
        
        # Setup session with connection pooling and retries
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST"],
            backoff_factor=1
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set session timeout
        self.session_timeout = 300  # 5 minutes
        
        print(f"Initialized OptimizedGeoPixelClient with connection pooling:")
        print(f"  - API URL: {api_base_url}")
        print(f"  - Pool connections: {pool_connections}")
        print(f"  - Pool max size: {pool_maxsize}")
        print(f"  - Max workers: {max_workers}")
    
    def check_health(self):
        """Check if the API server is healthy"""
        try:
            print(f"Checking API health at {self.health_url}")
            response = self.session.get(self.health_url, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Health check result: {result}")
                return True
            else:
                print(f"❌ Health check failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error checking API health: {str(e)}")
            return False
    
    def process_single_image(self, image_path, query):
        """Process a single image (for compatibility)"""
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return None, None
        
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print("❌ Error: Image file is empty")
            return None, None
        
        try:
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {'query': query}
                
                response = self.session.post(
                    self.api_process_url, 
                    files=files, 
                    data=data, 
                    timeout=self.session_timeout
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Process prediction masks if available
                pred_masks = None
                if "pred_masks_base64" in result:
                    try:
                        masks_encoded = result["pred_masks_base64"]
                        masks = []
                        for mask_b64 in masks_encoded:
                            mask_data = base64.b64decode(mask_b64)
                            mask_img = Image.open(BytesIO(mask_data))
                            mask_np = np.array(mask_img) > 0
                            masks.append(mask_np)
                        
                        if masks:
                            pred_masks = np.stack(masks)
                            print(f"✅ Successfully decoded {len(masks)} prediction masks")
                    except Exception as e:
                        print(f"⚠️  Error decoding prediction masks: {str(e)}")
                
                return result, pred_masks
            else:
                print(f"❌ API request failed with status code {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                return None, None
                
        except Exception as e:
            print(f"❌ Error sending request: {str(e)}")
            return None, None
    
    def process_tiles_batch(self, tiles_data):
        """Process multiple tiles in a single optimized batch request"""
        if not tiles_data:
            print("❌ Error: No tiles data provided")
            return None
        
        print(f"🚀 Processing {len(tiles_data)} tiles in batch mode")
        
        # Prepare tiles for JSON payload
        processed_tiles = []
        for i, tile in enumerate(tiles_data):
            try:
                image_path = tile.get('image_path', '')
                query = tile.get('query', '')
                
                if not image_path or not query:
                    print(f"⚠️  Skipping tile {i}: Missing image_path or query")
                    continue
                
                if not os.path.exists(image_path):
                    print(f"⚠️  Skipping tile {i}: Image file not found: {image_path}")
                    continue
                
                # Encode image as base64
                try:
                    with open(image_path, 'rb') as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    processed_tiles.append({
                        'query': query,
                        'image_base64': image_data,
                        'tile_id': tile.get('tile_id', f'tile_{i}')
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error encoding tile {i}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"⚠️  Error processing tile {i}: {str(e)}")
                continue
        
        if not processed_tiles:
            print("❌ Error: No valid tiles to process")
            return None
        
        # Send batch request to unified /process endpoint
        try:
            payload = {'tiles': processed_tiles}
            
            start_time = time.time()
            response = self.session.post(
                self.api_process_url,  # Use unified endpoint
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.session_timeout * 2  # Double timeout for batch
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch processing completed in {processing_time:.2f} seconds")
                
                # Handle unified API response format
                if 'results' in result:
                    successful_tiles = len([r for r in result['results'] if 'error' not in r])
                    total_tiles = result.get('batch_size', len(result['results']))
                    print(f"   Processed {successful_tiles}/{total_tiles} tiles")
                    
                    # Convert to expected format for backward compatibility
                    return {
                        "tile_results": result['results'],
                        "total_tiles": total_tiles,
                        "processed_tiles": successful_tiles,
                        "processing_time": f"{processing_time:.2f}s",
                        "processing_info": result.get('processing_info', {})
                    }
                else:
                    # Handle single tile response or old format
                    return result
            else:
                print(f"❌ Batch request failed with status code {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                return None
                
        except Exception as e:
            print(f"❌ Error in batch processing: {str(e)}")
            return None
    
    def process_tiles_parallel(self, tiles_data, use_batch=True):
        """Process tiles with optimal strategy selection"""
        if not tiles_data:
            return None
        
        num_tiles = len(tiles_data)
        print(f"🎯 Processing {num_tiles} tiles...")
        
        # Strategy selection based on tile count
        if use_batch and num_tiles > 1:
            print("📦 Using batch processing strategy")
            return self.process_tiles_batch(tiles_data)
        
        elif num_tiles > self.max_workers:
            print(f"⚡ Using parallel processing with {self.max_workers} workers")
            return self._process_tiles_threaded(tiles_data)
        
        else:
            print("🔄 Using sequential processing")
            return self._process_tiles_sequential(tiles_data)
    
    def _process_tiles_sequential(self, tiles_data):
        """Process tiles sequentially"""
        results = []
        start_time = time.time()
        
        for i, tile in enumerate(tiles_data):
            print(f"Processing tile {i+1}/{len(tiles_data)}")
            result, pred_masks = self.process_single_image(
                tile.get('image_path', ''), 
                tile.get('query', '')
            )
            
            if result:
                result['tile_id'] = tile.get('tile_id', f'tile_{i}')
                if pred_masks is not None:
                    result['pred_masks_shape'] = pred_masks.shape
                results.append(result)
            else:
                results.append({"error": f"Failed to process tile {i}"})
        
        processing_time = time.time() - start_time
        print(f"✅ Sequential processing completed in {processing_time:.2f} seconds")
        
        return {
            "tile_results": results,
            "total_tiles": len(tiles_data),
            "processed_tiles": len([r for r in results if "error" not in r]),
            "processing_time": f"{processing_time:.2f}s"
        }
    
    def _process_tiles_threaded(self, tiles_data):
        """Process tiles using thread pool"""
        results = []
        start_time = time.time()
        
        def process_tile(tile_data):
            i, tile = tile_data
            result, pred_masks = self.process_single_image(
                tile.get('image_path', ''), 
                tile.get('query', '')
            )
            
            if result:
                result['tile_id'] = tile.get('tile_id', f'tile_{i}')
                if pred_masks is not None:
                    result['pred_masks_shape'] = pred_masks.shape
                return result
            else:
                return {"error": f"Failed to process tile {i}"}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_tile = {
                executor.submit(process_tile, (i, tile)): i 
                for i, tile in enumerate(tiles_data)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_tile):
                tile_idx = future_to_tile[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ Completed tile {tile_idx + 1}/{len(tiles_data)}")
                except Exception as exc:
                    print(f"❌ Tile {tile_idx} generated an exception: {exc}")
                    results.append({"error": f"Exception in tile {tile_idx}: {str(exc)}"})
        
        processing_time = time.time() - start_time
        print(f"✅ Parallel processing completed in {processing_time:.2f} seconds")
        
        return {
            "tile_results": results,
            "total_tiles": len(tiles_data),
            "processed_tiles": len([r for r in results if "error" not in r]),
            "processing_time": f"{processing_time:.2f}s"
        }
    
    def save_masked_image(self, base64_image, output_path):
        """Save a base64-encoded image to a file"""
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            image.save(output_path)
            print(f"✅ Masked image saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving masked image: {str(e)}")
            return False
    
    def close(self):
        """Close the session"""
        self.session.close()
        print("🔒 Session closed")

def demo_optimized_client():
    """Demo function showing how to use the optimized client"""
    # Configuration
    api_base_url = "https://bpds0xic8d0b7g-5000.proxy.runpod.net/"
    output_dir = "output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize optimized client
    client = OptimizedGeoPixelClient(
        api_base_url=api_base_url,
        pool_connections=10,
        pool_maxsize=20,
        max_workers=6
    )
    
    try:
        # Check API health
        if not client.check_health():
            print("⚠️  API health check failed, but continuing...")
        
        # Example 1: Single image processing
        print("\n" + "="*50)
        print("SINGLE IMAGE PROCESSING")
        print("="*50)
        
        single_image_path = "GeoPixel/images/example1.png"
        single_query = "Please give me a segmentation mask for grey car."
        
        if os.path.exists(single_image_path):
            result, pred_masks = client.process_single_image(single_image_path, single_query)
            
            if result:
                print(f"✅ Single processing result: {result.get('text_response', 'No response')[:100]}...")
                if "masked_image_base64" in result:
                    output_path = os.path.join(output_dir, "single_masked_result.jpg")
                    client.save_masked_image(result["masked_image_base64"], output_path)
        
        # Example 2: Batch tile processing
        print("\n" + "="*50)
        print("BATCH TILE PROCESSING")
        print("="*50)
        
        # Create sample tiles data (you would replace this with your actual tiles)
        tiles_data = [
            {
                "image_path": "GeoPixel/images/example1.png",
                "query": "Please give me a segmentation mask for buildings.",
                "tile_id": "tile_001"
            },
            {
                "image_path": "GeoPixel/images/example1.png", 
                "query": "Please give me a segmentation mask for vehicles.",
                "tile_id": "tile_002"
            },
            {
                "image_path": "GeoPixel/images/example1.png",
                "query": "Please give me a segmentation mask for roads.",
                "tile_id": "tile_003"
            }
        ]
        
        # Filter only existing images
        valid_tiles = [tile for tile in tiles_data if os.path.exists(tile["image_path"])]
        
        if valid_tiles:
            # Test batch processing
            print(f"🚀 Testing batch processing with {len(valid_tiles)} tiles")
            batch_result = client.process_tiles_parallel(valid_tiles, use_batch=True)
            
            if batch_result and "tile_results" in batch_result:
                print(f"✅ Batch processing completed:")
                print(f"   Total tiles: {batch_result.get('total_tiles', 0)}")
                print(f"   Processed: {batch_result.get('processed_tiles', 0)}")
                print(f"   Processing time: {batch_result.get('processing_time', 'Unknown')}")
                
                # Save masked images from batch
                for i, tile_result in enumerate(batch_result["tile_results"]):
                    if "masked_image_base64" in tile_result:
                        output_path = os.path.join(output_dir, f"batch_tile_{i}_masked.jpg")
                        client.save_masked_image(tile_result["masked_image_base64"], output_path)
        else:
            print("⚠️  No valid tile images found for batch processing demo")
        
        print(f"\n✅ Demo completed! Check the '{output_dir}' directory for results.")
        
    finally:
        client.close()

if __name__ == "__main__":
    demo_optimized_client()