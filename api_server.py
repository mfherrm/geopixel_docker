import os
import sys
import re
import cv2
import torch
import random
import base64
import numpy as np
import transformers
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Setup Python path and Hydra configuration before importing GeoPixel
current_dir = os.path.dirname(os.path.abspath(__file__))
geopixel_dir = os.path.join(current_dir, 'GeoPixel')
if geopixel_dir not in sys.path:
    sys.path.insert(0, geopixel_dir)

# Initialize Hydra with the correct config path for SAM2
try:
    from hydra import initialize_config_dir, GlobalHydra
    import hydra
    
    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize Hydra with the SAM2 config directory
    sam2_config_dir = os.path.join(geopixel_dir, 'model', 'sam2_configs')
    sam2_config_dir = os.path.abspath(sam2_config_dir)
    
    if os.path.exists(sam2_config_dir):
        initialize_config_dir(config_dir=sam2_config_dir, version_base=None)
        print(f"Hydra initialized with config directory: {sam2_config_dir}")
    else:
        print(f"Warning: SAM2 config directory not found: {sam2_config_dir}")
        
except Exception as e:
    print(f"Warning: Failed to initialize Hydra: {str(e)}")

# Now import GeoPixel after setting up the environment
from GeoPixel.model.geopixel import GeoPixelForCausalLM

app = Flask(__name__)

# Global variables to store model and tokenizer
global_model = None
global_tokenizer = None
vis_save_path = "./vis_output"

def rgb_color_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def load_model(version="MBZUAI/GeoPixel-7B-RES"):
    """Load the model and tokenizer once at startup"""
    global global_model, global_tokenizer
    
    print(f'Initializing tokenizer from: {version}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx, bop_token_idx, eop_token_idx = [
        tokenizer(token, add_special_tokens=False).input_ids[0] for token in ['[SEG]','<p>', '</p>']
    ]
   
    kwargs = {"torch_dtype": torch.bfloat16}    
    geo_model_args = {
        "vision_pretrained": 'facebook/sam2-hiera-large',
        "seg_token_idx": seg_token_idx,  # segmentation token index
        "bop_token_idx": bop_token_idx,  # beginning of phrase token index
        "eop_token_idx": eop_token_idx   # end of phrase token index
    }
    
    # Load model 
    print(f'Loading model from: {version}')
    model = GeoPixelForCausalLM.from_pretrained(
        version, 
        low_cpu_mem_usage=True, 
        **kwargs,
        **geo_model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer
    
    model = model.bfloat16().cuda().eval()
    
    global_model = model
    global_tokenizer = tokenizer
    
    print("Model loaded successfully!")

def process_query(query, image_path):
    """Process a query with an image and return the results"""
    global global_model, global_tokenizer
    
    try:
        print(f"Processing query for image: {image_path}")
        
        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            return {"error": f"File not found: {image_path}"}, None
        
        # Check if the file is readable
        try:
            with open(image_path, 'rb') as f:
                pass
            print(f"Image file is readable")
        except Exception as e:
            print(f"Error: Cannot read image file: {str(e)}")
            return {"error": f"Cannot read image file: {str(e)}"}, None
        
        # Check if the file is a valid image
        try:
            test_img = cv2.imread(image_path)
            if test_img is None:
                print(f"Error: Invalid image file or format not supported")
                return {"error": "Invalid image file or format not supported"}, None
            print(f"Image dimensions: {test_img.shape}")
        except Exception as e:
            print(f"Error: Failed to read image with OpenCV: {str(e)}")
            return {"error": f"Failed to read image with OpenCV: {str(e)}"}, None
        
        # Prepare the image for the model
        image = [image_path]
        
        # Run inference
        print("Running inference...")
        try:
            # Use CUDA streams and mixed precision for optimal performance
            cuda_stream = torch.cuda.Stream()
            with torch.cuda.stream(cuda_stream):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        response, pred_masks = global_model.evaluate(global_tokenizer, query, images=image, max_new_tokens=300)
            cuda_stream.synchronize()
            print("Inference completed successfully")
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise
        
        # Process the response
        result = {"text_response": response.replace("\n", " ").replace("  ", " ")}
        print(f"Text response: {result['text_response'][:100]}...")
        
        # If we have masks, process the image
        masked_image_path = None
        try:
            if pred_masks is not None and '[SEG]' in response:
                try:
                    print("Processing segmentation masks...")
                    pred_masks = pred_masks[0]
                    pred_masks = pred_masks.detach().cpu().numpy()
                    pred_masks = pred_masks > 0
                    image_np = cv2.imread(image_path)
                    try:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        if image_np is None:
                            print(f"Warning: cv2.imread returned None for {image_path}")
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        print("Image loaded and converted to RGB")
                
                        save_img = image_np.copy()
                        pattern = r'<p>(.*?)</p>\s*\[SEG\]'
                        matched_text = re.findall(pattern, response)
                        phrases = [text.strip() for text in matched_text]
                        print(f"Found {len(phrases)} segmentation phrases")

                        for i in range(pred_masks.shape[0]):
                            mask = pred_masks[i]
                            
                            color = [random.randint(0, 255) for _ in range(3)]
                            if matched_text and i < len(phrases):
                                phrases[i] = rgb_color_text(phrases[i], color[0], color[1], color[2])
                            mask_rgb = np.stack([mask, mask, mask], axis=-1)
                            color_mask = np.array(color, dtype=np.uint8) * mask_rgb

                            save_img = np.where(mask_rgb,
                                (save_img * 0.5 + color_mask * 0.5).astype(np.uint8),
                                save_img)
                                    
                        if matched_text:
                            split_desc = response.split('[SEG]')
                            cleaned_segments = [re.sub(r'<p>(.*?)</p>', '', part).strip() for part in split_desc]
                            reconstructed_desc = ""
                            for i, part in enumerate(cleaned_segments):
                                reconstructed_desc += part + ' '
                                if i < len(phrases):
                                    reconstructed_desc += phrases[i] + ' '
                            print(reconstructed_desc)
                        else:
                            print(response.replace("\n", "").replace("  ", " "))
                
                        # Save the masked image
                        try:
                            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                            os.makedirs(vis_save_path, exist_ok=True)
                            save_path = "{}/{}_masked.jpg".format(
                                vis_save_path, os.path.basename(image_path).split(".")[0]
                            )
                            cv2.imwrite(save_path, save_img)
                            masked_image_path = save_path
                            result["masked_image_path"] = masked_image_path
                            print(f"Masked image saved to: {masked_image_path}")
                        except Exception as e:
                            print(f"Error saving masked image: {str(e)}")
                            result["masked_image_error"] = f"Failed to save masked image: {str(e)}"
                    except Exception as e:
                        print(f"Error processing masks: {str(e)}")
                        
                        traceback.print_exc()
                        result["mask_error"] = f"Error processing masks: {str(e)}"
                except Exception as e:
                    print(f"Error while processing: {str(e)}")
                    traceback.print_exc()
                    return {"error": f"Unhandled error: {str(e)}"}, None
        except Exception as e:
            print(f"Unhandled error in process_query: {str(e)}")
            traceback.print_exc()
            return {"error": f"Unhandled error: {str(e)}"}, None
                
        print("Query processing complete")
        return result, masked_image_path, pred_masks
    except Exception as e:
        print(f"Unhandled exception in process_query: {str(e)}")
        traceback.print_exc()
        return {"error": f"Unhandled error: {str(e)}"}, None, None

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        "name": "GeoPixel API",
        "description": "API for processing images with the GeoPixel model",
        "endpoints": {
            "/": "This information",
            "/health": "Check if the model is loaded and ready (GET)",
            "/process": "Process an image with a query (POST)"
        },
        "usage": {
            "health_check": "GET /health",
            "process_image": "POST /process with 'image' file and 'query' form data"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if global_model is None or global_tokenizer is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok", "message": "Model is loaded and ready"}), 200

@app.route('/process', methods=['POST'])
def process_request():
    """Process an image and query"""
    try:
        print("Received /process request")
        
        if 'image' not in request.files:
            print("Error: No image file provided")
            return jsonify({"error": "No image file provided"}), 400
        
        if 'query' not in request.form:
            print("Error: No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        query = request.form['query']
        image_file = request.files['image']
        
        print(f"Processing query: {query}")
        print(f"Image filename: {image_file.filename}")
        
        # Check if model is loaded
        if global_model is None or global_tokenizer is None:
            print("Error: Model not loaded")
            return jsonify({"error": "Model not loaded. Please try again later."}), 500
        
        # Create temp directory if it doesn't exist
        os.makedirs('/tmp', exist_ok=True)
        
        # Save the uploaded image
        try:
            filename = secure_filename(image_file.filename)
            temp_path = os.path.join('/tmp', filename)
            image_file.save(temp_path)
            print(f"Image saved to: {temp_path}")
            
            # Check if the file was saved correctly
            if not os.path.exists(temp_path):
                print(f"Error: Failed to save image to {temp_path}")
                return jsonify({"error": f"Failed to save uploaded image"}), 500
                
            if os.path.getsize(temp_path) == 0:
                print(f"Error: Saved image is empty")
                return jsonify({"error": "Uploaded image is empty"}), 400
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return jsonify({"error": f"Error saving image: {str(e)}"}), 500
        
        # Process the query
        try:
            result, masked_image_path, pred_masks = process_query(query, temp_path)
            print(f"Query processed. Result: {result.keys()}")
            
            # Include prediction masks in the result if available
            if pred_masks is not None:
                try:
                    # Convert numpy masks to a serializable format
                    # First, ensure it's a numpy array and convert to binary format
                    if hasattr(pred_masks, 'detach'):
                        # If it's a PyTorch tensor, detach and convert to numpy
                        masks_np = pred_masks.detach().cpu().numpy()
                    else:
                        # If it's already a numpy array
                        masks_np = pred_masks
                    
                    # Convert to binary (0 or 1)
                    masks_binary = (masks_np > 0).astype(np.uint8)
                    
                    # Encode each mask as base64
                    masks_encoded = []
                    for i in range(masks_binary.shape[0]):
                        mask = masks_binary[i]
                        # Convert to PNG format
                        _, buffer = cv2.imencode('.png', mask * 255)
                        # Encode as base64
                        mask_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        masks_encoded.append(mask_b64)
                    
                    # Add to result
                    result["pred_masks_base64"] = masks_encoded
                    result["pred_masks_count"] = len(masks_encoded)
                    print(f"Added {len(masks_encoded)} prediction masks to response")
                except Exception as e:
                    print(f"Error encoding prediction masks: {str(e)}")
                    traceback.print_exc()
                    result["pred_masks_error"] = f"Error encoding masks: {str(e)}"
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error processing query: {str(e)}"}), 500
        
        # If there was an error
        if "error" in result:
            print(f"Error in result: {result['error']}")
            return jsonify(result), 400
        
        # If we have a masked image, read it and encode as base64
        if masked_image_path and os.path.exists(masked_image_path):
            try:
                with open(masked_image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    result["masked_image_base64"] = img_data
                    print("Masked image encoded successfully")
            except Exception as e:
                print(f"Error encoding masked image: {str(e)}")
                # Continue without the masked image
                result["masked_image_error"] = str(e)
        
        print("Request processed successfully")
        return jsonify(result), 200
    except Exception as e:
        print(f"Unhandled error in process_request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        # Load the model at startup
        print("Starting GeoPixel API server...")
        print("Loading model - this may take a few minutes...")
        load_model()
        
        # Run the Flask app
        print("Model loaded successfully! Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"ERROR: Failed to start API server: {str(e)}")
        import traceback
        traceback.print_exc()