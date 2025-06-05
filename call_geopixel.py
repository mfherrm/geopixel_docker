import os
import re
import gc
import sys
import cv2
import torch
import random
import argparse
import numpy as np
import transformers
import time

torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 precision
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 precision

start = time.time()
# Get the absolute path to the directory containing 'fachanwendung's parent
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)

geopixel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GeoPixel'))

# Add the GeoPixel directory to the Python path if it's not already there
if geopixel_path not in sys.path:
    sys.path.append(geopixel_path)

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '...', '...', '...', 'GeoPixel', 'model'))

# Add the GeoPixel directory to the Python path if it's not already there
if model_path not in sys.path:
    sys.path.append(model_path)

# Now you can try to import from GeoPixel.chat
from GeoPixel.chat import parse_args, rgb_color_text  # Make sure rgb_color_text is imported
from model.geopixel import GeoPixelForCausalLM # type: ignore

def get_geopixel_result(args, objects):
    start = time.time()
    
    try:

        # Parse arguments
        try:
            args = parse_args(args)
            print(f"Arguments parsed successfully: {args}")
        except Exception as e:
            print(f"Error parsing arguments: {str(e)}")
            raise

        # Create output directory
        try:
            os.makedirs(args.vis_save_path, exist_ok=True)
            print(f"Output directory created: {args.vis_save_path}")
        except Exception as e:
            print(f"Error creating output directory: {str(e)}")
            raise

        # Initialize tokenizer
        print(f'Initializing tokenizer from: {args.version}')
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.version,
                cache_dir=None,
                padding_side='right',
                use_fast=False,
                trust_remote_code=True,
            )
            tokenizer.pad_token = tokenizer.unk_token
            print("Tokenizer initialized successfully")
        except Exception as e:
            print(f"Error initializing tokenizer: {str(e)}")
            raise

        # Get special token indices
        try:
            seg_token_idx, bop_token_idx, eop_token_idx = [
                tokenizer(token, add_special_tokens=False).input_ids[0] for token in ['[SEG]','<p>', '</p>']
            ]
            print(f"Special token indices: SEG={seg_token_idx}, BOP={bop_token_idx}, EOP={eop_token_idx}")
        except Exception as e:
            print(f"Error getting special token indices: {str(e)}")
            raise
       
        # Prepare model arguments
        kwargs = {"torch_dtype": torch.bfloat16}
        geo_model_args = {
            "vision_pretrained": 'facebook/sam2-hiera-large',
            "seg_token_idx" : seg_token_idx, # segmentation token index
            "bop_token_idx" : bop_token_idx, # begining of phrase token index
            "eop_token_idx" : eop_token_idx  # end of phrase token index
        }
        print(f"Model arguments prepared: {geo_model_args}")

        # Clear memory
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load model
        print(f'Loading model from: {args.version}')
        
        # Load the model without quantization to avoid PIL.ImageFont deepcopy issues
        try:
            print("Loading model with the following parameters:")
            print(f"  - Model version: {args.version}")
            print(f"  - Low CPU memory usage: True")
            print(f"  - Device map: None (disabled to avoid meta tensor issues)")
            print(f"  - Using quantization: None (disabled to avoid PIL.ImageFont issues)")
            print(f"  - Using flash attention: True")
            
            model = GeoPixelForCausalLM.from_pretrained(
                args.version,
                low_cpu_mem_usage=True,
                **kwargs,
                **geo_model_args,
                attn_implementation="flash_attention_2"
            )
            
            # Move model to GPU manually
            model = model.to("cuda")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Set model configuration
        try:
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.bos_token_id = tokenizer.bos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.tokenizer = tokenizer
            print("Model configuration set")
        except Exception as e:
            print(f"Error setting model configuration: {str(e)}")
            raise
        
        # Set model to eval mode
        model.eval()
        print("Model set to evaluation mode")
        
        # Apply additional optimizations if available
        if hasattr(torch, 'compile'):
            try:
                print("Compiling model with torch.compile()...")
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compilation successful")
            except Exception as e:
                print(f"Model compilation failed: {str(e)}")
        
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        except Exception as e:
            print(f"Gradient checkpointing not supported: {str(e)}")

        # Prepare query and image
        try:
            query = f"Please return segmentation masks of all {', '.join(objects)}"
            print(f"Query: {query}")
            
            image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "images", "example1-RES.jpg"))
            print(f"Image path: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"Error: File not found at {image_path}")
                # Try to list files in the directory to help debugging
                try:
                    image_dir = os.path.dirname(image_path)
                    if os.path.exists(image_dir):
                        print(f"Files in {image_dir}:")
                        for file in os.listdir(image_dir):
                            print(f"  - {file}")
                    else:
                        print(f"Directory {image_dir} does not exist")
                except Exception as e:
                    print(f"Error listing directory: {str(e)}")
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = [image_path]
            print("Image prepared for processing")
        except Exception as e:
            print(f"Error preparing query and image: {str(e)}")
            raise

        # Run inference
        print("Running inference...")
        try:
            # Use CUDA streams and mixed precision for optimal performance
            cuda_stream = torch.cuda.Stream()
            with torch.cuda.stream(cuda_stream):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        response, pred_masks = model.evaluate(tokenizer, query, images=image, max_new_tokens=300)
            cuda_stream.synchronize()
            print("Inference completed successfully")
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise
            
    except Exception as e:
        print(f"ERROR: An error occurred during model execution: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Process results if we have valid masks
    try:
        if pred_masks is not None and '[SEG]' in response:
            try:
                pred_masks = pred_masks[0]
                pred_masks = pred_masks.detach().cpu().numpy()
                pred_masks = pred_masks > 0
                print(f"Masks processed: shape={pred_masks.shape}")
                
                try:
                    image_np = cv2.imread(image_path)
                    if image_np is None:
                        print(f"Warning: cv2.imread returned None for {image_path}")
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    print("Image loaded and converted to RGB")
                    
                    save_img = image_np.copy()
                    pattern = r'<p>(.*?)</p>\s*\[SEG\]'
                    matched_text = re.findall(pattern, response)
                    phrases = [text.strip() for text in matched_text]
                    print(f"Found {len(phrases)} text phrases to match with masks")

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
                    
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                    save_path = "{}/{}_masked.jpg".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0]
                        )
                    cv2.imwrite(save_path, save_img)
                    print("{} has been saved.".format(save_path))
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
            except Exception as e:
                print(f"Error processing masks: {str(e)}")
        else:
            if pred_masks is None:
                print("No masks were generated")
            else:
                print("No segmentation tokens found in response")
            print(response.replace("\n", "").replace("  ", " "))
    except Exception as e:
        print(f"Error in post-processing: {str(e)}")

    # Clean up
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleaned up")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

    # Calculate runtime
    end = time.time()
    runtime = end-start
    print(f"Finished in {runtime} seconds.")

    return pred_masks
