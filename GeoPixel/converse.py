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
from model.geopixel import GeoPixelForCausalLM
import time

torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 precision
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 precision


def rgb_color_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def parse_args(args):
    parser = argparse.ArgumentParser(description="Chat with GeoPixel")
    parser.add_argument("--version", default="MBZUAI/GeoPixel-7B-RES")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    return parser.parse_args(args)

def main(args):
    start = time.time()
    MODEL_CACHE = {}

    # Get the absolute path to the GeoPixel directory
    geopixel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Add to Python path if needed
    if geopixel_path not in sys.path:
        sys.path.append(geopixel_path)

    args = parse_args(args)

    os.makedirs(args.vis_save_path, exist_ok=True)

    print(f'initialing tokenizer from: {args.version}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
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
        "seg_token_idx" : seg_token_idx, # segmentation token index
        "bop_token_idx" : bop_token_idx, # begining of phrase token index
        "eop_token_idx" : eop_token_idx  # end of phrase token index
    }

    torch.cuda.empty_cache()
    gc.collect()
    
    # Load model 
    print(f'Load model from: {args.version}')

    model_key = args.version
    if model_key in MODEL_CACHE:
        print(f"Using cached model for: {args.version}")
        model = MODEL_CACHE[model_key]
    else:
        print(f'Loading model from: {args.version}')
        # Load model directly without DeepSpeed
        model = GeoPixelForCausalLM.from_pretrained(
            args.version, 
            low_cpu_mem_usage=True, 
            **kwargs,
            **geo_model_args, 
            attn_implementation="flash_attention_2"
        )
        
    MODEL_CACHE[model_key] = model

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer
    
    model = model.bfloat16().cuda().eval()

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

    # Add after model loading
    try:
        # Check if the model has attention modules that support flash attention
        if hasattr(model, 'enable_flash_attention'):
            model.enable_flash_attention()
            print("Flash Attention enabled")
    except Exception as e:
        print(f"Flash Attention not supported: {str(e)}")


    
    query = "Could you provide a segmentation mask for red car in this image?"
    #query = f"Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases." 
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "example1-RES.jpg"))
    if not os.path.exists(image_path):
        print("File not found in {}".format(image_path))

    image = [image_path]

    cuda_stream = torch.cuda.Stream()
    with torch.cuda.stream(cuda_stream):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            response, pred_masks = model.evaluate(tokenizer, query, images=image, max_new_tokens=300)
    cuda_stream.synchronize()

    
    if pred_masks and '[SEG]' in response:
        pred_masks = pred_masks[0]
        pred_masks = pred_masks.detach().cpu().numpy()
        pred_masks = pred_masks > 0
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        save_img = image_np.copy()
        pattern = r'<p>(.*?)</p>\s*\[SEG\]'
        matched_text = re.findall(pattern, response)
        phrases = [text.strip() for text in matched_text]

        for i in range(pred_masks.shape[0]):
            mask = pred_masks[i]
            
            color = [random.randint(0, 255) for _ in range(3)]
            if matched_text:
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
    else:
        print(response.replace("\n", "").replace("  ", " "))

    torch.cuda.empty_cache()
    gc.collect()

    end = time.time()
    print("Finished in ", end-start, " seconds.")

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    main(args)