import torch
import os
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import re

# 1. Define Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# 2. Load the 2B Model
print("Loading model... (this may take a moment)")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

# 3. Load Processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels=512*28*28, 
    max_pixels=1280*28*28
)
print("Model loaded successfully!\n")

def extract_invoice_data(image_path):
    """Extract invoice data from an image file"""
    prompt_text = """You are an expert document analyst. 
    Analyze the invoice image specifically for tractor sales details.

    CRITICAL RULES FOR EXTRACTION:
    1. DEALER: Find the main dealer name at the top (usually ends in Motors, Tractors, etc).
    2. MODEL SELECTION (Most Important): 
       - Look for a table or list of models.
       - Find the SPECIFIC ROW that has a handwritten Tick Mark (✓), Checkmark, or Circle.
       - If a price is handwritten next to only one model, choose that one.
       - IGNORE models that are listed but not marked.
    3. HP: Extract the Horse Power from the SAME ROW as the selected model.

    Output format: Return valid JSON only.
    {
      "_visual_check": "Describe strictly which row has the tick mark or checkmark.",
      "dealer_name": "string",
      "model_name": "string",
      "horse_power": "number",
      "asset_cost": "number"
    }
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    raw_response = output_text[0]
    
    # Optional: Clean markdown code blocks if the model adds them (e.g. ```json ... ```)
    clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    
    return clean_json

