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
    prompt_text = """You are a precision document extraction agent. Analyze the tractor invoice image and return data according to these STRICT rules:

    CRITICAL EXTRACTION PROTOCOL:

    1. DEALER NAME (Distinguish from Brand):
       - Look for the primary business name in the central header area.
       - **CRITICAL:** Do NOT return the tractor brand name (e.g., "SWARAJ", "MAHINDRA", "TAFE", "ESCORTS") as the dealer name. These are manufacturer logos.
       - The Dealer Name usually contains words like "Agencies", "Motors", "Tractors", or "Automobiles". 
       - If the name is in a local script (like Marathi or Hindi), transcribe it into English or its phonetic equivalent (e.g., "Siddhanath Agro Agency").

    2. MODEL NAME (Strict Pattern: [Number] [Word]):
       - Locate the tractor description in the 'Particulars' or 'Vivarana' column.
       - The model name MUST follow this exact pattern: [3-digit number] [Remaining variant text].
       - Example: For "Swaraj 744 FE", return "744 FE". For "Target 630", return "630 Target".
       - NEVER include "HP" in this field.

    3. HORSE POWER (Strictly Numerical Integer):
       - Scan for the specific label "H.P.", "HP", or "Horse Power".
       - Extract ONLY the 2-digit integer (e.g., 29, 41, 48). 
       - Return as a raw number. DO NOT include the string "HP".

    4. ASSET COST (Strictly Numerical Integer):
       - Extract the final grand total value.
       - Return a raw number only. Remove all text like "Rs", symbols, and commas.

    Output format: Return valid JSON only.
    {
      "dealer_name": "string",
      "model_name": "string",
      "horse_power": number,
      "asset_cost": number
    }
    """
    dsprompt_text = """You are a precision document extraction agent. Analyze the tractor invoice image and return data according to these STRICT structural and type rules:


    CRITICAL EXTRACTION PROTOCOL:

    1. DEALER NAME:
       - Extract the primary business name from the letterhead (e.g., "SAI MOTORS").

    2. MODEL NAME (Strict Pattern: [Number] [Word]):
       - Locate the tractor description in the 'Particulars' or 'Vivarana' column.
       - The model name MUST follow this exact pattern: [3-digit number] [Remaining variant text].
       - Example: If the text is "Swaraj tractor 744 FE Model 48 HP", you must return "744 FE".
       - Example: If the text is "Powertrac 439 Plus", you must return "439 Plus".
       - NEVER include "HP" or horsepower values in the model name field.

    3. HORSE POWER (Strictly Numerical Integer):
       - Scan for the specific label "H.P.", "HP", or "Horse Power".
       - Extract only the 2-digit integer immediately next to that label.
       - Return this as a raw number. DO NOT include the string "HP".
       - Example: From "48 HP", return 48.

    4. ASSET COST (Strictly Numerical Integer):
       - Extract the final grand total value from the amount column.
       - Return this as a raw number. Remove all currency symbols (Rs, $) and commas.

    Output format: Return valid JSON only.
    {
      "dealer_name": "string",
      "model_name": "string",
      "horse_power": number,
      "asset_cost": number
    }
    """
    ewprompt_text = """You are a precision document extraction agent. Your goal is to extract tractor invoice data while strictly distinguishing between Model Numbers and Horse Power (HP).

    EXTRACTION PROTOCOL:

    1. DEALER NAME:
       - Extract the primary business name from the letterhead (e.g., Kapishwara Tractors, SAI MOTORS).

    2. MODEL NAME:
       - Identify the handwritten tractor model in the 'Particulars' or 'Vivarana' column.
       - Include the brand and the 3-digit series number (e.g., "Powertrac 439 Plus", "Swaraj 744 FE").

    3. HORSE POWER (HP) - CRITICAL VALIDATION:
       - **STEP A:** Scan for the specific handwritten label "H.P." or "HP".
       - **STEP B:** Extract ONLY the 2-digit number written immediately next to that label (e.g., 41, 48, 50).
       - **STEP C (Conflict Check):** Compare this value to the Model Name. 
         - If the Model Name is "439", the HP CANNOT be 439.
         - If the Model Name is "744", the HP CANNOT be 744.
       - **RULE:** If you cannot find a 2-digit number explicitly labeled as HP, look for a 2-digit number followed by 'HP' (e.g., '48 HP').Strictly Output the numerical part.

    4. ASSET COST:
       - Extract the final grand total numerical value from the amount column.

    Output format: Return valid JSON only.
    {
      "dealer_name": "string",
      "model_name": "string",
      "horse_power": "number",
      "asset_cost": "number"
    }
    """
    eprompt_text = """You are an expert document analyst.
    Analyze the invoice image specifically for tractor sales details. The document format varies (some are handwritten, some are checklists).

    CRITICAL EXTRACTION LOGIC (Follow in Order):

    1. DEALER NAME:
       - Identify the large bold header at the top (e.g., Kapishwara Tractors, National Motors).

    2. MODEL NAME (Use Priority Logic):
       - PRIORITY A (Handwritten Description): First, look at the "Particulars", "Description", or "Vivarana" column. If a model name is written there by hand (e.g., "Powertrac 439", "MF 241"), EXTRACT THAT.
       - PRIORITY B (Checklist/Tick Mark): If there is no handwritten model name, look for a printed list of models. Select the row that has a handwritten Checkmark (✓), Tick, or X inside a box or next to the name.
       - Combine the Brand (Escorts, Massey, etc.) with the Model Number.

    3. HORSE POWER (HP):
       - Look for the text label "H.P." or "HP" specifically.
       - It is often written near the model name or Cylinder count (e.g., "41 HP", "42 HP").
       - CAUTION: Do not confuse the Model Number (like 241, 439, 7250) with the HP. HP is usually a 2-digit number between 30 and 90.

    4. ASSET COST:
       - Extract the Grand Total / Final Price written in the Amount column.

    Output format: Return valid JSON only.
    {
      "_visual_reasoning": "Briefly explain: Did you find a handwritten name or a tick mark?",
      "dealer_name": "string",
      "model_name": "string",
      "horse_power": "number",
      "asset_cost": "number"
    }
    """
    prompt_text_2 = """You are an expert document analyst. 
    Analyze the invoice image specifically for tractor sales details.

    CRITICAL RULES FOR EXTRACTION:
    1. DEALER: Find the main dealer name at the top (usually ends in Motors, Tractors, etc).
    2. MODEL SELECTION (Most Important): 
       - Look for a table or list of models.
       - Find the SPECIFIC ROW that has a handwritten Tick Mark (✓), Checkmark, or Circle.
       - If a price is handwritten next to only one model, choose that one.
       - IGNORE models that are listed but not marked.
       - The MODEL SELECTION is of two parts a number followed by a word
    3. HP: Extract the Horse Power from the SAME ROW as the selected model.The Horse Power is a two digit number.

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

