import torch
import argparse
import os
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

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
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

# 3. Load Processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels=256*28*28, 
    max_pixels=1280*28*28
)
print("Model loaded successfully!\n")

def extract_invoice_data(image_path):
    """Extract invoice data from an image file"""
    prompt_text_2 = f"""
    You are an information extraction engine.
    Return ONLY valid JSON.
    No explanations.
    No extra text.
    Extract the following fields in there original language from the invoice image.

    Required JSON schema:
    {{
      "dealer_name": string,
      "model_name": string,
      "horse_power": number,
      "asset_cost": number
    }}
    """
    wprompt_text = """You are an information extraction engine.
Return ONLY valid JSON.
No explanations.
No extra text.
Extract the following fields from the invoice image:

Required JSON schema:
{
  "dealer_name": string,
  "model_name": string,
  "horse_power": number,
  "asset_cost": number
}

Rules:
- dealer_name: Extract the business/company name (e.g., "Coppercity Tractors")
- model_name: Extract the tractor model (e.g., "DI-35", "GT-22")
- horse_power: Extract the HP value as a number
- asset_cost: Extract the total amount/price as a number (no currency symbols)
"""
    prompt_text = """You are an information extraction engine.
Return ONLY valid JSON.
No explanations.
No extra text.
Extract the following fields from the invoice image:

Required JSON schema:
{
  "dealer_name": string,
  "model_name": string,
  "horse_power": number,
  "asset_cost": number,
}

CRITICAL RULES:
- dealer_name: Extract the business/company name (use names ending with Tractors,Motors,Traders,Automobiles etc if available) (e.g., "Coppercity Tractors")
- model_name: IMPORTANT - If multiple models are listed, find the one with a tick mark (✓), checkmark, or any marking next to it. Only extract that model name. If no mark is visible, extract the model with a price filled in.
- horse_power: Extract the HP value as a number (look near the selected model_name)
- asset_cost: Extract the total amount/price as a number (no currency symbols, no commas)


Look carefully for:
- Tick marks (✓)
- Checkmarks
- Hand-written marks or circles
- Filled price amounts (models without prices are not selected)


"""
    wprompt_text = """You are an information extraction engine.
Return ONLY valid JSON.
No explanations.
No extra text.
Extract the following fields from the invoice image:

Required JSON schema:
{
  "dealer_name": string,
  "model_name": string,
  "horse_power": number,
  "asset_cost": number
}

CRITICAL RULES:
1. dealer_name: Extract the main business/company name from the header (e.g., "Mahindra", "Coppercity Tractors")

2. model_name: VERY IMPORTANT
   - If invoice shows a table/list with multiple models, find the row with:
     * A tick mark (✓)
     * A checkmark
     * A hand-written circle or mark
     * A filled price/amount in that row
   - Extract ONLY the model name from that marked row
   - Ignore all other model names

3. horse_power: Extract the HP value from THE SAME ROW as the selected model
   - Must be from the same line/row as the model_name
   - Extract only the number (e.g., if "39 HP", return 39)

4. asset_cost: Extract the total price/amount
   - Look for the largest number, usually at bottom right
   - Remove commas and currency symbols
   - Return as number only

VISUAL CUES TO LOOK FOR:
- Checkmarks or tick marks next to model names
- Hand-written prices next to specific models
- Circles or underlines highlighting models
- The selected row will have MORE information filled in than others
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
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def main():
    parser = argparse.ArgumentParser(description='Extract invoice data from tractor invoice images')
    parser.add_argument('image_path', type=str, help='Path to the invoice image file')
    parser.add_argument('--output', '-o', type=str, help='Optional: Save output to JSON file')
    parser.add_argument('--pretty', '-p', action='store_true', help='Pretty print the output')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found!")
        return
    
    print(f"Processing: {args.image_path}")
    print("-" * 50)
    
    try:
        result = extract_invoice_data(args.image_path)
        
        if args.pretty:
            print("\n=== EXTRACTED DATA ===")
            print(result)
            print("=" * 50)
        else:
            print(result)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"image": args.image_path, "result": result}, f, indent=2)
            print(f"\nSaved to: {args.output}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
