import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# 1. Define Quantization Config (Optional for 2B, but keeps it fast)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# 2. Load the 2B Model (Change the ID here)
# This model fits easily into 8GB VRAM
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",  # <--- CHANGED TO 2B
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

# 3. Load Processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", # <--- CHANGED TO 2B
    min_pixels=256*28*28, 
    max_pixels=1280*28*28
)

def extract_invoice_data(image_path):
    prompt_text = """
    Analyze this invoice and extract the following into JSON:
    - Dealer Name (Transliterate to English if needed)
    - Tractor Model (e.g. Swaraj 744, Sonalika DI-55)
    - Horse Power (HP)
    - Total Price (Numeric)
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

# --- TEST ---
if __name__ == "__main__":
    # Replace with your actual image filename
    try:
        result = extract_invoice_data("file_101.png") 
        print(result)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
