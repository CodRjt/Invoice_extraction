import gradio as gr
import cv2
import numpy as np
import json
from utils.bextractor import extract_invoice_data 
from utils.yolo import get_boxes

def process_invoice(image_path):
    # 1. AI Extraction (Requires File Path)
    # image_path is a string like "/tmp/gradio/xyz.jpg"
    out = extract_invoice_data(image_path)
    
    # 2. YOLO Detection (Requires File Path)
    boxes = get_boxes(image_path)
    
    # 3. Load Image for OpenCV (Requires NumPy Array)
    # We must read the file into a matrix to draw on it
    annotated_img = cv2.imread(image_path)
    # Convert from BGR (OpenCV default) to RGB (Gradio default) for correct display
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Parse JSON output
    fields = json.loads(out)
    print(out)
    print("+"*239)
    # Safely handle boxes
    stamp_box = []
    if boxes.get(0): # Check if class 0 exists
        stamp_box = boxes[0]['bbox'] # Expected [x1, y1, x2, y2]

    signature_box = []
    if boxes.get(1): # Check if class 1 exists
        signature_box = boxes[1]['bbox']

    # Update fields
    fields['stamp_bbox'] = stamp_box
    fields['signature_bbox'] = signature_box
    hp,=fields.get("horse_power", 0),
    print(hp)
    if hp<10:
        hp*=10
    if hp>80:
        hp=45
    # --- Draw Signature Box (Blue) ---
    if signature_box:
        # Ensure coordinates are integers
        x1, y1, x2, y2 = map(int, signature_box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red/Blue
        cv2.putText(annotated_img, "Signature", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # --- Draw Stamp Box (Green) ---
    if stamp_box:
        x1, y1, x2, y2 = map(int, stamp_box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated_img, "Stamp", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    raw_cost = fields.get('asset_cost', 0)
    try:
        # Remove '$' and ',' if they exist in the string, then convert to float
        clean_cost = float(str(raw_cost).replace('$', '').replace(',', ''))
        if clean_cost>2000000:
            clean_cost//=10
            clean_cost=int(clean_cost)
        formatted_cost = f"Rs {clean_cost:,}"
    except ValueError:
        # If conversion fails (e.g., value is "N/A"), just use the raw text
        formatted_cost = str(raw_cost)
    return (
        fields.get("dealer_name", "N/A"),
        fields.get("model_name", "N/A"),
        hp, 
        formatted_cost,    
        True if signature_box else False,
        True if stamp_box else False,
        annotated_img
    )

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Invoice Extraction & Verification")
    gr.Markdown("Upload an invoice image to extract key asset details and verify the presence of stamps and signatures.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # INPUT: Must be filepath for the AI models
            invoice_input = gr.Image(label="Upload Invoice Image", type="filepath")
            submit_btn = gr.Button("Extract Details", variant="primary")
            
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Extracted Information")
                dealer = gr.Textbox(label="Dealer Name (Fuzzy Match)")
                model = gr.Textbox(label="Model Name (Exact Match)")
                hp = gr.Number(label="Horse Power")
                cost = gr.Textbox(label="Asset Cost")
            
            with gr.Row():
                sig_present = gr.Checkbox(label="Signature Detected")
                stamp_present = gr.Checkbox(label="Stamp Detected")

    with gr.Row():
        # OUTPUT: Gradio receives the NumPy array we drew on
        output_image = gr.Image(label="Detection Visualization")

    submit_btn.click(
        fn=process_invoice,
        inputs=[invoice_input],
        outputs=[dealer, model, hp, cost, sig_present, stamp_present, output_image]
    )

demo.launch()
