import gradio as gr
import cv2
import numpy as np

# Mock function to represent your backend ML model
def process_invoice(input_img):
    # 1. Logic for OCR/Extraction would go here
    # 2. Logic for Stamp/Signature Detection would go here
    
    # Mock Data for demonstration
    extracted_data = {
        "dealer_name": "Agro-Trac Equipment Corp",
        "model_name": "X-700 Turbo",
        "hp": 75,
        "price": 45000.00,
        "signature_bbox": [50, 400, 150, 500], # [y1, x1, y2, x2]
        "stamp_bbox": [60, 100, 180, 250]
    }

    # Draw bounding boxes on the image for the UI
    annotated_img = input_img.copy()
    
    # Draw Signature Box (Blue)
    s_box = extracted_data["signature_bbox"]
    cv2.rectangle(annotated_img, (s_box[1], s_box[0]), (s_box[3], s_box[2]), (255, 0, 0), 3)
    cv2.putText(annotated_img, "Signature", (s_box[1], s_box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Draw Stamp Box (Green)
    st_box = extracted_data["stamp_bbox"]
    cv2.rectangle(annotated_img, (st_box[1], st_box[0]), (st_box[3], st_box[2]), (0, 255, 0), 3)
    cv2.putText(annotated_img, "Stamp", (st_box[1], st_box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return (
        extracted_data["dealer_name"],
        extracted_data["model_name"],
        extracted_data["hp"],
        f"${extracted_data['price']:,}",
        "Yes" if s_box else "No",
        "Yes" if st_box else "No",
        annotated_img
    )

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚜 Invoice Extraction & Verification")
    gr.Markdown("Upload an invoice image to extract key asset details and verify the presence of stamps and signatures.")
    
    with gr.Row():
        with gr.Column(scale=1):
            invoice_input = gr.Image(label="Upload Invoice Image")
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
        output_image = gr.Image(label="Detection Visualization (Signature & Stamp)")

    submit_btn.click(
        fn=process_invoice,
        inputs=[invoice_input],
        outputs=[dealer, model, hp, cost, sig_present, stamp_present, output_image]
    )

demo.launch()
