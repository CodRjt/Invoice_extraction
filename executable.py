from utils.bextractor import extract_invoice_data 
from utils.yolo import get_boxes
import argparse
import json
import os
from datetime import datetime
def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='Extract invoice data from tractor invoice images')
    parser.add_argument('image_path', type=str, help='Path to the invoice image file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found!")
        return
    
    print(f"Processing: {args.image_path}")
    print("-" * 50)
    
    try:
        out = extract_invoice_data(args.image_path)
        boxes = get_boxes(args.image_path)
        end_time = datetime.now()
        fields = json.loads(out)
        curr_id = 0
        with open("utils/id.txt", "r") as f:

            curr_id = int(f.read().strip())

        curr_id += 1
        
        if boxes[0] != None:
            stamp = {"present": True, "bbox": boxes[0]['bbox']}
        else:
            stamp = {"present": False, "bbox": []}
        if boxes[1] != None:
            signature = {"present": True, "bbox": boxes[1]['bbox']}
        else:
            signature = {"present": False, "bbox": []}

        fields['signature'] = signature
        fields['stamp'] = stamp

        confidence = boxes[0]['confidence'] if boxes[0]['confidence'] else 0 + boxes[1]['confidence'] if boxes[1]['confidence'] else 0
        confidence /= 2
        with open("utils/id.txt", "w") as f:
            f.write(f"{curr_id}")
        result = {}
        result['doc_id']  = f"invoice_{curr_id}"
        result["fields"] = fields
        result['confidence'] = confidence
        result["processing_time_sec"] = (end_time-start_time).total_seconds()
        val=(1/result["processing_time_sec"]*3600)

        result["cost_estimated_usd"]  = 0.052/val
        with open(f"outputs/{result['doc_id']}.json", "w") as f:
            json.dump(result, f)
        print(f"\nSaved to: outputs/{result['doc_id']}.json")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
