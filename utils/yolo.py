from ultralytics import YOLO

def get_boxes(image_path):

    # Load your custom trained model
    model = YOLO('utils/actual_best.pt')
    results = model(image_path, save=True)
    best_boxes = {
        0: None,  # will store dict for class 0
        1: None   # will store dict for class 1
    }

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if cls in best_boxes:
                if best_boxes[cls] is None or conf > best_boxes[cls]["confidence"]:
                    best_boxes[cls] = {
                        "class": cls,
                        "confidence": conf,
                        "bbox": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
                    }
            return best_boxes

