import sys
from pathlib import Path
import os
import json
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

# -----------------------------
# Path Fix for Imports
# -----------------------------
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# -----------------------------
# Import Custom DFINE Model
# -----------------------------
from d_fine.dfine import DFINE, build_model

# -----------------------------
# Helper Classes and Functions
# -----------------------------
class BoundingBox:
    def __init__(self, x1, y1, x2, y2, score, label):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.label = label


def preprocess_image(image_path: str, img_size=(448, 1280)):
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height)
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, tensor, orig_size


def postprocess_predictions(preds, orig_size, class_names: List[str], conf_thresh=0.0):
    """
    Handle DFINE output dict {'pred_logits', 'pred_boxes'}.
    Converts normalized boxes to original image size and filters by confidence threshold.
    """
    boxes = []

    pred_logits = preds['pred_logits']  # [1, num_queries, num_classes]
    pred_boxes = preds['pred_boxes']    # [1, num_queries, 4]

    # Remove batch dimension
    pred_logits = pred_logits[0]  # [num_queries, num_classes]
    pred_boxes = pred_boxes[0]    # [num_queries, 4]

    # Convert logits to probabilities
    probs = torch.softmax(pred_logits, dim=-1)
    scores, labels = probs.max(dim=-1)

    for score, label_idx, box in zip(scores, labels, pred_boxes):
        score = score.item()
        label_idx = label_idx.item()
        if score < conf_thresh:
            continue

        # Convert normalized [cx, cy, w, h] to [x1, y1, x2, y2]
        cx, cy, w, h = box
        cx, cy, w, h = cx.item(), cy.item(), w.item(), h.item()
        x1 = (cx - w/2) * orig_size[0]
        x2 = (cx + w/2) * orig_size[0]
        y1 = (cy - h/2) * orig_size[1]
        y2 = (cy + h/2) * orig_size[1]

        # Fix inverted boxes
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Clamp to image size
        x1 = max(0, min(x1, orig_size[0]-1))
        x2 = max(0, min(x2, orig_size[0]-1))
        y1 = max(0, min(y1, orig_size[1]-1))
        y2 = max(0, min(y2, orig_size[1]-1))

        # Skip zero-area boxes
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue

        boxes.append(BoundingBox(x1, y1, x2, y2, score, class_names[label_idx]))

    return boxes


def save_image_with_boxes(image: Image.Image, boxes: List[BoundingBox], save_path: str):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for box in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline="red", width=2)
        if font:
            draw.text((box.x1, max(box.y1 - 10, 0)), f"{box.label} {box.score:.2f}", fill="red", font=font)
    image.save(save_path)


def save_json(boxes: List[BoundingBox], save_path: str):
    data = []
    for box in boxes:
        data.append({
            "label": box.label,
            "score": box.score,
            "bbox": [box.x1, box.y1, box.x2, box.y2]
        })
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


# -----------------------------
# Object Detection Pipeline
# -----------------------------
class ObjectDetectionPipeline:
    def __init__(self, model_path, class_names, device="cpu"):
        self.device = device
        self.class_names = class_names
        self.model = build_model(model_name="l", num_classes=len(class_names), device=device, pretrained_model_path=model_path)
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            preds = self.model(image_tensor.to(self.device))
        return preds


# -----------------------------
# Main Function
# -----------------------------
def main():
    # Use model_2.pt instead of model.pt
    model_path = Path("/Users/borde/label_studio_local/model_2.pt")

    class_names = [
        "0_Adhering_Skin","1_Blanched","2_Broken_Blanched","3_Mission","4_Carmel",
        "5_Chip_Scratch_1_4","6_Discolor","7_Doubles","8_Foreign_Material_Hull","9_NonPareil",
        "10_OD_Brownspot","11_SD_Insect_Damage","12_Specks","13_Split_Broken","14_LooseSkin_Dust_Particle",
        "15_OD_Gummy","16_SD_Pinhole","17_Inshell","18_Embeddedshell","19_FM_Other",
        "20_FM_Rock_Dirtball","21_FM_Pistachio","22_FM_Walnut","23_FM_Plastic","24_FM_Metal",
        "25_FM_Glass","26_SD_Mold","27_SD_Decay","28_SD_Other","29_SD_Frass",
        "30_OD_Shrivel","31_OD_Discolor","32_Chip_Scratch_1_8","33_California","34_Fold_Deformed"
    ]

    # Ask user for images folder or single image
    img_input = input("Enter path to images folder or single image for inference: ").strip()
    img_path = Path(img_input)
    if img_path.is_file():
        images = [img_path]
    elif img_path.is_dir():
        images = list(img_path.rglob("*.[jJ][pP][gG]")) + list(img_path.rglob("*.[pP][nN][gG]"))
        images = [img for img in images if img.is_file()]
    else:
        print("No valid images found!")
        return

    print(f"Found {len(images)} image(s) for inference.")
    if not images:
        print("No images found in the folder!")
        return

    # Ask user for confidence threshold
    conf_thresh = float(input("Enter confidence threshold (0.0 to show all predictions): ").strip() or 0.0)

    # Prepare output folders
    output_base = Path("outputs")
    json_folder = output_base / "json"
    vis_folder = output_base / "visualizations"
    json_folder.mkdir(parents=True, exist_ok=True)
    vis_folder.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = ObjectDetectionPipeline(model_path, class_names, device="cpu")

    # Process each image
    for img_file in images:
        image, tensor, orig_size = preprocess_image(str(img_file))
        preds = pipeline.predict(tensor)

        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC CODE - SAVE RAW LOGITS FOR COMPARISON
        # ═══════════════════════════════════════════════════════════════════
        pred_logits = preds['pred_logits']
        if isinstance(pred_logits, (list, tuple)):
            pred_logits = pred_logits[-1]

        pred_logits_cpu = pred_logits[0].cpu()  # [num_queries, num_classes]

        # Get image name
        image_basename = img_file.name
        base_name = img_file.stem
        output_file = f"logits_working_{base_name}.pt"

        torch.save({
            'logits': pred_logits_cpu,
            'shape': pred_logits_cpu.shape,
            'image': image_basename,
            'first_query_logits': pred_logits_cpu[0, :].tolist(),
            'first_10_queries_first_10_classes': pred_logits_cpu[:10, :10].tolist()
        }, output_file)

        print(f"\n{'='*70}")
        print(f"✅ SAVED RAW LOGITS TO: {output_file}")
        print(f"   Full path: {os.path.abspath(output_file)}")
        print(f"   Logits shape: {pred_logits_cpu.shape}")
        print(f"   First query, first 10 class logits: {pred_logits_cpu[0, :10].tolist()}")
        print(f"{'='*70}\n")
        # ═══════════════════════════════════════════════════════════════════
        # END DIAGNOSTIC CODE
        # ═══════════════════════════════════════════════════════════════════

        print(f"Raw predictions for {img_file.name}:", preds)

        boxes = postprocess_predictions(preds, orig_size, class_names, conf_thresh=conf_thresh)

        json_path = json_folder / f"{img_file.stem}.json"
        save_json(boxes, str(json_path))

        # Image name stays same (no "_vis")
        vis_path = vis_folder / f"{img_file.stem}.png"
        save_image_with_boxes(image, boxes, str(vis_path))

        print(f"Saved {len(boxes)} detections for {img_file.name}")
        print(f"JSON: {json_path}, Visualization: {vis_path}")


if __name__ == "__main__":
    main()
