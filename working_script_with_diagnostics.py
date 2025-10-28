"""
Modified version of your working script with diagnostic output added.

INSTRUCTIONS:
1. Run this on the SAME image you tested with the standalone script
2. This will create a file called "logits_working_<imagename>.pt"
3. Share both:
   - logits_working_<imagename>.pt (from this script)
   - logits_<imagename>.pt (from standalone script)

If the logits match → score calculation issue
If the logits differ → preprocessing issue
"""

import torch
import torchvision.transforms as T
from PIL import Image
from typing import List
import json
import os

# YOUR MODEL IMPORTS HERE - Replace with your actual imports
# from d_fine import build_model  # Or whatever your imports are


class AlmondDefectDetector:
    def __init__(self, model_path: str, class_names: List[str], device="cpu"):
        self.device = torch.device(device)
        self.class_names = class_names

        # REPLACE THIS WITH YOUR ACTUAL MODEL LOADING CODE
        # self.model = build_model(model_name="l", num_classes=len(class_names),
        #                         device=device, pretrained_model_path=model_path)
        print(f"⚠️  YOU NEED TO UNCOMMENT YOUR MODEL LOADING CODE")
        print(f"⚠️  Replace the commented lines with your actual build_model() call")

    def preprocess_image(self, image_path: str, img_size=(448, 1280)):
        """Preprocess image - EXACT copy from your working script"""
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (width, height)

        transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])
        tensor = transform(image).unsqueeze(0)

        # DIAGNOSTIC: Print preprocessing info
        print(f"[Preprocessing] Original size: {orig_size}")
        print(f"[Preprocessing] Resize to: {img_size}")
        print(f"[Preprocessing] Tensor shape: {tensor.shape}")
        print(f"[Preprocessing] Tensor range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

        return image, tensor, orig_size

    def predict(self, tensor):
        """Run inference - EXACT copy from your working script"""
        tensor = tensor.to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)

        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC CODE ADDED - THIS IS THE CRITICAL PART
        # ═══════════════════════════════════════════════════════════════════
        pred_logits = preds['pred_logits']
        if isinstance(pred_logits, (list, tuple)):
            pred_logits = pred_logits[-1]

        pred_logits_cpu = pred_logits[0].cpu()  # [num_queries, num_classes]

        print(f"\n[DIAGNOSTIC] Raw model logits:")
        print(f"             Shape: {pred_logits_cpu.shape}")
        print(f"             Min: {pred_logits_cpu.min().item():.3f}")
        print(f"             Max: {pred_logits_cpu.max().item():.3f}")
        print(f"             Mean: {pred_logits_cpu.mean().item():.3f}")
        print(f"             First query, first 10 classes: {pred_logits_cpu[0, :10].tolist()}")

        # SAVE TO FILE
        output_file = "logits_working_temp.pt"
        torch.save({
            'logits': pred_logits_cpu,
            'shape': pred_logits_cpu.shape,
            'first_query_logits': pred_logits_cpu[0, :].tolist(),
            'first_10_queries_first_10_classes': pred_logits_cpu[:10, :10].tolist()
        }, output_file)
        print(f"\n✅ SAVED RAW LOGITS TO: {output_file}")
        print(f"   Share this file for comparison!\n")
        # ═══════════════════════════════════════════════════════════════════

        return preds

    def postprocess_predictions(self, preds, orig_size, conf_thresh=0.0):
        """Postprocess predictions - EXACT copy from your working script"""
        pred_logits = preds['pred_logits']  # [1, num_queries, num_classes]
        pred_boxes = preds['pred_boxes']    # [1, num_queries, 4]

        pred_logits = pred_logits[0]  # [num_queries, num_classes]
        pred_boxes = pred_boxes[0]    # [num_queries, 4]

        # Convert logits to probabilities
        probs = torch.softmax(pred_logits, dim=-1)
        scores, labels = probs.max(dim=-1)

        # Filter by confidence
        keep = scores > conf_thresh
        scores = scores[keep]
        labels = labels[keep]
        pred_boxes = pred_boxes[keep]

        # Convert boxes from cxcywh to xyxy
        orig_w, orig_h = orig_size
        boxes = pred_boxes.clone()
        boxes[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * orig_w  # x1
        boxes[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * orig_h  # y1
        boxes[:, 2] = (pred_boxes[:, 0] + pred_boxes[:, 2] / 2) * orig_w  # x2
        boxes[:, 3] = (pred_boxes[:, 1] + pred_boxes[:, 3] / 2) * orig_h  # y2

        # Create results
        results = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.cpu().numpy()
            results.append({
                "label": self.class_names[label.item()],
                "score": score.item(),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

        return results


def main():
    print("="*70)
    print("WORKING SCRIPT WITH DIAGNOSTIC OUTPUT")
    print("="*70)

    # YOUR CLASS NAMES
    class_names = [
        "0_Stain", "1_Blemish", "2_Insect_Bite_Injury", "3_Brown_Spot",
        "4_Mold", "5_Chipped_Scratched_Peeling", "6_Immature", "7_Doubles",
        "8_Gum", "9_Perforated_Pinholes", "10_Black_Brown_Spot",
        "11_Scorched", "12_Dark_Brown_Shriveled", "13_Cracked",
        "14_Rust_Spot", "15_Broken", "16_Sliver", "17_Dirty", "18_Kernel_Rot",
        "19_Moldy_Gummy", "20_Discoloured", "21_Jump_Off", "22_Fancy",
        "23_Serious_Damage", "24_Light_Damage", "25_Light_Amber",
        "26_Medium_Amber", "27_Amber", "28_Particle", "29_Shell_Pieces",
        "30_Shell_Particle", "31_Foreign_Material", "32_Off_Skin",
        "33_Light_Insect_Damage", "34_Good"
    ]

    # ⚠️ IMPORTANT: UPDATE THESE PATHS
    model_path = "/Users/borde/label_studio_local/model_2.pt"  # YOUR MODEL PATH
    image_path = input("\nEnter path to test image: ").strip()

    print(f"\nModel: {model_path}")
    print(f"Image: {image_path}")

    # Initialize detector
    print("\n[1/4] Loading model...")
    detector = AlmondDefectDetector(model_path, class_names, device="cpu")

    # Preprocess
    print("\n[2/4] Preprocessing image...")
    image, tensor, orig_size = detector.preprocess_image(image_path)

    # Inference + SAVE RAW LOGITS
    print("\n[3/4] Running inference and saving raw logits...")
    preds = detector.predict(tensor)

    # Postprocess
    print("\n[4/4] Postprocessing...")
    conf_thresh = float(input("\nEnter confidence threshold (0.0 to show all predictions): ").strip())
    results = detector.postprocess_predictions(preds, orig_size, conf_thresh)

    # Display results
    print(f"\n{'='*70}")
    print(f"DETECTIONS: {len(results)} objects (confidence > {conf_thresh})")
    print(f"{'='*70}\n")

    for i, det in enumerate(results, 1):
        print(f"{i}. {det['label']}: {det['score']:.4f} at {det['bbox']}")

    # Save JSON
    output_json = "working_script_output.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved detections to: {output_json}")

    # RENAME THE LOGITS FILE TO MATCH IMAGE NAME
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    final_logits_name = f"logits_working_{base_name}.pt"

    if os.path.exists("logits_working_temp.pt"):
        os.rename("logits_working_temp.pt", final_logits_name)
        print(f"\n✅ Raw logits saved to: {final_logits_name}")
        print(f"\n" + "="*70)
        print(f"📤 NEXT STEPS:")
        print(f"   1. Find the file: {final_logits_name}")
        print(f"   2. Also get: logits_{base_name}.pt (from standalone script)")
        print(f"   3. Share BOTH files for comparison")
        print(f"="*70)


if __name__ == "__main__":
    main()
