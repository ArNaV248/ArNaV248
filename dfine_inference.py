#!/usr/bin/env python3
"""
D-FINE Object Detection Inference Script
Single-file inference for D-FINE model trained on almond defect detection.

Model Details:
- Architecture: DFINE-L (Large)
- Input Size: 1280x448
- Classes: 35 almond defect types
- Model Path: /Users/borde/arnav/model_2.pt
- DFINE Source: /Users/borde/label_studio_local/src/d_fine/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import nms, box_convert
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model configuration
MODEL_PATH = "/Users/borde/arnav/model_2.pt"
DFINE_SOURCE_PATH = "/Users/borde/label_studio_local/src/d_fine/"
MODEL_NAME = "l"  # DFINE-Large
INPUT_SIZE = (1280, 448)  # width, height
NUM_CLASSES = 35

# Class names for almond defects
CLASS_NAMES = [
    "0_Adhering_Skin", "1_Blanched", "2_Broken_Blanched", "3_Mission", "4_Carmel",
    "5_Chip_Scratch_1_4", "6_Discolor", "7_Doubles", "8_Foreign_Material_Hull",
    "9_NonPareil", "10_OD_Brownspot", "11_SD_Insect_Damage", "12_Specks",
    "13_Split_Broken", "14_LooseSkin_Dust_Particle", "15_OD_Gummy", "16_SD_Pinhole",
    "17_Inshell", "18_Embeddedshell", "19_FM_Other", "20_FM_Rock_Dirtball",
    "21_FM_Pistachio", "22_FM_Walnut", "23_FM_Plastic", "24_FM_Metal",
    "25_FM_Glass", "26_SD_Mold", "27_SD_Decay", "28_SD_Other", "29_SD_Frass",
    "30_OD_Shrivel", "31_OD_Discolor", "32_Chip_Scratch_1_8", "33_California",
    "34_Fold_Deformed",
]

# Output directories
VIS_DIR = "visualize"
JSON_DIR = "json"

# ==============================================================================
# DEVICE SELECTION
# ==============================================================================

def select_device():
    """
    Select the best available device.
    Priority: cuda:1 (if 2+ GPUs) > cuda:0 (if 1 GPU) > cpu
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            device = torch.device("cuda:1")
            print(f"[Device] Using cuda:1 (detected {gpu_count} GPUs)")
        else:
            device = torch.device("cuda:0")
            print(f"[Device] Using cuda:0 (detected {gpu_count} GPU)")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (no GPU detected)")

    return device

# ==============================================================================
# MODEL LOADING
# ==============================================================================

def setup_dfine_import():
    """Add DFINE source directory to Python path."""
    if os.path.exists(DFINE_SOURCE_PATH):
        parent_dir = os.path.dirname(DFINE_SOURCE_PATH.rstrip('/'))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        print(f"[Import] Added to sys.path: {parent_dir}")
        return True
    else:
        print(f"[Error] DFINE source not found at: {DFINE_SOURCE_PATH}")
        return False

def load_model(device):
    """
    Load the D-FINE model with trained weights.

    Returns:
        model: Loaded D-FINE model ready for inference
    """
    print("\n" + "="*70)
    print("LOADING D-FINE MODEL")
    print("="*70)

    # Setup import path
    if not setup_dfine_import():
        print("[Error] Cannot import D-FINE. Please check DFINE_SOURCE_PATH.")
        sys.exit(1)

    try:
        # Import D-FINE components
        from src.d_fine.dfine import build_model
        from src.d_fine.utils import load_tuning_state
        print("[Import] Successfully imported D-FINE modules")
    except ImportError as e:
        print(f"[Error] Failed to import D-FINE: {e}")
        print("\nPlease ensure the following files exist:")
        print(f"  - {DFINE_SOURCE_PATH}dfine.py")
        print(f"  - {DFINE_SOURCE_PATH}utils.py")
        print(f"  - {DFINE_SOURCE_PATH}configs.py")
        print(f"  - {DFINE_SOURCE_PATH}arch/hgnetv2.py")
        print(f"  - {DFINE_SOURCE_PATH}arch/hybrid_encoder.py")
        print(f"  - {DFINE_SOURCE_PATH}arch/dfine_decoder.py")
        sys.exit(1)

    # Build model
    print(f"[Model] Building D-FINE-{MODEL_NAME.upper()} model...")
    model = build_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        device=device,
        img_size=INPUT_SIZE,
        pretrained_model_path=None
    )
    print(f"[Model] Model architecture built successfully")

    # Load checkpoint
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model checkpoint not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"[Model] Loading checkpoint from: {MODEL_PATH}")

    try:
        # Method 1: Try load_tuning_state (handles various checkpoint formats)
        model = load_tuning_state(model, MODEL_PATH)
        print("[Model] Checkpoint loaded successfully using load_tuning_state")
    except Exception as e1:
        print(f"[Warning] load_tuning_state failed: {e1}")
        print("[Model] Attempting direct state_dict loading...")

        try:
            # Method 2: Direct state_dict loading
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load with strict=False to allow minor mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"[Warning] Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"[Warning] Unexpected keys: {len(unexpected_keys)}")

            print("[Model] Checkpoint loaded successfully (direct method)")

        except Exception as e2:
            print(f"[Error] Failed to load checkpoint: {e2}")
            sys.exit(1)

    model = model.to(device)
    model.eval()

    print(f"[Model] Model ready for inference on {device}")
    print("="*70 + "\n")

    return model

# ==============================================================================
# PREPROCESSING
# ==============================================================================

def preprocess_image(image_path: str, target_size: Tuple[int, int] = INPUT_SIZE):
    """
    Preprocess image for D-FINE inference.

    Args:
        image_path: Path to input image
        target_size: Target size (width, height)

    Returns:
        tensor: Preprocessed image tensor [1, 3, H, W]
        original_image: Original PIL image
        original_size: Original image size (width, height)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Resize to model input size
    image_resized = image.resize(target_size, Image.BILINEAR)

    # Convert to tensor and normalize to [0, 1]
    transform = T.Compose([
        T.ToTensor(),  # Converts to [0, 1] and changes to CHW format
    ])

    tensor = transform(image_resized)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    return tensor, image, original_size

# ==============================================================================
# INFERENCE
# ==============================================================================

@torch.no_grad()
def run_inference(model, image_tensor, device):
    """
    Run D-FINE inference on preprocessed image.

    Args:
        model: D-FINE model
        image_tensor: Preprocessed image tensor
        device: Device to run on

    Returns:
        outputs: Model output dictionary
    """
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    return outputs

# ==============================================================================
# POST-PROCESSING
# ==============================================================================

def postprocess_outputs(
    outputs: Dict,
    original_size: Tuple[int, int],
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-process D-FINE outputs to get final detections.

    D-FINE typically outputs:
    - pred_logits: [batch, num_queries, num_classes]
    - pred_boxes: [batch, num_queries, 4] in normalized cxcywh format

    Args:
        outputs: Model outputs
        original_size: Original image size (width, height)
        confidence_threshold: Minimum confidence score
        nms_threshold: NMS IoU threshold

    Returns:
        boxes: Final boxes in xyxy format, scaled to original image [N, 4]
        scores: Confidence scores [N]
        labels: Class labels [N]
    """
    # Extract predictions
    pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes]
    pred_boxes = outputs['pred_boxes']    # [batch, num_queries, 4]

    # Handle multi-level outputs (if decoder returns list of predictions)
    if isinstance(pred_logits, (list, tuple)):
        pred_logits = pred_logits[-1]  # Use final decoder layer
    if isinstance(pred_boxes, (list, tuple)):
        pred_boxes = pred_boxes[-1]

    # Remove batch dimension
    pred_logits = pred_logits[0]  # [num_queries, num_classes]
    pred_boxes = pred_boxes[0]    # [num_queries, 4]

    # Get confidence scores and labels
    scores = F.softmax(pred_logits, dim=-1)  # [num_queries, num_classes]
    max_scores, labels = scores.max(dim=-1)  # [num_queries]

    # Filter by confidence threshold
    keep_mask = max_scores > confidence_threshold

    if keep_mask.sum() == 0:
        # No detections
        return np.array([]), np.array([]), np.array([])

    filtered_boxes = pred_boxes[keep_mask]    # [N, 4]
    filtered_scores = max_scores[keep_mask]   # [N]
    filtered_labels = labels[keep_mask]       # [N]

    # Convert boxes from normalized cxcywh to xyxy format
    # D-FINE boxes are typically in normalized [cx, cy, w, h] format
    boxes_cxcywh = filtered_boxes.cpu()

    # Convert cxcywh to xyxy
    boxes_xyxy = box_convert(boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')

    # Scale to original image size
    img_w, img_h = original_size
    boxes_xyxy[:, [0, 2]] *= img_w  # x coordinates
    boxes_xyxy[:, [1, 3]] *= img_h  # y coordinates

    # Clip boxes to image boundaries
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, img_w)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, img_h)

    # Apply NMS
    keep_indices = nms(boxes_xyxy, filtered_scores.cpu(), nms_threshold)

    final_boxes = boxes_xyxy[keep_indices].numpy()
    final_scores = filtered_scores[keep_indices].cpu().numpy()
    final_labels = filtered_labels[keep_indices].cpu().numpy()

    return final_boxes, final_scores, final_labels

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_detections(
    image_path: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: str
):
    """
    Draw bounding boxes on image and save.

    Args:
        image_path: Path to original image
        boxes: Bounding boxes in xyxy format [N, 4]
        scores: Confidence scores [N]
        labels: Class labels [N]
        output_path: Path to save visualization
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Try to load a nice font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # Prepare label text
        class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class_{int(label)}"
        text = f"{class_name}: {score*100:.1f}%"

        # Draw text background
        try:
            bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle(bbox, fill='red')
            draw.text((x1, y1 - 20), text, fill='white', font=font)
        except:
            # Fallback for older PIL versions
            draw.text((x1, y1 - 20), text, fill='yellow', font=font)

    # Save visualization
    image.save(output_path)
    print(f"[Output] Saved visualization: {output_path}")

# ==============================================================================
# JSON OUTPUT
# ==============================================================================

def save_json_output(
    image_name: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: str
):
    """
    Save detection results as JSON.

    Args:
        image_name: Name of the image file
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        labels: Class labels [N]
        output_path: Path to save JSON file
    """
    detections = []

    for box, score, label in zip(boxes, scores, labels):
        class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class_{int(label)}"

        detection = {
            "class": class_name,
            "class_id": int(label),
            "confidence": float(score),
            "bbox": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            }
        }
        detections.append(detection)

    output = {
        "image": image_name,
        "num_detections": len(detections),
        "detections": detections
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[Output] Saved JSON: {output_path}")

# ==============================================================================
# MAIN INFERENCE PIPELINE
# ==============================================================================

def process_single_image(
    model,
    device,
    image_path: str,
    output_dir: str,
    confidence_threshold: float,
    nms_threshold: float,
    save_vis: bool = True,
    save_json: bool = True
):
    """
    Process a single image through the complete pipeline.

    Args:
        model: D-FINE model
        device: Device to run on
        image_path: Path to input image
        output_dir: Directory to save outputs
        confidence_threshold: Confidence threshold for detections
        nms_threshold: NMS IoU threshold
        save_vis: Whether to save visualization
        save_json: Whether to save JSON output
    """
    print(f"\n[Processing] {image_path}")

    # Preprocess
    image_tensor, original_image, original_size = preprocess_image(image_path, INPUT_SIZE)

    # Inference
    outputs = run_inference(model, image_tensor, device)

    # Post-process
    boxes, scores, labels = postprocess_outputs(
        outputs,
        original_size,
        confidence_threshold,
        nms_threshold
    )

    print(f"[Results] Found {len(boxes)} detections")

    # Prepare output paths
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    # Save visualization
    if save_vis:
        vis_path = os.path.join(output_dir, VIS_DIR, image_name)
        visualize_detections(image_path, boxes, scores, labels, vis_path)

    # Save JSON
    if save_json:
        json_path = os.path.join(output_dir, JSON_DIR, f"{base_name}.json")
        save_json_output(image_name, boxes, scores, labels, json_path)

    # Print detections
    if len(boxes) > 0:
        print(f"\nDetections:")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels), 1):
            class_name = CLASS_NAMES[int(label)]
            print(f"  {i}. {class_name}: {score*100:.1f}% at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

def process_directory(
    model,
    device,
    input_dir: str,
    output_dir: str,
    confidence_threshold: float,
    nms_threshold: float,
    save_vis: bool = True,
    save_json: bool = True
):
    """
    Process all images in a directory.

    Args:
        model: D-FINE model
        device: Device to run on
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        confidence_threshold: Confidence threshold
        nms_threshold: NMS threshold
        save_vis: Whether to save visualizations
        save_json: Whether to save JSON outputs
    """
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []

    for file_name in sorted(os.listdir(input_dir)):
        if os.path.splitext(file_name)[1].lower() in image_extensions:
            image_files.append(os.path.join(input_dir, file_name))

    print(f"\n[Directory] Found {len(image_files)} images in {input_dir}")

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*70}")
        print(f"Image {i}/{len(image_files)}")
        print(f"{'='*70}")

        try:
            process_single_image(
                model,
                device,
                image_path,
                output_dir,
                confidence_threshold,
                nms_threshold,
                save_vis,
                save_json
            )
        except Exception as e:
            print(f"[Error] Failed to process {image_path}: {e}")
            import traceback
            traceback.print_exc()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-FINE Object Detection Inference for Almond Defect Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python dfine_inference.py --input /path/to/image.jpg --conf 0.3

  # Process directory
  python dfine_inference.py --input /path/to/images/ --conf 0.3 --nms 0.5

  # Save only JSON (no visualization)
  python dfine_inference.py --input image.jpg --no-vis
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image or directory'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )

    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.3,
        help='Confidence threshold (default: 0.3)'
    )

    parser.add_argument(
        '--nms', '-n',
        type=float,
        default=0.5,
        help='NMS IoU threshold (default: 0.5)'
    )

    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='Skip saving visualizations'
    )

    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip saving JSON outputs'
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.join(args.output, VIS_DIR), exist_ok=True)
    os.makedirs(os.path.join(args.output, JSON_DIR), exist_ok=True)

    # Select device
    device = select_device()

    # Load model
    model = load_model(device)

    # Process input
    if os.path.isfile(args.input):
        # Single image
        process_single_image(
            model,
            device,
            args.input,
            args.output,
            args.conf,
            args.nms,
            save_vis=not args.no_vis,
            save_json=not args.no_json
        )
    elif os.path.isdir(args.input):
        # Directory
        process_directory(
            model,
            device,
            args.input,
            args.output,
            args.conf,
            args.nms,
            save_vis=not args.no_vis,
            save_json=not args.no_json
        )
    else:
        print(f"[Error] Input path not found: {args.input}")
        sys.exit(1)

    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print(f"Outputs saved to: {args.output}")
    if not args.no_vis:
        print(f"  Visualizations: {os.path.join(args.output, VIS_DIR)}")
    if not args.no_json:
        print(f"  JSON files: {os.path.join(args.output, JSON_DIR)}")
    print()

if __name__ == '__main__':
    main()
