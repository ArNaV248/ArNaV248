#!/usr/bin/env python3
"""
Complete Object Detection Pipeline using PyTorch

This script demonstrates a complete object detection pipeline including:
1. Preprocessing: Image loading, resizing, normalization
2. Inference: Running the model
3. Post-processing: NMS, extracting final bounding boxes, labels, and scores
"""

import torch
import torchvision
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
import numpy as np
import argparse
import os
from typing import Dict, List, Tuple


# COCO class names (91 classes, index 0 is background)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ObjectDetectionPipeline:
    """Complete object detection pipeline with preprocessing, inference, and post-processing."""

    def __init__(
        self,
        model_name: str = 'fasterrcnn_resnet50_fpn',
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize the object detection pipeline.

        Args:
            model_name: Name of the pretrained model to use
            confidence_threshold: Minimum confidence score for detections
            nms_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load pretrained model
        self.model = self._load_model(model_name)
        self.model.eval()

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _load_model(self, model_name: str) -> torch.nn.Module:
        """
        Load a pretrained object detection model.

        Args:
            model_name: Name of the model

        Returns:
            Loaded model
        """
        print(f"Loading model: {model_name}")

        if model_name == 'fasterrcnn_resnet50_fpn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                weights='DEFAULT'
            )
        elif model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=True,
                weights='DEFAULT'
            )
        elif model_name == 'retinanet_resnet50_fpn':
            model = torchvision.models.detection.retinanet_resnet50_fpn(
                pretrained=True,
                weights='DEFAULT'
            )
        elif model_name == 'ssd300_vgg16':
            model = torchvision.models.detection.ssd300_vgg16(
                pretrained=True,
                weights='DEFAULT'
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.to(self.device)
        return model

    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess the input image.

        Steps:
        1. Load image from disk
        2. Convert to RGB if necessary
        3. Convert to PyTorch tensor
        4. Normalize to [0, 1] range

        Args:
            image_path: Path to the input image

        Returns:
            Preprocessed tensor and original PIL image
        """
        print(f"Preprocessing image: {image_path}")

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()

        # Convert to tensor and normalize
        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        print(f"Image preprocessed: shape={image_tensor.shape}, dtype={image_tensor.dtype}")

        return image_tensor, original_image

    def inference(self, image_tensor: torch.Tensor) -> List[Dict]:
        """
        Run inference on the preprocessed image.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            List of predictions (one dict per image in batch)
        """
        print("Running inference...")

        with torch.no_grad():
            predictions = self.model(image_tensor)

        print(f"Inference complete: {len(predictions)} predictions")

        return predictions

    def post_process(
        self,
        predictions: List[Dict],
        image_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Post-process the model predictions.

        Steps:
        1. Extract boxes, scores, and labels
        2. Filter by confidence threshold
        3. Apply Non-Maximum Suppression (NMS)
        4. Convert to numpy arrays

        Args:
            predictions: Raw predictions from the model
            image_size: Original image size (width, height)

        Returns:
            Dictionary containing:
                - boxes: Final bounding boxes [N, 4] in format [x1, y1, x2, y2]
                - labels: Class labels [N]
                - scores: Confidence scores [N]
                - class_names: Human-readable class names [N]
        """
        print("Post-processing predictions...")

        # Get predictions for first image in batch
        pred = predictions[0]

        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()
        labels = pred['labels'].cpu()

        print(f"Raw predictions: {len(boxes)} detections")

        # Filter by confidence threshold
        keep_mask = scores >= self.confidence_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        print(f"After confidence filtering ({self.confidence_threshold}): {len(boxes)} detections")

        if len(boxes) == 0:
            print("No detections after confidence filtering")
            return {
                'boxes': np.array([]),
                'labels': np.array([]),
                'scores': np.array([]),
                'class_names': np.array([])
            }

        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []

        unique_labels = labels.unique()

        for label in unique_labels:
            # Get detections for this class
            label_mask = labels == label
            label_boxes = boxes[label_mask]
            label_scores = scores[label_mask]

            # Apply NMS
            keep_indices = nms(label_boxes, label_scores, self.nms_threshold)

            final_boxes.append(label_boxes[keep_indices])
            final_scores.append(label_scores[keep_indices])
            final_labels.append(torch.full((len(keep_indices),), label.item()))

        # Concatenate all classes
        if len(final_boxes) > 0:
            final_boxes = torch.cat(final_boxes, dim=0)
            final_scores = torch.cat(final_scores, dim=0)
            final_labels = torch.cat(final_labels, dim=0)
        else:
            final_boxes = torch.tensor([])
            final_scores = torch.tensor([])
            final_labels = torch.tensor([])

        print(f"After NMS ({self.nms_threshold}): {len(final_boxes)} detections")

        # Convert to numpy
        final_boxes = final_boxes.numpy()
        final_scores = final_scores.numpy()
        final_labels = final_labels.numpy().astype(int)

        # Get class names
        class_names = np.array([COCO_CLASSES[label] for label in final_labels])

        return {
            'boxes': final_boxes,
            'labels': final_labels,
            'scores': final_scores,
            'class_names': class_names
        }

    def detect(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Complete detection pipeline: preprocess -> inference -> post-process.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing final detections
        """
        # Preprocess
        image_tensor, original_image = self.preprocess(image_path)
        image_size = original_image.size

        # Inference
        predictions = self.inference(image_tensor)

        # Post-process
        results = self.post_process(predictions, image_size)

        return results

    def visualize_results(
        self,
        image_path: str,
        results: Dict[str, np.ndarray],
        output_path: str = None
    ):
        """
        Visualize detection results on the image.

        Args:
            image_path: Path to the input image
            results: Detection results from post_process
            output_path: Path to save the output image (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
            return

        # Load image
        image = Image.open(image_path)

        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        # Draw each detection
        for i in range(len(results['boxes'])):
            box = results['boxes'][i]
            label = results['class_names'][i]
            score = results['scores'][i]

            # Extract coordinates
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            label_text = f"{label}: {score:.2f}"
            ax.text(
                x1, y1 - 5,
                label_text,
                color='white',
                fontsize=10,
                bbox=dict(facecolor='red', alpha=0.7)
            )

        ax.axis('off')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()


def print_results(results: Dict[str, np.ndarray]):
    """Print detection results in a formatted way."""
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)

    if len(results['boxes']) == 0:
        print("No objects detected.")
        return

    print(f"Total detections: {len(results['boxes'])}\n")

    for i in range(len(results['boxes'])):
        box = results['boxes'][i]
        label = results['class_names'][i]
        score = results['scores'][i]

        print(f"Detection {i+1}:")
        print(f"  Class: {label}")
        print(f"  Confidence: {score:.4f}")
        print(f"  Bounding Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        print()


def main():
    """Main function to run the object detection pipeline."""
    parser = argparse.ArgumentParser(
        description='PyTorch Object Detection Pipeline'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='fasterrcnn_resnet50_fpn',
        choices=[
            'fasterrcnn_resnet50_fpn',
            'fasterrcnn_mobilenet_v3_large_fpn',
            'retinanet_resnet50_fpn',
            'ssd300_vgg16'
        ],
        help='Model to use for detection'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--nms',
        type=float,
        default=0.5,
        help='NMS IoU threshold'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output visualization'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = ObjectDetectionPipeline(
        model_name=args.model,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms,
        device=args.device
    )

    # Run detection
    results = pipeline.detect(args.image)

    # Print results
    print_results(results)

    # Visualize
    if not args.no_viz:
        pipeline.visualize_results(args.image, results, args.output)


if __name__ == '__main__':
    main()
