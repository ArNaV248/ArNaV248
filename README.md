# PyTorch Object Detection Pipeline

A complete, production-ready object detection pipeline using PyTorch and torchvision.

## Features

- **Preprocessing**: Automatic image loading, resizing, and normalization
- **Inference**: Support for multiple pretrained models (Faster R-CNN, RetinaNet, SSD)
- **Post-processing**:
  - Confidence-based filtering
  - Non-Maximum Suppression (NMS)
  - Clean output with bounding boxes, class labels, and confidence scores
- **Visualization**: Optional visualization of detection results

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python object_detection_pipeline.py --image path/to/your/image.jpg
```

### Advanced Usage

```bash
python object_detection_pipeline.py \
    --image path/to/image.jpg \
    --model fasterrcnn_resnet50_fpn \
    --confidence 0.7 \
    --nms 0.5 \
    --device cuda \
    --output results.jpg
```

### Available Arguments

- `--image`: Path to input image (required)
- `--model`: Model to use for detection (default: fasterrcnn_resnet50_fpn)
  - `fasterrcnn_resnet50_fpn`: Faster R-CNN with ResNet-50 FPN backbone
  - `fasterrcnn_mobilenet_v3_large_fpn`: Faster R-CNN with MobileNet V3 backbone
  - `retinanet_resnet50_fpn`: RetinaNet with ResNet-50 FPN backbone
  - `ssd300_vgg16`: SSD300 with VGG16 backbone
- `--confidence`: Confidence threshold for detections (default: 0.5)
- `--nms`: NMS IoU threshold (default: 0.5)
- `--device`: Device to run on (cuda/cpu, default: auto-detect)
- `--output`: Path to save visualization (optional)
- `--no-viz`: Skip visualization

## Using as a Library

```python
from object_detection_pipeline import ObjectDetectionPipeline

# Initialize pipeline
pipeline = ObjectDetectionPipeline(
    model_name='fasterrcnn_resnet50_fpn',
    confidence_threshold=0.5,
    nms_threshold=0.5
)

# Run detection
results = pipeline.detect('image.jpg')

# Access results
boxes = results['boxes']          # Bounding boxes [N, 4]
labels = results['labels']        # Class IDs [N]
scores = results['scores']        # Confidence scores [N]
class_names = results['class_names']  # Class names [N]

# Print detections
for i in range(len(boxes)):
    print(f"Detected {class_names[i]} with confidence {scores[i]:.2f}")
    print(f"  Box: {boxes[i]}")
```

## Pipeline Architecture

### 1. Preprocessing

The preprocessing step handles:
- Image loading from disk
- RGB conversion
- Tensor conversion
- Normalization to [0, 1] range
- Batch dimension addition
- Device transfer

### 2. Inference

The inference step:
- Runs the pretrained model on the preprocessed image
- Uses torch.no_grad() for efficiency
- Returns raw predictions including boxes, scores, and labels

### 3. Post-processing

The post-processing step includes:
- **Confidence filtering**: Removes low-confidence detections
- **Non-Maximum Suppression (NMS)**: Applied per-class to remove duplicate detections
- **Format conversion**: Converts tensors to numpy arrays
- **Class name mapping**: Maps class IDs to human-readable names

## Output Format

The pipeline returns a dictionary with:
- `boxes`: NumPy array of shape [N, 4] with bounding boxes in [x1, y1, x2, y2] format
- `labels`: NumPy array of shape [N] with class IDs
- `scores`: NumPy array of shape [N] with confidence scores
- `class_names`: NumPy array of shape [N] with human-readable class names

## Supported Classes

The models are trained on the COCO dataset with 80 object classes including:
- People and animals (person, dog, cat, horse, etc.)
- Vehicles (car, truck, bus, bicycle, etc.)
- Indoor objects (chair, couch, tv, laptop, etc.)
- Food items (pizza, sandwich, apple, etc.)
- And many more!

## Performance Notes

- **GPU Acceleration**: Automatically uses CUDA if available
- **Model Speed**:
  - Faster R-CNN ResNet-50: High accuracy, moderate speed
  - Faster R-CNN MobileNet V3: Lower accuracy, faster speed
  - RetinaNet ResNet-50: Balanced accuracy and speed
  - SSD300 VGG16: Fast inference, good for real-time applications

## License

This code uses pretrained models from torchvision which are subject to their respective licenses.
