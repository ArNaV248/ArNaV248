# Label Studio YOLOv5 ML Backend Documentation

**Project**: Almond Quality Defect Detection System
**Model**: YOLOv5
**Framework**: Label Studio ML Backend
**Date**: 2025-11-04

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Model Details](#model-details)
4. [Code Explanation](#code-explanation)
5. [Data Flow](#data-flow)
6. [Requirements & Dependencies](#requirements--dependencies)
7. [Setup Instructions](#setup-instructions)
8. [API Response Format](#api-response-format)
9. [Common Errors & Troubleshooting](#common-errors--troubleshooting)
10. [Configuration](#configuration)

---

## Overview

This is a **Label Studio Machine Learning Backend** that integrates **YOLOv5 object detection** for automated image annotation. The system is designed for **almond quality inspection**, detecting 35 different types of defects.

### Key Features:
- ✅ Real-time object detection using YOLOv5
- ✅ AWS S3 image storage integration
- ✅ 35 defect categories for almond quality control
- ✅ REST API backend for Label Studio
- ✅ GPU/CPU support with automatic device selection
- ✅ Half-precision (FP16) inference for faster processing

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Label Studio UI (http://127.0.0.1:8080/)                   │
│  - User uploads/views images                                 │
│  - Displays ML predictions as bounding boxes                 │
│  - Allows manual annotation corrections                      │
└────────────────┬───────────────────────▲─────────────────────┘
                 │                       │
                 │ HTTP POST Request     │ JSON Response
                 │ (task data)           │ (predictions)
                 ▼                       │
┌──────────────────────────────────────────────────────────────┐
│  ML Backend (Python Flask/Label Studio ML)                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. Receive task with S3 image URL                     │  │
│  │  2. Download image from S3 → base64 → PIL Image        │  │
│  │  3. Preprocess (letterbox, normalize, tensorize)       │  │
│  │  4. Run YOLOv5 inference (best.pt model)               │  │
│  │  5. Post-process (NMS, coordinate scaling)             │  │
│  │  6. Format predictions as Label Studio JSON            │  │
│  │  7. Return HTTP response                               │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────┬───────────────────────▲─────────────────────┘
                 │                       │
                 │ boto3.get_object()    │
                 ▼                       │
         ┌──────────────────┐           │
         │   AWS S3 Bucket  │           │
         │  (Image Storage) │           │
         │  s3://bucket/... │           │
         └──────────────────┘           │
                                        │
                          ┌─────────────┴────────────────┐
                          │  YOLOv5 Model (best.pt)      │
                          │  - 35 defect classes         │
                          │  - Image size: 1280px        │
                          │  - FP16/FP32 support         │
                          └──────────────────────────────┘
```

---

## Model Details

### Model Information
- **Architecture**: YOLOv5 (Ultralytics)
- **Model File**: `best.pt` (custom trained weights)
- **Input Size**: 1280x1280 pixels
- **Precision**: FP16 (half-precision) on GPU, FP32 on CPU
- **Inference Time**: ~20-50ms per image (GPU dependent)

### Defect Categories (35 Classes)

The model detects 35 different defect types in almonds:

| ID | Category Name | Description |
|----|--------------|-------------|
| 0 | 0_Adhering_Skin | Skin stuck to almond |
| 1 | 1_Blanched | Properly blanched almond |
| 2 | 2_Broken_Blanched | Broken blanched pieces |
| 3 | 3_Mission | Mission variety |
| 4 | 4_Carmel | Carmel variety |
| 5 | 5_Chip_Scratch_1_4 | Minor surface damage (≥1/4 size) |
| 6 | 6_Discolor | Color abnormality |
| 7 | 7_Doubles | Twin almonds |
| 8 | 8_Foreign_Material_Hull | Hull contamination |
| 9 | 9_NonPareil | NonPareil variety |
| 10 | 10_OD_Brownspot | Brownspot defect (other defect) |
| 11 | 11_SD_Insect_Damage | Insect damage (serious defect) |
| 12 | 12_Specks | Small spots/marks |
| 13 | 13_Split_Broken | Split or broken almonds |
| 14 | 14_LooseSkin_Dust_Particle | Loose skin or dust |
| 15 | 15_OD_Gummy | Gummy texture (other defect) |
| 16 | 16_SD_Pinhole | Pinhole damage (serious defect) |
| 17 | 17_Inshell | Still in shell |
| 18 | 18_Embeddedshell | Shell embedded in kernel |
| 19 | 19_FM_Other | Other foreign material |
| 20 | 20_FM_Rock_Dirtball | Rock or dirt contamination |
| 21 | 21_FM_Pistachio | Pistachio contamination |
| 22 | 22_FM_Walnut | Walnut contamination |
| 23 | 23_FM_Plastic | Plastic contamination |
| 24 | 24_FM_Metal | Metal contamination |
| 25 | 25_FM_Glass | Glass contamination |
| 26 | 26_SD_Mold | Mold (serious defect) |
| 27 | 27_SD_Decay | Decay (serious defect) |
| 28 | 28_SD_Other | Other serious defect |
| 29 | 29_SD_Frass | Frass/insect waste (serious defect) |
| 30 | 30_OD_Shrivel | Shriveled almonds |
| 31 | 31_OD_Discolor | Discoloration (other defect) |
| 32 | 32_Chip_Scratch_1_8 | Minor surface damage (≥1/8 size) |
| 33 | 33_California | California variety |
| 34 | 34_Fold | Folded/wrinkled almonds |

**Category Prefixes**:
- `OD_` = Other Defects
- `SD_` = Serious Defects
- `FM_` = Foreign Material

---

## Code Explanation

### Main Class: `Yolov5Model`

```python
class Yolov5Model(LabelStudioMLBase):
    def __init__(self, pretrained_model_path=pretrained_model_path, **kwargs):
        # Inherits from Label Studio ML Base class
```

**Inheritance**: Extends `LabelStudioMLBase` which provides:
- `predict()` method interface
- `fit()` method for active learning
- Caching mechanisms
- HTTP server capabilities

### Key Methods

#### 1. `__init__()` - Model Initialization

```python
def __init__(self, pretrained_model_path=pretrained_model_path, **kwargs):
    self.device = select_device('')  # Auto-select GPU/CPU
    self.model = attempt_load(pretrained_model_path, map_location=self.device)
    self.model.eval()  # Set to evaluation mode

    self.stride = int(self.model.stride.max())  # Model stride (usually 32)
    self.imgsz = check_img_size(1280, s=self.stride)  # Ensure size is multiple of stride
    self.half = self.device.type != 'cpu'  # Use FP16 only on GPU
```

**What happens**:
- Loads `best.pt` model weights
- Detects available GPU (CUDA) or falls back to CPU
- Enables half-precision (FP16) for 2x faster inference on GPU
- Sets image size to 1280 pixels

#### 2. `_get_image_from_s3()` - Image Download

```python
def _get_image_from_s3(self, s3_url: str) -> str:
    # Parse: s3://bucket-name/path/to/image.jpg
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc  # "bucket-name"
    key = parsed_url.path.lstrip('/')  # "path/to/image.jpg"

    # Download from S3
    _img_bytes = s3.get_object(Bucket=bucket_name, Key=key).get('Body').read()

    # Encode to base64
    return base64.b64encode(_img_bytes).decode('utf-8')
```

**What happens**:
- Parses S3 URL (e.g., `s3://my-bucket/images/photo.jpg`)
- Uses boto3 to download image bytes from AWS S3
- Converts to base64 string (though it's decoded immediately after)

#### 3. `predict()` - Main Inference Pipeline

**Step 1: Image Retrieval & Preprocessing**

```python
# Get image from S3
image_b64 = self._get_image_from_s3(image_url)

# Decode base64 → PIL Image → NumPy array
img_bytes = base64.b64decode(image_b64)
img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
im0 = np.array(img_pil)  # Original image (H, W, C)
original_height, original_width = im0.shape[:2]
```

**Step 2: Letterbox Resize**

```python
# Resize with aspect ratio preservation (adds gray padding)
im = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]
```

Example: 1920x1080 → 1280x720 with gray bars on sides

**Step 3: Tensor Conversion**

```python
# Convert to PyTorch tensor format
im = im.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
im = np.ascontiguousarray(im)  # Ensure memory contiguity

im = torch.from_numpy(im).to(self.device)
im = im.half() if self.half else im.float()  # FP16 or FP32
im /= 255.0  # Normalize to [0, 1]
if im.ndimension() == 3:
    im = im.unsqueeze(0)  # Add batch dimension: (C, H, W) → (1, C, H, W)
```

**Step 4: Model Inference**

```python
with torch.no_grad():  # Disable gradient computation
    pred = self.model(im, augment=False)[0]
```

Output shape: `(1, 25200, 40)` where:
- 25200 = number of predictions
- 40 = [x, y, w, h, confidence, class_0_prob, ..., class_34_prob]

**Step 5: Non-Maximum Suppression (NMS)**

```python
pred = non_max_suppression(
    pred,
    conf_thres=0.25,    # Minimum confidence threshold
    iou_thres=0.45,     # IoU threshold for NMS
    classes=None,        # Detect all classes
    agnostic=False,      # Class-specific NMS
    max_det=1000         # Maximum 1000 detections per image
)
```

**What NMS does**:
- Filters out predictions with confidence < 0.25
- Removes overlapping boxes (IoU > 0.45) keeping highest confidence
- Returns filtered detections

**Step 6: Coordinate Scaling**

```python
det = pred[0]
if len(det):
    # Scale from letterbox coordinates → original image coordinates
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
```

**Step 7: Format Results**

```python
for *xyxy, conf, cls in reversed(det):
    x1, y1, x2, y2 = xyxy
    label = category_map.get(int(cls), f'class_{int(cls)}')

    predictions.append({
        'value': {
            "x": float(x1) / original_width * 100,      # Convert to %
            "y": float(y1) / original_height * 100,
            "width": (float(x2) - float(x1)) / original_width * 100,
            "height": (float(y2) - float(y1)) / original_height * 100,
            "rectanglelabels": [label]
        },
        "score": float(conf)
    })
```

Converts pixel coordinates → percentages for Label Studio

#### 4. `fit()` - Training Hook (Placeholder)

```python
def fit(self, event, data, **kwargs):
    # Called when annotations are created/updated
    # Could be used for active learning / model retraining
    # Currently just caches data
    pass
```

**Events that trigger `fit()`**:
- `ANNOTATION_CREATED`: New annotation added
- `ANNOTATION_UPDATED`: Existing annotation modified
- `START_TRAINING`: Manual training trigger

**Potential uses**:
- Collect human-corrected annotations
- Retrain model periodically
- Active learning strategies

---

## Data Flow

### Request Flow (Prediction)

```
1. Label Studio UI
   ↓ POST /predict
   {
     "tasks": [
       {"data": {"image": "s3://my-bucket/img001.jpg"}}
     ]
   }

2. ML Backend receives request
   ↓ boto3.get_object()

3. AWS S3 returns image bytes
   ↓ Decode → PIL → NumPy → PyTorch Tensor

4. YOLOv5 Model
   ↓ Inference + NMS

5. Format predictions
   ↓ Return JSON

6. Label Studio UI
   ↓ Render bounding boxes on image
```

### Response Format

```json
[
  {
    "result": [
      {
        "name": "11_SD_Insect_Damage",
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": 1920,
        "original_height": 1080,
        "image_rotation": 0,
        "value": {
          "rotation": 0,
          "x": 45.2,
          "y": 30.1,
          "width": 10.5,
          "height": 8.3,
          "rectanglelabels": ["11_SD_Insect_Damage"]
        },
        "score": 0.856
      }
    ],
    "model_version": "v1"
  }
]
```

**Important**:
- Coordinates are in **percentages** (0-100), not pixels
- `x`, `y` = top-left corner of bounding box
- `width`, `height` = box dimensions as % of image size

---

## Requirements & Dependencies

### Complete `requirements.txt`

```txt
# Core ML/DL frameworks
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0

# Image processing
Pillow>=9.0.0
opencv-python>=4.5.0

# Data handling
pandas>=1.3.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# YOLOv5 dependencies
scipy>=1.7.0
tqdm>=4.62.0
PyYAML>=5.4.0

# Label Studio ML Backend
label-studio-ml>=1.0.9

# AWS S3 integration
boto3>=1.24.0

# HTTP/API
requests>=2.26.0

# Optional but recommended for YOLOv5
# tensorboard>=2.11.0  # for training visualization
# thop>=0.1.1  # for FLOPs calculation
```

### Dependency Breakdown

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | >=2.0.0 | PyTorch deep learning framework |
| **torchvision** | >=0.15.0 | Image transformations and utilities |
| **numpy** | >=1.21.0 | Numerical operations |
| **Pillow** | >=9.0.0 | Image loading (PIL) |
| **opencv-python** | >=4.5.0 | Image preprocessing (letterbox, augmentations) |
| **pandas** | >=1.3.0 | Data handling (imported in code) |
| **matplotlib** | >=3.5.0 | Plotting/visualization |
| **seaborn** | >=0.11.0 | Statistical visualizations |
| **scipy** | >=1.7.0 | Scientific computing (used by YOLOv5 NMS) |
| **tqdm** | >=4.62.0 | Progress bars |
| **PyYAML** | >=5.4.0 | YAML config parsing for YOLOv5 |
| **label-studio-ml** | >=1.0.9 | **CRITICAL**: ML backend framework |
| **boto3** | >=1.24.0 | **CRITICAL**: AWS S3 client |
| **requests** | >=2.26.0 | HTTP requests |

### Additional Requirement: YOLOv5 Repository

Your code imports directly from YOLOv5 source:

```python
YOLOV5_ROOT = "/app/yolov5"
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device
```

**You must clone the repository**:

```bash
git clone https://github.com/ultralytics/yolov5.git /app/yolov5
```

**Or** modify code to use the pip package instead.

---

## Setup Instructions

### 1. Clone YOLOv5 Repository

```bash
# Option A: Clone to /app/yolov5
sudo mkdir -p /app
cd /app
git clone https://github.com/ultralytics/yolov5.git yolov5

# Option B: Clone to project directory
cd /home/user/ArNaV248
git clone https://github.com/ultralytics/yolov5.git yolov5
# Then update YOLOV5_ROOT in code to: "/home/user/ArNaV248/yolov5"
```

### 2. Install Dependencies

```bash
cd /home/user/ArNaV248
pip install -r requirements.txt
```

### 3. Configure AWS Credentials

For boto3 S3 access:

```bash
# Option A: AWS CLI
aws configure
# Enter: Access Key ID, Secret Access Key, Region

# Option B: Environment variables
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"

# Option C: IAM Role (if running on EC2)
# No configuration needed - automatic
```

### 4. Place Model Weights

```bash
# Copy your trained YOLOv5 model to project root
cp /path/to/your/best.pt /home/user/ArNaV248/best.pt

# Or update pretrained_model_path variable in code
```

### 5. Configure Label Studio Connection

Update constants in code:

```python
LABEL_STUDIO_URL = 'http://127.0.0.1:8080/'  # Your Label Studio URL
LABEL_STUDIO_API_KEY = 'your_api_key_here'   # Get from Label Studio settings
```

### 6. Start the ML Backend

```bash
# Method 1: Using label-studio-ml
label-studio-ml start /home/user/ArNaV248

# Method 2: Using Python directly
python your_ml_backend_file.py
```

### 7. Connect to Label Studio

1. Open Label Studio UI: http://127.0.0.1:8080/
2. Go to Settings → Machine Learning
3. Add ML Backend:
   - URL: `http://localhost:9090` (default ML backend port)
   - Click "Validate and Save"

### 8. Test Prediction

1. Upload image to Label Studio
2. Open labeling interface
3. ML predictions should appear automatically as bounding boxes

---

## API Response Format

### Prediction Response Structure

```python
[
  {
    "result": [
      # List of predictions (bounding boxes)
    ],
    "model_version": "v1"
  }
]
```

### Single Prediction Object

```python
{
    "name": "11_SD_Insect_Damage",           # Display name
    "[showInline]": "true",                  # Show in UI
    "from_name": "label",                    # Label config reference
    "to_name": "image",                      # Image field reference
    "type": "rectanglelabels",               # Annotation type
    "original_width": 1920,                  # Original image width (px)
    "original_height": 1080,                 # Original image height (px)
    "image_rotation": 0,                     # Rotation angle
    "value": {
        "rotation": 0,
        "x": 45.2,                           # X position (%)
        "y": 30.1,                           # Y position (%)
        "width": 10.5,                       # Box width (%)
        "height": 8.3,                       # Box height (%)
        "rectanglelabels": ["11_SD_Insect_Damage"]
    },
    "score": 0.856                           # Confidence score (0-1)
}
```

### Coordinate System

**IMPORTANT**: All coordinates are percentages (0-100), not pixels!

```
Original Image: 1920x1080 pixels
Bounding Box: (868, 325) to (1070, 415) pixels

Converted to percentages:
x = 868 / 1920 * 100 = 45.2%
y = 325 / 1080 * 100 = 30.1%
width = (1070 - 868) / 1920 * 100 = 10.5%
height = (415 - 325) / 1080 * 100 = 8.3%
```

---

## Common Errors & Troubleshooting

### 1. `FileNotFoundError: Model file not found at best.pt`

**Error**:
```
FileNotFoundError: Model file not found at best.pt
```

**Cause**: Model weights file is missing

**Solution**:
```bash
# Check if file exists
ls -lh /home/user/ArNaV248/best.pt

# If missing, copy from training directory
cp /path/to/yolov5/runs/train/exp/weights/best.pt /home/user/ArNaV248/

# Or update path in code
pretrained_model_path = "/full/path/to/best.pt"
```

---

### 2. `ModuleNotFoundError: No module named 'label_studio_ml'`

**Error**:
```
ModuleNotFoundError: No module named 'label_studio_ml'
```

**Cause**: Missing Label Studio ML package

**Solution**:
```bash
pip install label-studio-ml>=1.0.9
```

---

### 3. `ModuleNotFoundError: No module named 'models'`

**Error**:
```
ModuleNotFoundError: No module named 'models'
```

**Cause**: YOLOv5 repository not cloned or path not set

**Solution**:
```bash
# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5.git /app/yolov5

# Verify path in code matches
YOLOV5_ROOT = "/app/yolov5"  # Must match clone location
```

---

### 4. `botocore.exceptions.NoCredentialsError`

**Error**:
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Cause**: AWS credentials not configured

**Solution**:
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_DEFAULT_REGION="us-east-1"

# Option 3: Credentials file
mkdir -p ~/.aws
cat > ~/.aws/credentials <<EOF
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
EOF
```

---

### 5. `RuntimeError: CUDA out of memory`

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
```

**Cause**: GPU memory exhausted

**Solution**:
```python
# Option 1: Force CPU inference
self.device = select_device('cpu')

# Option 2: Reduce image size
self.imgsz = check_img_size(640, s=self.stride)  # Instead of 1280

# Option 3: Disable half-precision
self.half = False

# Option 4: Process images in smaller batches
```

---

### 6. `cv2.error: OpenCV(4.x) error`

**Error**:
```
cv2.error: OpenCV(4.x.x) .../resize.cpp:xxx: error: (-215:Assertion failed)
```

**Cause**: Missing opencv-python or version mismatch

**Solution**:
```bash
# Uninstall all OpenCV variants
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y

# Reinstall correct version
pip install opencv-python>=4.5.0
```

---

### 7. `Connection refused` when connecting to Label Studio

**Error**:
```
requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionRefusedError(111, 'Connection refused'))
```

**Cause**: ML backend or Label Studio not running

**Solution**:
```bash
# Start Label Studio
label-studio start

# Start ML Backend (in separate terminal)
label-studio-ml start /home/user/ArNaV248

# Check ports
netstat -tulpn | grep -E '8080|9090'
```

---

### 8. No predictions showing in Label Studio

**Possible causes**:

**A. ML Backend not connected**
```bash
# In Label Studio UI:
# Settings → Machine Learning → Check if backend is "Connected"
# If not, verify URL and click "Validate and Save"
```

**B. Confidence threshold too high**
```python
# Lower confidence threshold in code
pred = non_max_suppression(
    pred,
    conf_thres=0.1,  # Lower from 0.25 to 0.1
    iou_thres=0.45,
    ...
)
```

**C. Wrong labeling config**
```xml
<!-- Label Studio config should match code -->
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="0_Adhering_Skin"/>
    <Label value="1_Blanched"/>
    <!-- ... all 35 labels ... -->
  </RectangleLabels>
</View>
```

---

### 9. S3 Access Denied

**Error**:
```
botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the GetObject operation: Access Denied
```

**Cause**: Insufficient S3 permissions

**Solution**:
```json
// Add this IAM policy to your AWS user/role
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    }
  ]
}
```

---

### 10. Predictions are offset/wrong size

**Cause**: Coordinate conversion error

**Debug**:
```python
# Add logging to check coordinates
print(f"Original image: {original_width}x{original_height}")
print(f"Letterbox image: {im.shape}")
print(f"Detection (pixels): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
print(f"Detection (percent): x={x/100*original_width}, y={y/100*original_height}")
```

**Solution**: Ensure `scale_coords()` is called correctly to map letterbox → original coordinates

---

## Configuration

### Environment Variables

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"

# Label Studio
export LABEL_STUDIO_URL="http://127.0.0.1:8080/"
export LABEL_STUDIO_API_KEY="your_api_key"

# Model Configuration
export MODEL_PATH="/home/user/ArNaV248/best.pt"
export YOLOV5_ROOT="/app/yolov5"
export CONFIDENCE_THRESHOLD="0.25"
export IOU_THRESHOLD="0.45"
export IMAGE_SIZE="1280"
```

### Code Configuration Points

```python
# File: your_ml_backend.py

# --- Model Settings ---
pretrained_model_path = "best.pt"           # Path to YOLOv5 weights
YOLOV5_ROOT = "/app/yolov5"                 # YOLOv5 repo location

# --- Label Studio Connection ---
LABEL_STUDIO_URL = 'http://127.0.0.1:8080/'
LABEL_STUDIO_API_KEY = '100926765214fd262cf9243b620e2ec72c9219ad'

# --- Inference Settings ---
self.imgsz = check_img_size(1280, s=self.stride)  # Image size (640, 1280, etc.)
conf_thres = 0.25                                  # Confidence threshold
iou_thres = 0.45                                   # NMS IoU threshold
max_det = 1000                                     # Max detections per image

# --- Device Selection ---
self.device = select_device('')              # '' = auto, 'cpu' = force CPU, '0' = GPU 0
self.half = self.device.type != 'cpu'        # Use FP16 on GPU
```

---

## Performance Optimization

### 1. Batch Processing

Current code processes one image at a time. For better throughput:

```python
# Process multiple images in batch
im = torch.cat([im1, im2, im3], dim=0)  # Shape: (3, 3, 1280, 1280)
pred = self.model(im, augment=False)
```

### 2. TensorRT Optimization

For production deployment:

```bash
# Export to TensorRT
python export.py --weights best.pt --include engine --device 0 --half

# Load TensorRT model in code
self.model = attempt_load('best.engine')
```

### 3. Image Size Trade-offs

| Size | Speed | Accuracy | Memory |
|------|-------|----------|--------|
| 640 | Fast | Good | Low |
| 1280 | Medium | Better | Medium |
| 1920 | Slow | Best | High |

### 4. Caching

Add caching for frequently accessed images:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _get_image_from_s3_cached(self, s3_url: str):
    return self._get_image_from_s3(s3_url)
```

---

## Summary

### What This System Does

1. ✅ **Receives** image URLs from Label Studio
2. ✅ **Downloads** images from AWS S3
3. ✅ **Runs** YOLOv5 object detection
4. ✅ **Detects** 35 types of almond defects
5. ✅ **Returns** predictions to Label Studio as JSON
6. ✅ **Displays** bounding boxes in Label Studio UI

### What This System Does NOT Do

- ❌ Save visualization images to disk
- ❌ Save JSON predictions to files
- ❌ Automatically retrain the model (fit() is placeholder)
- ❌ Store images locally (streams from S3)

### Key Files

| File | Purpose |
|------|---------|
| `best.pt` | YOLOv5 trained model weights |
| `requirements.txt` | Python dependencies |
| `your_ml_backend.py` | Main ML backend code |
| `/app/yolov5/` | YOLOv5 source code |

---

## Additional Resources

- [Label Studio ML Backend Documentation](https://labelstud.io/guide/ml.html)
- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
- [AWS boto3 S3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Last Updated**: 2025-11-04
**Author**: Generated with Claude Code
**Version**: 1.0
