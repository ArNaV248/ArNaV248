# D-FINE Label Studio ML Backend

Complete Label Studio integration for your custom D-FINE object detection model.

## 🔄 Workflow

```
┌─────────────────┐
│  Label Studio   │
│  (Port 8080)    │
└────────┬────────┘
         │ 1. Sends task with S3 image URL
         ▼
┌─────────────────┐
│  This ML Backend│
│  (Port 9090)    │
│  DFINE-L Model  │
└────────┬────────┘
         │ 2. Downloads from S3
         ▼
┌─────────────────┐
│   Amazon S3     │
│   (Images)      │
└────────┬────────┘
         │ 3. Returns image bytes
         ▼
┌─────────────────┐
│  D-FINE Inference│
│  35 Almond      │
│  Defect Classes │
└────────┬────────┘
         │ 4. Returns predictions (JSON with bboxes)
         ▼
┌─────────────────┐
│  Label Studio   │
│  UI Displays    │
│  Bounding Boxes │
└─────────────────┘
```

**Key Points:**
- ✅ NO file outputs (no visualize/, no json/)
- ✅ Everything rendered in Label Studio UI
- ✅ Images loaded from S3
- ✅ Same D-FINE model as standalone script

---

## 📋 What This Script Does

### **Model Used:**
- **Architecture:** D-FINE-L (Large) with HGNetv2 backbone
- **Model File:** `/Users/borde/arnav/model_2.pt` (⚠️ Update path in script line 41)
- **Input Size:** 1280 x 448
- **Classes:** 35 almond defect categories

### **Image Source:**
- Downloads from **Amazon S3** via boto3
- Format: `s3://bucket-name/path/to/image.jpg`

### **Inference:**
1. Downloads image from S3
2. Preprocesses: PIL → Resize(1280x448) → Tensor
3. Runs D-FINE inference
4. Applies confidence filtering (default: 0.3)
5. Applies NMS if enabled (default: disabled)
6. Converts boxes to Label Studio format (percentages)

### **Output Destination:**
- **Returns JSON to Label Studio** (NOT files!)
- Label Studio displays bounding boxes in UI
- No `visualize/` or `json/` folders created

---

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install label-studio-ml torch torchvision pillow boto3 numpy
```

### 2. Configure AWS S3

Make sure your AWS credentials are configured:

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your region (e.g., us-east-1)
```

### 3. Update Model Path

Edit `dfine_labelstudio_backend.py` line 41:

```python
MODEL_PATH = "/Users/borde/arnav/model_2.pt"  # ← CHANGE THIS PATH
```

### 4. Update Label Studio Settings (Optional)

Edit lines 37-38 if your Label Studio is on a different host/port:

```python
LABEL_STUDIO_URL = 'http://127.0.0.1:8080/'
LABEL_STUDIO_API_KEY = '100926765214fd262cf9243b620e2ec72c9219ad'
```

---

## 🎯 Usage

### Start the ML Backend

```bash
label-studio-ml start dfine_labelstudio_backend --port 9090
```

Output:
```
[INFO] Initializing D-FINE model from /Users/borde/arnav/model_2.pt
[INFO] Using CUDA device: NVIDIA GeForce RTX 3090
[INFO] Loading checkpoint from: /Users/borde/arnav/model_2.pt
[INFO] Checkpoint loaded successfully
[INFO] Model initialized successfully
[INFO] Input size: (1280, 448) (W x H)
[INFO] Number of classes: 35
✓ ML Backend running on http://0.0.0.0:9090
```

### Connect Label Studio

1. Open Label Studio UI: http://localhost:8080
2. Go to **Settings → Machine Learning**
3. Add ML Backend:
   - URL: `http://localhost:9090`
   - Click "Validate and Save"

4. Import tasks with S3 image URLs:
   ```json
   {
     "data": {
       "image": "s3://my-bucket/almonds/image001.jpg"
     }
   }
   ```

5. Label Studio will automatically:
   - Send image URL to ML backend
   - Receive predictions
   - Display bounding boxes in UI

---

## 🎨 Inference Parameters

### Confidence Threshold (default: 0.3)

Control detection sensitivity via Label Studio API or context:

```python
# In Label Studio ML backend settings:
context = {
    'confidence_threshold': 0.5  # Only show high-confidence detections
}
```

**Recommended values:**
- `0.05-0.10`: Maximum detections (many false positives)
- `0.20-0.30`: Balanced (default)
- `0.50+`: Only very confident detections

### NMS Threshold (default: 1.0 = disabled)

Remove duplicate/overlapping boxes:

```python
context = {
    'nms_threshold': 0.5  # Remove boxes with IoU > 0.5
}
```

**Recommended values:**
- `1.0`: Disabled (default)
- `0.5`: Moderate overlap removal
- `0.3`: Aggressive overlap removal

---

## 📊 Output Format

### Label Studio JSON Response

```json
{
  "result": [
    {
      "from_name": "label",
      "to_name": "image",
      "type": "rectanglelabels",
      "value": {
        "x": 10.5,           # % from left
        "y": 20.3,           # % from top
        "width": 15.2,       # % width
        "height": 18.7,      # % height
        "rectanglelabels": ["0_Adhering_Skin"]
      },
      "score": 0.85         # Confidence score
    }
  ],
  "model_version": "dfine_v1"
}
```

### Where Visualizations Happen

**NOT in this script!** Label Studio's frontend renders the bounding boxes:

```
dfine_labelstudio_backend.py → Returns JSON with bbox coordinates
                              ↓
                    Label Studio UI → Draws boxes on image
                              ↓
                    User sees annotated image in browser
```

---

## 🔧 Customization

### Change Model Path

Edit line 41:

```python
MODEL_PATH = "/path/to/your/model_2.pt"
```

### Change Input Size

Edit line 43:

```python
INPUT_SIZE = (1920, 640)  # (width, height)
```

### Change Number of Classes

Edit line 44:

```python
NUM_CLASSES = 50  # If you have 50 classes
```

And update `CLASS_NAMES` list (lines 46-56).

### Add Preprocessing

Modify the `predict()` method around line 1635:

```python
# Before inference
image_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

# Add custom preprocessing here:
# image_tensor = custom_normalize(image_tensor)
```

---

## 🐛 Troubleshooting

### 1. Model File Not Found

```
FileNotFoundError: Model file not found at /Users/borde/arnav/model_2.pt
```

**Solution:** Update `MODEL_PATH` on line 41 to correct path.

### 2. AWS S3 Access Denied

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution:** Configure AWS credentials:

```bash
aws configure
```

### 3. No Detections

**Possible causes:**
- Confidence threshold too high → Lower to 0.1 or 0.05
- Model not trained on similar images
- Wrong model file loaded

**Debug:**
```python
# Add logging in predict() method:
logger.info(f"Max score before filtering: {max_scores.max().item()}")
logger.info(f"Detections above threshold: {keep_mask.sum().item()}")
```

### 4. Too Many False Positives

**Solution:**
- Increase confidence threshold to 0.5 or 0.7
- Enable NMS with threshold 0.5 or 0.3

### 5. Duplicate Boxes on Same Object

**Solution:**
- Enable NMS: `nms_threshold = 0.5`

---

## 📦 Comparison with YOLOv5 Backend

| Feature | YOLOv5 Backend | D-FINE Backend (This Script) |
|---------|---------------|------------------------------|
| Model | YOLOv5 | D-FINE-L (HGNetv2) |
| Model Loading | `attempt_load()` | Custom `build_model()` + `load_checkpoint()` |
| Preprocessing | Letterbox + normalize | Resize(448x1280) + ToTensor |
| Inference | `model(im)` + NMS | `model(image_tensor)` (NMS built-in) |
| Output Format | Same (Label Studio JSON) | Same (Label Studio JSON) |
| S3 Integration | ✅ Same | ✅ Same |
| Visualization | Label Studio UI | Label Studio UI |

**Key Difference:** D-FINE has built-in transformer-based detection, while YOLOv5 uses anchor-based detection.

---

## 📝 Example: Testing Locally

### Test Script (Optional)

Create `test_dfine_backend.py`:

```python
import base64
from PIL import Image
from io import BytesIO

# Simulate Label Studio task
task = {
    'data': {
        'image': 's3://your-bucket/test-image.jpg'
    }
}

# Initialize backend
from dfine_labelstudio_backend import DFineModel
model = DFineModel()

# Run prediction
predictions = model.predict([task])

print(f"Found {len(predictions[0]['result'])} detections")
for pred in predictions[0]['result']:
    label = pred['value']['rectanglelabels'][0]
    score = pred['score']
    print(f"  - {label}: {score:.3f}")
```

Run:
```bash
python test_dfine_backend.py
```

---

## 🚀 Production Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install torch torchvision pillow boto3 numpy label-studio-ml

# Copy backend script
COPY dfine_labelstudio_backend.py /app/
COPY model_2.pt /app/

# Expose port
EXPOSE 9090

# Start backend
CMD ["label-studio-ml", "start", "dfine_labelstudio_backend", "--port", "9090", "--host", "0.0.0.0"]
```

Build and run:
```bash
docker build -t dfine-backend .
docker run -p 9090:9090 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY dfine-backend
```

---

## 📚 Resources

- **Label Studio Docs:** https://labelstud.io/guide/ml.html
- **D-FINE Paper:** (Add link to your paper/repo)
- **PyTorch:** https://pytorch.org/
- **AWS S3:** https://aws.amazon.com/s3/

---

## ✅ Summary

**This script replaces YOLOv5 with your D-FINE model while keeping:**
- ✅ Same Label Studio integration
- ✅ Same S3 workflow
- ✅ Same JSON output format
- ✅ Same UI rendering

**Differences:**
- ✨ Uses D-FINE-L instead of YOLOv5
- ✨ Custom preprocessing (1280x448)
- ✨ 35 almond defect classes
- ✨ No external config files needed (all embedded)

**No files are created** - everything is returned to Label Studio for display in the web UI!
