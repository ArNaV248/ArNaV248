# 🚀 Quick Start Guide - D-FINE Label Studio Backend

Get your D-FINE object detection model running with Label Studio in 5 minutes!

---

## ⚡ 1-Minute Setup

```bash
# 1. Install dependencies
pip install -r requirements_dfine_backend.txt

# 2. Configure AWS (if using S3)
aws configure

# 3. Edit model path in dfine_labelstudio_backend.py
# Line 41: MODEL_PATH = "/Users/borde/arnav/model_2.pt"

# 4. Start the ML backend
label-studio-ml start dfine_labelstudio_backend --port 9090

# 5. Open Label Studio (http://localhost:8080)
# Go to Settings → Machine Learning → Add Backend
# URL: http://localhost:9090
```

---

## 📝 Step-by-Step

### Step 1: Install Python Packages

```bash
pip install label-studio-ml torch torchvision pillow boto3 numpy
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import label_studio_ml; print('Label Studio ML installed')"
```

### Step 2: Configure AWS S3 Access

```bash
aws configure
```

Enter:
- **AWS Access Key ID:** `AKIAIOSFODNN7EXAMPLE`
- **AWS Secret Access Key:** `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
- **Region:** `us-east-1` (or your S3 bucket region)
- **Output format:** `json`

**Test S3 access:**
```bash
aws s3 ls s3://your-bucket-name/
```

### Step 3: Update Model Path

Edit `dfine_labelstudio_backend.py`:

```python
# Line 41
MODEL_PATH = "/path/to/your/model_2.pt"  # ← Change this!
```

**Verify model file exists:**
```bash
ls -lh /path/to/your/model_2.pt
```

### Step 4: Start Label Studio (if not running)

```bash
# Install Label Studio
pip install label-studio

# Start Label Studio
label-studio start
```

Opens at: http://localhost:8080

### Step 5: Start D-FINE ML Backend

```bash
cd /home/user/ArNaV248
label-studio-ml start dfine_labelstudio_backend --port 9090
```

**Expected output:**
```
[INFO] Initializing D-FINE model from /path/to/model_2.pt
[INFO] Using CUDA device: NVIDIA GeForce RTX 3090
[INFO] Model initialized successfully
[INFO] Input size: (1280, 448) (W x H)
[INFO] Number of classes: 35
✓ ML Backend running on http://0.0.0.0:9090
```

### Step 6: Connect Backend to Label Studio

1. Open Label Studio: http://localhost:8080
2. Create or open a project
3. Go to **Settings** (⚙️ icon)
4. Click **Machine Learning** tab
5. Click **Add Model**
6. Enter backend URL: `http://localhost:9090`
7. Click **Validate and Save**

**Success message:**
```
✓ Connection successful
✓ Model version: dfine_v1
```

### Step 7: Import Data with S3 URLs

**Option A: Import JSON file**

Create `tasks.json`:
```json
[
  {"data": {"image": "s3://my-bucket/almonds/image001.jpg"}},
  {"data": {"image": "s3://my-bucket/almonds/image002.jpg"}},
  {"data": {"image": "s3://my-bucket/almonds/image003.jpg"}}
]
```

Import:
```
Settings → Import → Upload File → tasks.json
```

**Option B: Import via API**

```python
import requests

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "your-api-key-here"

headers = {"Authorization": f"Token {API_KEY}"}

tasks = [
    {"data": {"image": "s3://my-bucket/almonds/image001.jpg"}},
    {"data": {"image": "s3://my-bucket/almonds/image002.jpg"}},
]

response = requests.post(
    f"{LABEL_STUDIO_URL}/api/projects/1/import",
    headers=headers,
    json=tasks
)
print(response.json())
```

### Step 8: Get Predictions

1. Open any task in Label Studio
2. Click **"Get Predictions"** button (or auto-predict if enabled)
3. D-FINE will:
   - Download image from S3
   - Run inference
   - Return bounding boxes
4. Label Studio displays boxes on the image

---

## 🎯 Test with Sample Image

### Using Local Image (Testing)

Modify the backend temporarily to test with a local image:

```python
# In dfine_labelstudio_backend.py, predict() method:

# Comment out S3 download:
# image_b64 = self._get_image_from_s3(image_url)

# Load local image instead:
from pathlib import Path
img_pil = Image.open("/path/to/test_image.jpg").convert('RGB')
```

Then:
```bash
# Restart backend
label-studio-ml start dfine_labelstudio_backend --port 9090

# In Label Studio, use dummy URL:
{"data": {"image": "file:///dummy.jpg"}}
```

---

## 🔧 Adjust Inference Parameters

### Lower Confidence Threshold (More Detections)

Edit line 1621 in `dfine_labelstudio_backend.py`:

```python
confidence_threshold = 0.1  # Was 0.3, now 0.1 (more sensitive)
```

### Enable NMS (Remove Duplicate Boxes)

Edit line 1622:

```python
nms_threshold = 0.5  # Was 1.0 (disabled), now 0.5 (enabled)
```

**Restart backend after changes:**
```bash
# Ctrl+C to stop
label-studio-ml start dfine_labelstudio_backend --port 9090
```

---

## 📊 Expected Results

### Good Detection Example

```
[INFO] Processing image: s3://bucket/almond_001.jpg
[INFO] Detection: 0_Adhering_Skin, conf=0.850, bbox=[120.5, 80.3, 200.1, 150.7]
[INFO] Detection: 5_Chip_Scratch_1_4, conf=0.720, bbox=[300.2, 120.8, 380.5, 200.3]
[INFO] Detection: 9_NonPareil, conf=0.680, bbox=[450.1, 90.5, 520.3, 180.2]
[INFO] Total detections: 3
```

### Label Studio UI Shows

- ✅ Red bounding boxes on the image
- ✅ Class labels above each box
- ✅ Confidence scores (e.g., "0_Adhering_Skin: 0.85")
- ✅ Ability to edit/refine predictions

---

## 🐛 Common Issues & Fixes

### Issue 1: "Model file not found"

**Error:**
```
FileNotFoundError: Model file not found at /Users/borde/arnav/model_2.pt
```

**Fix:**
```bash
# Find your model file
find ~ -name "model_2.pt" 2>/dev/null

# Update line 41 in dfine_labelstudio_backend.py
MODEL_PATH = "/actual/path/to/model_2.pt"
```

---

### Issue 2: "Unable to locate AWS credentials"

**Error:**
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Fix:**
```bash
# Option 1: Configure via CLI
aws configure

# Option 2: Set environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"

# Option 3: Use IAM role (EC2/ECS)
# No action needed if running on AWS with IAM role
```

---

### Issue 3: "No detections found"

**Symptoms:**
```
[INFO] No detections above threshold
[INFO] Total detections: 0
```

**Fix:**
```python
# Edit dfine_labelstudio_backend.py line 1621
confidence_threshold = 0.05  # Lower from 0.3 to 0.05

# Restart backend
```

---

### Issue 4: "Connection refused to ML backend"

**Error in Label Studio:**
```
✗ Connection failed: Connection refused (http://localhost:9090)
```

**Fix:**
```bash
# Check if backend is running
curl http://localhost:9090/health

# If not running, start it:
label-studio-ml start dfine_labelstudio_backend --port 9090

# Check firewall/port availability
netstat -an | grep 9090
```

---

### Issue 5: "Too many false positives"

**Symptoms:** Label Studio shows 100+ boxes, most are wrong

**Fix:**
```python
# Edit dfine_labelstudio_backend.py

# Option 1: Increase confidence threshold
confidence_threshold = 0.5  # Line 1621

# Option 2: Enable NMS
nms_threshold = 0.5  # Line 1622

# Restart backend
```

---

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Model file exists and path is correct
- [ ] AWS credentials configured (`aws s3 ls` works)
- [ ] ML backend is running (`curl http://localhost:9090/health`)
- [ ] Label Studio shows backend as connected (green checkmark)
- [ ] S3 image URLs are correct format (`s3://bucket/key`)
- [ ] Images are accessible from your AWS account
- [ ] PyTorch with CUDA is working (if using GPU)

**Test PyTorch GPU:**
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📚 Next Steps

### 1. Batch Prediction

Process multiple images:
```bash
# In Label Studio, select multiple tasks
# Click "Predict" → All selected tasks get predictions
```

### 2. Active Learning

Label Studio can prioritize uncertain predictions:
```
Settings → Machine Learning → Enable "Active Learning"
```

### 3. Export Annotations

Export in COCO, YOLO, or Pascal VOC format:
```
Export → Choose format → Download
```

### 4. Fine-tune Model

Use Label Studio annotations to retrain:
```python
# Export annotations
# Train D-FINE model with new data
# Update model_2.pt
# Restart backend
```

---

## 🎓 Advanced Configuration

### Run Backend on Different Port

```bash
label-studio-ml start dfine_labelstudio_backend --port 8090
```

Update Label Studio ML backend URL: `http://localhost:8090`

### Run Backend on Different Host

```bash
label-studio-ml start dfine_labelstudio_backend --host 0.0.0.0 --port 9090
```

Access from other machines: `http://your-ip:9090`

### Enable Debug Logging

```python
# Add to top of dfine_labelstudio_backend.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Different GPU

```python
# Edit __init__ method (line 1604)
self.device = torch.device("cuda:1")  # Use GPU 1 instead of GPU 0
```

---

## 🎉 Success!

You should now see:
- ✅ D-FINE backend running on port 9090
- ✅ Label Studio connected to backend
- ✅ Images loading from S3
- ✅ Bounding boxes displayed on images
- ✅ 35 almond defect classes detected

**Example workflow:**
1. Upload 100 S3 image URLs to Label Studio
2. Click "Predict All" to get D-FINE predictions
3. Review and correct predictions in Label Studio UI
4. Export final annotations in COCO format
5. Use for model evaluation or retraining

---

## 📞 Need Help?

- Read full documentation: `README_DFINE_LABELSTUDIO.md`
- Check Label Studio docs: https://labelstud.io/guide/ml.html
- Test with standalone script first: `dfine_standalone_inference.py`

**Happy annotating! 🚀**
