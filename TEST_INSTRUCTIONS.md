# 🧪 Testing D-FINE Label Studio Backend

## Prerequisites

Before testing, you need:

1. ✅ Model file: `model_2.pt`
   - Your original script mentions: `/Users/borde/arnav/model_2.pt`
   - Copy it to this directory OR update the path in the script

2. ✅ Test image with almonds/objects
   - Any JPG/PNG image

3. ✅ Python packages installed

---

## 🚀 Testing Steps

### Step 1: Install Dependencies

```bash
cd /home/user/ArNaV248

# Install required packages
pip install torch torchvision pillow numpy
pip install label-studio-ml boto3

# OR use requirements file
pip install -r requirements_dfine_backend.txt
```

---

### Step 2: Update Model Path

**IMPORTANT:** Edit `dfine_labelstudio_backend.py` line 41

```python
# Change this line:
MODEL_PATH = "/Users/borde/arnav/model_2.pt"  # ← OLD PATH

# To your actual model location:
MODEL_PATH = "/home/user/ArNaV248/model_2.pt"  # ← NEW PATH
# OR wherever your model_2.pt is located
```

Quick edit with sed:
```bash
# If model is in current directory
sed -i 's|MODEL_PATH = "/Users/borde/arnav/model_2.pt"|MODEL_PATH = "/home/user/ArNaV248/model_2.pt"|' dfine_labelstudio_backend.py

# OR if model is elsewhere
sed -i 's|MODEL_PATH = ".*"|MODEL_PATH = "/path/to/your/model_2.pt"|' dfine_labelstudio_backend.py
```

---

### Step 3: Test Locally (WITHOUT Label Studio or S3)

This tests the D-FINE model directly using a local image:

```bash
cd /home/user/ArNaV248

# Basic test
python test_dfine_backend_local.py --image /path/to/your/test_image.jpg

# Lower confidence for more detections
python test_dfine_backend_local.py --image test.jpg --conf 0.1

# Higher confidence for fewer detections
python test_dfine_backend_local.py --image test.jpg --conf 0.5
```

**Expected output:**
```
==================================================================
D-FINE LABEL STUDIO BACKEND - LOCAL TEST
==================================================================

✓ Image found: test.jpg
📥 Loading image...
   Image size: 1920 x 1080
🔧 Importing D-FINE backend...
   ✓ Backend imported successfully
🚀 Initializing D-FINE model...
   ✓ Model initialized successfully
🎯 Running inference...
   ✓ Inference completed

==================================================================
RESULTS
==================================================================

📊 Total detections: 15

✅ Detections found!

  1. 0_Adhering_Skin
     Confidence: 0.850
     BBox (pixels): [120, 80, 200, 150]

  2. 5_Chip_Scratch_1_4
     Confidence: 0.720
     BBox (pixels): [300, 120, 380, 200]
  ...
```

---

### Step 4: Start Label Studio Backend (Full Integration)

Only do this after local test succeeds!

```bash
cd /home/user/ArNaV248

# Start the ML backend
label-studio-ml start dfine_labelstudio_backend --port 9090
```

**Expected output:**
```
[INFO] Initializing D-FINE model from /home/user/ArNaV248/model_2.pt
[INFO] Using CUDA device: NVIDIA GeForce RTX 3090
[INFO] Model initialized successfully
[INFO] Input size: (1280, 448) (W x H)
[INFO] Number of classes: 35
✓ ML Backend running on http://0.0.0.0:9090
```

---

### Step 5: Connect to Label Studio

1. **Start Label Studio** (in another terminal):
   ```bash
   label-studio start
   ```

2. **Open Label Studio UI:** http://localhost:8080

3. **Add ML Backend:**
   - Go to: **Settings → Machine Learning**
   - Click: **"Add Model"**
   - Enter URL: `http://localhost:9090`
   - Click: **"Validate and Save"**

4. **Test with S3 images:**
   - Import tasks with S3 URLs: `s3://bucket/image.jpg`
   - Open a task
   - Click **"Get Predictions"**
   - See bounding boxes appear! ✨

---

## 🐛 Troubleshooting

### Issue 1: "Model file not found"

```bash
# Find your model file
find /home/user -name "model_2.pt" 2>/dev/null
find /Users -name "model_2.pt" 2>/dev/null

# Update the path in dfine_labelstudio_backend.py line 41
```

---

### Issue 2: "No module named 'torch'"

```bash
pip install torch torchvision
```

---

### Issue 3: "No module named 'label_studio_ml'"

```bash
pip install label-studio-ml
```

---

### Issue 4: "Unable to locate AWS credentials" (for S3)

```bash
# Only needed if using S3 images
aws configure

# Enter your AWS credentials
```

---

### Issue 5: "No detections found"

**Solution 1:** Lower confidence threshold
```bash
python test_dfine_backend_local.py --image test.jpg --conf 0.05
```

**Solution 2:** Check if image contains objects model was trained on
- Model expects: almonds, defects, etc.
- Won't detect: cats, dogs, cars

**Solution 3:** Verify model file is correct
```bash
# Check model file size (should be ~150-200 MB)
ls -lh /path/to/model_2.pt
```

---

## ✅ Success Criteria

You'll know it's working when:

1. ✅ Local test shows detections:
   ```bash
   python test_dfine_backend_local.py --image test.jpg
   # Output: "📊 Total detections: 15"
   ```

2. ✅ Backend starts without errors:
   ```bash
   label-studio-ml start dfine_labelstudio_backend --port 9090
   # Output: "✓ ML Backend running"
   ```

3. ✅ Label Studio connects (green checkmark in UI)

4. ✅ Predictions appear when you click "Get Predictions"

---

## 📊 Test Examples

### Example 1: Test with placeholder image

If you don't have a test image:

```bash
# Create a dummy image using Python
python3 << EOF
from PIL import Image
import numpy as np

# Create random image
img = Image.fromarray(np.random.randint(0, 255, (448, 1280, 3), dtype=np.uint8))
img.save('test_image.jpg')
print("✓ Created test_image.jpg")
EOF

# Test with it (won't detect anything, but tests the pipeline)
python test_dfine_backend_local.py --image test_image.jpg --conf 0.05
```

---

### Example 2: Test with different confidence levels

```bash
# Very sensitive (many detections)
python test_dfine_backend_local.py --image almond.jpg --conf 0.05

# Balanced (recommended)
python test_dfine_backend_local.py --image almond.jpg --conf 0.3

# Very strict (only high-confidence)
python test_dfine_backend_local.py --image almond.jpg --conf 0.7
```

---

## 🎯 Next Steps After Successful Test

1. **Read full docs:** `README_DFINE_LABELSTUDIO.md`
2. **Deploy to Label Studio:** Follow Step 4 above
3. **Import S3 image URLs** to Label Studio
4. **Get predictions** and refine annotations

---

## 📞 Need Help?

- **Model path issues:** Check line 41 in `dfine_labelstudio_backend.py`
- **No detections:** Try `--conf 0.05` or lower
- **Installation issues:** Run `pip install -r requirements_dfine_backend.txt`
- **Full documentation:** See `README_DFINE_LABELSTUDIO.md`

---

Happy testing! 🚀
