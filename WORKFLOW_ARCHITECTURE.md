# 🔄 D-FINE Label Studio Workflow - Complete System Architecture

Detailed workflow showing all folders, components, and data flow.

**⚠️ IMPORTANT:** This guide assumes you're **adding** D-FINE to an **existing** Label Studio setup!

---

## 🎯 **Deployment Scenario**

### **What You Already Have:**
- ✅ Label Studio running (for walnut detection)
- ✅ YOLOv5 ML backend on port 9090 (for walnuts)
- ✅ S3 buckets with walnut images

### **What You're Adding:**
- ⭐ D-FINE ML backend on port 9091 (for almonds) ← **This repo!**
- ⭐ New almond detection project in Label Studio
- ⭐ S3 bucket/folder with almond images

### **Result:**
- 🎉 **One Label Studio instance** serving **two projects** with **two different ML backends**
- 🎉 Walnut detection continues to work (YOLOv5)
- 🎉 Almond detection now available (D-FINE)

**📖 For detailed deployment steps, see: `MULTI_BACKEND_DEPLOYMENT.md`**

---

## 🏗️ **Your Complete System (Existing + New)**

```
┌─────────────────────────────────────────────────────────────────────┐
│          MULTI-BACKEND ARCHITECTURE (Walnuts + Almonds)            │
└─────────────────────────────────────────────────────────────────────┘

                    Label Studio Server
                    http://localhost:8080
                    (EXISTING - Already running)
                            │
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼

    YOLOv5 Backend    D-FINE Backend    (Future)
    Port: 9090        Port: 9091 ⭐     Port: 9092
    (Walnuts)         (Almonds-NEW)     (Optional)
    EXISTING          THIS REPO
```

---

## 📂 **Directory Structure & Roles**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────┘

📁 /home/user/walnut_detection/      ← EXISTING (YOLOv5 backend)
├── yolov5/                          ── YOLOv5 repository
├── yolo_labelstudio_backend.py      ── YOLOv5 ML backend
└── best.pt                          ── YOLOv5 model weights

📁 /home/user/ArNaV248/              ← NEW (D-FINE backend - This repo!)
├── dfine_labelstudio_backend.py     ── D-FINE ML Backend (port 9091)
├── setup_and_test.sh                ── Automated setup
├── test_dfine_backend_local.py      ── Local testing (no S3)
├── requirements_dfine_backend.txt   ── Python dependencies
└── model_2.pt                       ── D-FINE model weights (150-200 MB)

📁 ~/.local/share/label-studio/      ← SHARED by all backends
├── media/                           ── Temporary image cache (both projects)
├── logs/                            ── Label Studio logs
└── label_studio.sqlite3             ── Database (all tasks & annotations)

📁 S3 Buckets (AWS)                  ← Image storage
└── s3://your-company-bucket/
    ├── walnuts/                     ── Walnut images (YOLOv5 backend)
    │   ├── batch_001/
    │   │   ├── walnut_001.jpg
    │   │   └── walnut_002.jpg
    │   └── batch_002/
    │
    └── almonds/                     ── Almond images (D-FINE backend) ⭐
        ├── batch_001/
        │   ├── almond_001.jpg
        │   ├── almond_002.jpg
        │   └── ...
        └── batch_002/

📁 Label Studio UI (Browser)         ← Web interface
└── http://localhost:8080
    ├── Projects                     ── Organize tasks
    ├── Tasks                        ── Images to annotate
    ├── Annotations                  ── Human labels + ML predictions
    └── Export                       ── Download results

📁 ML Backend (Running Service)      ← D-FINE inference server
└── http://localhost:9090
    ├── /health                      ── Health check endpoint
    ├── /setup                       ── Model initialization
    ├── /predict                     ── Inference endpoint
    └── /webhook                     ── Annotation updates
```

---

## 🔄 **Complete Workflow - Step by Step**

### **Phase 1: Setup & Installation**

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: SETUP (One-time)                                  │
└─────────────────────────────────────────────────────────────┘

Developer's Machine
└── /home/user/ArNaV248/
    │
    ├─[1]─ Run: ./setup_and_test.sh
    │      │
    │      ├── Installs Python packages → /usr/local/lib/python3.x/
    │      │   ├── torch, torchvision  (deep learning)
    │      │   ├── label-studio-ml     (ML backend framework)
    │      │   ├── boto3               (AWS S3 client)
    │      │   └── pillow, numpy       (image processing)
    │      │
    │      ├── Finds model_2.pt → /home/user/ArNaV248/model_2.pt
    │      │   └── Updates MODEL_PATH in dfine_labelstudio_backend.py
    │      │
    │      └── Runs test → test_dfine_backend_local.py
    │          └── Loads model into GPU/CPU memory (4-6 GB RAM)
    │
    ├─[2]─ AWS Configuration
    │      └── Run: aws configure
    │          ├── Stores credentials → ~/.aws/credentials
    │          │   ├── AWS_ACCESS_KEY_ID
    │          │   └── AWS_SECRET_ACCESS_KEY
    │          └── Sets region → ~/.aws/config
    │
    └─[3]─ Label Studio Installation
           └── Run: pip install label-studio
               └── Installs to → /usr/local/bin/label-studio
```

---

### **Phase 2: Starting Services**

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: START SERVICES                                    │
└─────────────────────────────────────────────────────────────┘

Terminal 1: Label Studio
└── Run: label-studio start
    │
    ├── Creates data directory → ~/.local/share/label-studio/
    │   ├── label_studio.sqlite3  (stores tasks & annotations)
    │   ├── media/                (temporary image cache)
    │   └── logs/                 (application logs)
    │
    ├── Starts web server → http://localhost:8080
    │   ├── Web UI (React frontend)
    │   ├── API server (Django backend)
    │   └── WebSocket (real-time updates)
    │
    └── Ready to accept ML backend connections


Terminal 2: D-FINE ML Backend
└── Run: label-studio-ml start dfine_labelstudio_backend --port 9090
    │
    ├── Loads from → /home/user/ArNaV248/
    │   ├── dfine_labelstudio_backend.py
    │   └── model_2.pt
    │
    ├── Initializes D-FINE model
    │   ├── Loads weights into GPU memory (~4 GB VRAM)
    │   ├── Sets model to eval mode (no training)
    │   └── Prepares preprocessing pipeline
    │
    ├── Starts API server → http://localhost:9090
    │   ├── POST /setup      (model info)
    │   ├── POST /predict    (inference)
    │   ├── POST /webhook    (training callbacks)
    │   └── GET  /health     (health check)
    │
    └── Waits for Label Studio to send tasks
```

---

### **Phase 3: Data Import & Storage**

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: DATA IMPORT                                       │
└─────────────────────────────────────────────────────────────┘

User uploads task list (JSON file) to Label Studio
│
│  tasks.json:
│  [
│    {"data": {"image": "s3://my-bucket/almonds/img001.jpg"}},
│    {"data": {"image": "s3://my-bucket/almonds/img002.jpg"}},
│    ...
│  ]
│
└──> Label Studio imports tasks
     │
     ├── Stores in database → ~/.local/share/label-studio/label_studio.sqlite3
     │   └── Table: tasks
     │       ├── id: 1
     │       ├── data: {"image": "s3://..."}
     │       ├── project_id: 1
     │       └── created_at: timestamp
     │
     └── Shows in UI → http://localhost:8080/projects/1/data
         └── List of tasks ready for annotation
```

---

### **Phase 4: Inference Workflow** (🔥 Main Workflow!)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: INFERENCE WORKFLOW (When user clicks "Get Predictions")      │
└─────────────────────────────────────────────────────────────────────────┘

[1] Label Studio UI (Browser)
    └── User opens task #42
        └── Clicks "Get Predictions" button
            │
            ▼

[2] Label Studio Server (http://localhost:8080)
    └── Sends HTTP POST request to ML Backend
        │
        │  POST http://localhost:9090/predict
        │  Body: {
        │    "tasks": [
        │      {"data": {"image": "s3://my-bucket/almonds/img042.jpg"}}
        │    ]
        │  }
        │
        ▼

[3] D-FINE ML Backend (http://localhost:9090)
    └── Receives task in predict() method
        │
        │ File: dfine_labelstudio_backend.py
        │ Class: DFineModel
        │ Method: predict()
        │
        ▼

[4] Download Image from S3
    └── Method: _get_image_from_s3()
        │
        │ S3 Bucket: s3://my-bucket/
        │ ├── Bucket: my-bucket
        │ └── Key: almonds/img042.jpg
        │
        ├── boto3.client('s3').get_object()
        │   └── Downloads image bytes from AWS
        │
        ├── Converts to base64 string
        │   └── Returns: "iVBORw0KGgoAAAANSUhEUgAA..."
        │
        └── Image data in memory (not saved to disk)
            │
            ▼

[5] Preprocess Image
    └── Method: self.transform()
        │
        │ Input: PIL Image (1920x1080)
        │
        ├── Resize to (1280, 448)  ← D-FINE input size
        │   └── Distorts aspect ratio (no padding)
        │
        ├── Convert to tensor
        │   └── Shape: [1, 3, 448, 1280]  (batch, channels, H, W)
        │
        ├── Normalize to [0, 1]
        │   └── Divides pixel values by 255
        │
        └── Move to GPU/CPU
            └── tensor.to(device)
                │
                ▼

[6] D-FINE Inference (Deep Learning)
    └── Method: self.model(image_tensor)
        │
        │ Model Architecture:
        │ ├── Backbone: HGNetv2      (feature extraction)
        │ ├── Encoder: HybridEncoder (multi-scale features)
        │ └── Decoder: DFINETransformer (detection queries)
        │
        ├── Forward pass through neural network
        │   └── GPU computation (~50-80ms)
        │
        ├── Outputs (raw predictions):
        │   ├── pred_logits: [1, 300, 35]  (confidence for 35 classes)
        │   └── pred_boxes:  [1, 300, 4]   (bounding boxes in cxcywh format)
        │
        └── Returns: {"pred_logits": ..., "pred_boxes": ...}
            │
            ▼

[7] Postprocess Predictions
    └── Method: postprocess_outputs()
        │
        ├── Convert logits to probabilities
        │   └── probs = softmax(pred_logits)
        │       └── Shape: [300, 35]  (300 queries, 35 classes)
        │
        ├── Get max confidence per query
        │   └── max_scores, labels = probs.max(dim=-1)
        │       ├── max_scores: [0.85, 0.72, 0.15, ...]  (300 values)
        │       └── labels: [0, 5, 9, ...]  (class IDs)
        │
        ├── Filter by confidence threshold (default: 0.3)
        │   └── keep_mask = max_scores > 0.3
        │       └── Keeps only high-confidence detections
        │
        ├── Convert boxes: cxcywh → xyxy
        │   └── box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        │       ├── Input:  [cx=0.5, cy=0.5, w=0.2, h=0.3]  (normalized)
        │       └── Output: [x1=120, y1=80, x2=200, y2=150]  (pixels)
        │
        ├── Scale to original image size
        │   └── boxes *= [img_width, img_height]
        │       └── Example: 1920x1080
        │
        └── Optional: Apply NMS (remove duplicates)
            └── keep_indices = nms(boxes, scores, threshold=0.5)
                │
                ▼

[8] Format for Label Studio
    └── Converts to Label Studio JSON format
        │
        │ For each detection:
        │ {
        │   "from_name": "label",
        │   "to_name": "image",
        │   "type": "rectanglelabels",
        │   "value": {
        │     "x": 10.5,           ← % from left edge
        │     "y": 20.3,           ← % from top edge
        │     "width": 15.2,       ← % width
        │     "height": 18.7,      ← % height
        │     "rectanglelabels": ["0_Adhering_Skin"]
        │   },
        │   "score": 0.85          ← Confidence (0-1)
        │ }
        │
        └── Returns predictions array
            │
            ▼

[9] Send Back to Label Studio
    └── HTTP Response
        │
        │ Status: 200 OK
        │ Body: [
        │   {
        │     "result": [
        │       {detection1},
        │       {detection2},
        │       ...
        │     ],
        │     "model_version": "dfine_v1"
        │   }
        │ ]
        │
        ▼

[10] Label Studio Receives Predictions
     └── Stores in database
         │
         │ ~/.local/share/label-studio/label_studio.sqlite3
         │ └── Table: predictions
         │     ├── id: 123
         │     ├── task_id: 42
         │     ├── result: [JSON predictions]
         │     ├── score: 0.85
         │     ├── model_version: "dfine_v1"
         │     └── created_at: timestamp
         │
         ▼

[11] Label Studio UI Renders Predictions
     └── Browser receives predictions via API
         │
         │ JavaScript (React) renders bounding boxes
         │
         ├── Draws red rectangles on image
         │   └── Uses Canvas API or SVG
         │
         ├── Adds class labels above boxes
         │   └── "0_Adhering_Skin: 0.85"
         │
         └── User sees annotated image! ✨
             └── NO FILES CREATED - All in browser memory

═════════════════════════════════════════════════════════════════════

👁️  USER SEES:
    ┌────────────────────────────────────────┐
    │  [Image with red bounding boxes]       │
    │                                        │
    │  ┌──────────────┐                     │
    │  │0_Adhering_Skin│                    │
    │  │   0.85        │                    │
    │  └──────────────┘                     │
    │       ▼                                │
    │  ┌─────────────────┐                  │
    │  │ Red box around  │                  │
    │  │ detected almond │                  │
    │  └─────────────────┘                  │
    │                                        │
    └────────────────────────────────────────┘

═════════════════════════════════════════════════════════════════════
```

---

## 📊 **Data Flow Summary**

```
User's S3 Bucket                Label Studio              D-FINE ML Backend
(Image Storage)                 (Annotation UI)           (Inference Engine)
     │                               │                          │
     │                               │                          │
     │  [1] Store images             │                          │
     │  ─────────────────>           │                          │
     │                               │                          │
     │                               │  [2] Import S3 URLs      │
     │                               │  <────────────           │
     │                               │                          │
     │                               │  [3] User clicks         │
     │                               │      "Get Predictions"   │
     │                               │                          │
     │                               │  [4] Send task           │
     │                               │  ──────────────────────> │
     │                               │                          │
     │  [5] Download image           │                          │
     │  <────────────────────────────────────────────────────── │
     │  Return image bytes           │                          │
     │  ──────────────────────────────────────────────────────> │
     │                               │                          │
     │                               │                   [6] Preprocess
     │                               │                   [7] Inference
     │                               │                   [8] Postprocess
     │                               │                          │
     │                               │  [9] Return predictions  │
     │                               │  <────────────────────── │
     │                               │                          │
     │                         [10] Render boxes                │
     │                         [11] Show to user                │
     │                               │                          │
     ▼                               ▼                          ▼
```

---

## 🗂️ **Folder Responsibilities**

### 1️⃣ `/home/user/ArNaV248/` - D-FINE Backend Directory

**Purpose:** Contains all D-FINE ML backend code and configuration

**Contents:**
- `dfine_labelstudio_backend.py` - Main backend script (2000 lines)
- `model_2.pt` - Model weights (~150-200 MB)
- `test_dfine_backend_local.py` - Testing script
- `setup_and_test.sh` - Automated setup
- Documentation files (*.md)

**Permissions:** Read/write for your user

**When it's used:**
- ✅ At startup (loads model)
- ✅ During inference (runs predictions)
- ✅ Never writes output files (all in memory)

---

### 2️⃣ `~/.local/share/label-studio/` - Label Studio Data

**Purpose:** Label Studio's persistent storage

**Contents:**
```
~/.local/share/label-studio/
├── label_studio.sqlite3    ← Database (tasks, annotations, predictions)
├── media/                  ← Temporary image cache (auto-cleaned)
├── logs/                   ← Application logs
│   ├── label_studio.log
│   └── django.log
└── upload/                 ← Uploaded files (if not using S3)
```

**Size:** Grows with number of tasks (~100 MB per 1000 images)

**When it's used:**
- ✅ Every time Label Studio runs
- ✅ Stores all annotations permanently
- ✅ Caches downloaded S3 images temporarily

**Backup:** Important! Contains all your work
```bash
# Backup command
cp -r ~/.local/share/label-studio/ ~/label-studio-backup/
```

---

### 3️⃣ `s3://your-bucket/` - AWS S3 Image Storage

**Purpose:** Cloud storage for raw images

**Structure:**
```
s3://my-almond-bucket/
├── raw/                    ← Original images
│   ├── batch_001/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── batch_002/
├── processed/              ← (Optional) Processed images
└── exports/                ← (Optional) Label Studio exports
```

**Access:** Via AWS IAM credentials (`~/.aws/credentials`)

**When it's used:**
- ✅ Every time D-FINE backend makes a prediction
- ✅ Downloads image bytes on-demand
- ✅ Never modified by Label Studio

**Cost:** S3 storage + data transfer fees

---

### 4️⃣ `/tmp/` - Temporary Files (Auto-cleaned)

**Purpose:** Temporary storage during processing

**Contents:**
- Downloaded S3 images (Label Studio cache)
- Temporary PIL/OpenCV files
- Model compilation cache (PyTorch)

**Lifetime:** Deleted on system reboot

**When it's used:**
- ✅ During image download
- ✅ Auto-cleaned by OS

---

### 5️⃣ `~/.cache/torch/` - PyTorch Model Cache

**Purpose:** PyTorch compiled models and kernels

**Contents:**
- Compiled CUDA kernels
- Model optimization cache
- JIT compiled functions

**Size:** ~500 MB - 2 GB

**When it's used:**
- ✅ First time running model (compiles kernels)
- ✅ Speeds up subsequent runs

---

## 🔍 **Where Does Data Go?**

### ❌ **What's NOT Created:**

```
✗ No visualize/ folder
✗ No json/ folder
✗ No output/ folder
✗ No saved images with bounding boxes
✗ No prediction JSON files on disk
```

**All predictions are:**
- ✅ Returned to Label Studio via HTTP
- ✅ Stored in Label Studio's database
- ✅ Rendered in browser (client-side)
- ✅ Never saved as files

---

### ✅ **What IS Created:**

| Location | What | Size | Purpose |
|----------|------|------|---------|
| `~/.local/share/label-studio/label_studio.sqlite3` | SQLite database | 100 MB+ | Tasks, annotations, predictions |
| `~/.local/share/label-studio/media/` | Cached images | Varies | Temporary S3 downloads |
| `~/.local/share/label-studio/logs/` | Log files | 10-50 MB | Debugging |
| `~/.cache/torch/` | PyTorch cache | 500 MB - 2 GB | CUDA kernels |
| `/tmp/label-studio-*` | Temp files | Varies | Auto-cleaned |

---

## 🚀 **Performance & Resource Usage**

### Memory Usage:

```
Component                  RAM Usage       GPU VRAM
─────────────────────────  ─────────────   ────────────
Label Studio Server        500 MB - 1 GB   N/A
D-FINE ML Backend          2 - 3 GB        4 - 6 GB
  ├── Model weights        150 MB          150 MB
  ├── Feature maps         -               2 - 4 GB
  ├── Batch inference      500 MB          1 - 2 GB
  └── Python overhead      1 - 2 GB        N/A
Browser (Chrome)           500 MB - 2 GB   N/A (uses CPU)
─────────────────────────  ─────────────   ────────────
Total (typical)            3 - 6 GB        4 - 6 GB
```

### Disk Usage:

```
Component                           Disk Space
──────────────────────────────────  ──────────────
D-FINE backend files                70 MB
Model weights (model_2.pt)          150 - 200 MB
Label Studio installation           200 - 300 MB
Label Studio data (1000 tasks)      100 - 200 MB
PyTorch cache                       500 MB - 2 GB
Python packages (torch, etc.)       2 - 4 GB
──────────────────────────────────  ──────────────
Total (typical)                     3 - 7 GB
```

---

## 📂 **Quick Reference**

### Where is...?

| What | Where | Why |
|------|-------|-----|
| **D-FINE backend code** | `/home/user/ArNaV248/dfine_labelstudio_backend.py` | Your ML backend |
| **Model weights** | `/home/user/ArNaV248/model_2.pt` | D-FINE trained model |
| **Annotations database** | `~/.local/share/label-studio/label_studio.sqlite3` | All your work |
| **Raw images** | `s3://your-bucket/` | Cloud storage |
| **Temporary cache** | `~/.local/share/label-studio/media/` | Auto-cleaned |
| **Logs** | `~/.local/share/label-studio/logs/` | Debugging |
| **AWS credentials** | `~/.aws/credentials` | S3 access |
| **PyTorch cache** | `~/.cache/torch/` | Speed optimization |

---

## 🎯 **Summary**

### The 3 Main Components:

1. **S3 Bucket** (`s3://your-bucket/`)
   - Stores raw images
   - Read-only for Label Studio
   - Accessed via boto3

2. **Label Studio** (`~/.local/share/label-studio/`)
   - Annotation UI (browser)
   - Database (tasks, predictions)
   - Web server (port 8080)

3. **D-FINE Backend** (`/home/user/ArNaV248/`)
   - ML inference engine
   - Runs on port 9090
   - No file outputs (all in memory)

### Data Flow:
```
S3 → D-FINE Backend → Label Studio DB → Browser UI
     (downloads)       (predictions)      (renders)
```

**No files are created on disk** - everything stays in memory or database! ✨
