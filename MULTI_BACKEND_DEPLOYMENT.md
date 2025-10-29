# 🔄 Multi-Backend Deployment: Adding D-FINE to Existing Label Studio

## 📋 Your Current Setup

You already have:
- ✅ Label Studio running (for walnut detection)
- ✅ YOLOv5 ML backend running (for walnuts)
- ✅ S3 buckets with images

**Now adding:**
- ⭐ D-FINE ML backend (for almonds)

---

## 🏗️ **Multi-Backend Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 YOUR COMPLETE SYSTEM ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────────┘

                    Label Studio Server
                    http://localhost:8080
                    (Annotation UI - Single Instance)
                            │
                            │ Can connect to multiple ML backends
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼

    YOLOv5 Backend    D-FINE Backend    (Future Backend)
    Port: 9090        Port: 9091        Port: 9092
    (Walnuts)         (Almonds)         (Optional)
    │                 │                 │
    └─────────────────┴─────────────────┘
                      │
                      ▼
              Amazon S3 Buckets
              ├── s3://walnut-bucket/
              └── s3://almond-bucket/
```

---

## 📁 **Directory Structure (Existing + New)**

### **Your Server Layout:**

```
/home/user/                              ← Your server
│
├── walnut_detection/                    ← EXISTING (YOLOv5)
│   ├── yolov5/                          ── YOLOv5 repo
│   ├── yolo_labelstudio_backend.py      ── YOLOv5 backend script
│   ├── best.pt                          ── YOLOv5 model weights
│   └── (other YOLOv5 files)
│
├── ArNaV248/                            ← NEW (D-FINE for almonds)
│   ├── dfine_labelstudio_backend.py     ── D-FINE backend script
│   ├── model_2.pt                       ── D-FINE model weights
│   ├── setup_and_test.sh                ── Automated setup
│   ├── test_dfine_backend_local.py      ── Test script
│   ├── WORKFLOW_ARCHITECTURE.md         ── This file
│   └── (documentation files)
│
└── .local/share/label-studio/           ← SHARED (Both backends use this)
    ├── label_studio.sqlite3             ── Database (all projects)
    ├── media/                           ── Cached images (S3 downloads)
    └── logs/                            ── Application logs
```

---

## 🚀 **Deployment Steps**

### **Step 1: Transfer D-FINE Files to Server**

Choose one of these methods:

#### **Option A: Git Clone (Recommended)**

```bash
# SSH into your server
ssh user@your-server.com

# Clone from your GitHub
cd /home/user/
git clone https://github.com/ArNaV248/ArNaV248.git
cd ArNaV248

# Checkout the D-FINE branch
git checkout claude/session-011CUZYZYwgzGM9ydQURauXK
```

#### **Option B: SCP (Direct Copy)**

```bash
# From your local machine
cd /home/user/ArNaV248
scp -r * user@your-server.com:/home/user/ArNaV248/
```

#### **Option C: Docker (Containerized)**

```bash
# Create Dockerfile (see Docker section below)
docker build -t dfine-backend .
docker run -p 9091:9091 dfine-backend
```

---

### **Step 2: Install Dependencies (on Server)**

```bash
ssh user@your-server.com
cd /home/user/ArNaV248

# Install D-FINE dependencies
pip install -r requirements_dfine_backend.txt

# OR use automated setup
./setup_and_test.sh
```

---

### **Step 3: Update Configuration**

Edit `dfine_labelstudio_backend.py`:

```python
# Line 41: Update model path
MODEL_PATH = "/home/user/ArNaV248/model_2.pt"

# Line 37-38: Update Label Studio URL (if different)
LABEL_STUDIO_URL = 'http://your-server.com:8080/'  # Or localhost
LABEL_STUDIO_API_KEY = 'your-api-key-here'  # Get from Label Studio UI
```

---

### **Step 4: Run Both Backends Simultaneously**

You need to run backends on **different ports**!

#### **Terminal 1: YOLOv5 Backend (Existing)**

```bash
cd /home/user/walnut_detection
label-studio-ml start yolo_labelstudio_backend --port 9090

# Output:
# ✓ YOLOv5 Backend running on http://0.0.0.0:9090
```

#### **Terminal 2: D-FINE Backend (New)**

```bash
cd /home/user/ArNaV248
label-studio-ml start dfine_labelstudio_backend --port 9091

# Output:
# [INFO] Initializing D-FINE model from /home/user/ArNaV248/model_2.pt
# [INFO] Using CUDA device: NVIDIA GeForce RTX 3090
# [INFO] Model initialized successfully
# ✓ D-FINE Backend running on http://0.0.0.0:9091
```

#### **Using systemd (Production - Auto-restart)**

Create service files:

**File: `/etc/systemd/system/yolo-backend.service`**
```ini
[Unit]
Description=YOLOv5 Label Studio ML Backend (Walnuts)
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/user/walnut_detection
ExecStart=/usr/local/bin/label-studio-ml start yolo_labelstudio_backend --port 9090
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**File: `/etc/systemd/system/dfine-backend.service`**
```ini
[Unit]
Description=D-FINE Label Studio ML Backend (Almonds)
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/user/ArNaV248
ExecStart=/usr/local/bin/label-studio-ml start dfine_labelstudio_backend --port 9091
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable yolo-backend dfine-backend
sudo systemctl start yolo-backend dfine-backend

# Check status
sudo systemctl status yolo-backend
sudo systemctl status dfine-backend
```

---

### **Step 5: Connect Both Backends to Label Studio**

1. **Open Label Studio UI:** http://your-server:8080

2. **For Walnut Project:**
   - Go to walnut project → Settings → Machine Learning
   - Should already have YOLOv5: `http://localhost:9090`

3. **For Almond Project (New):**
   - Create new project or go to existing almond project
   - Settings → Machine Learning → Add Model
   - Enter URL: `http://localhost:9091`
   - Click "Validate and Save"
   - ✅ D-FINE backend connected!

---

## 🔄 **Complete Multi-Backend Workflow**

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP-BY-STEP WORKFLOW (Multiple Backends)                     │
└─────────────────────────────────────────────────────────────────┘

[1] Label Studio Server (Single Instance)
    └── http://localhost:8080
        ├── Project 1: Walnut Detection
        │   └── Connected to: YOLOv5 Backend (port 9090)
        │
        └── Project 2: Almond Detection  ← NEW!
            └── Connected to: D-FINE Backend (port 9091)

[2] User Opens Almond Task
    └── Label Studio UI
        ├── Project: "Almond Quality Detection"
        ├── Task: {"data": {"image": "s3://almond-bucket/img001.jpg"}}
        └── User clicks: "Get Predictions"
            │
            ▼

[3] Label Studio Routes to Correct Backend
    └── Checks project settings
        └── Sends request to: http://localhost:9091/predict
            │
            ▼

[4] D-FINE Backend Receives Request
    └── dfine_labelstudio_backend.py (port 9091)
        ├── Downloads from: s3://almond-bucket/img001.jpg
        ├── Runs D-FINE inference (35 almond defect classes)
        └── Returns predictions (bounding boxes)
            │
            ▼

[5] Label Studio Displays Results
    └── Renders bounding boxes in UI
        └── User sees almond detections! ✨


Meanwhile, YOLOv5 Backend (port 9090) handles walnut tasks independently!
```

---

## 📊 **Resource Usage (Both Backends Running)**

```
Component                    RAM          GPU VRAM      Port
───────────────────────────  ───────────  ────────────  ──────
Label Studio Server          1 GB         N/A           8080
YOLOv5 Backend (Walnuts)     2-3 GB       2-3 GB        9090
D-FINE Backend (Almonds)     2-3 GB       4-6 GB        9091
───────────────────────────  ───────────  ────────────  ──────
Total                        5-7 GB       6-9 GB        -
```

**GPU Recommendation:**
- Minimum: 12 GB VRAM (e.g., RTX 3080 Ti, RTX 4080)
- Recommended: 16+ GB VRAM (e.g., RTX 4090, A5000)

**If GPU memory is limited:**
```bash
# Option 1: Run one backend on GPU, one on CPU
# YOLOv5 on GPU (faster)
CUDA_VISIBLE_DEVICES=0 label-studio-ml start yolo_labelstudio_backend --port 9090

# D-FINE on CPU (slower but works)
CUDA_VISIBLE_DEVICES=-1 label-studio-ml start dfine_labelstudio_backend --port 9091

# Option 2: Use different GPUs (if available)
# YOLOv5 on GPU 0
CUDA_VISIBLE_DEVICES=0 label-studio-ml start yolo_labelstudio_backend --port 9090

# D-FINE on GPU 1
CUDA_VISIBLE_DEVICES=1 label-studio-ml start dfine_labelstudio_backend --port 9091
```

---

## 🗂️ **S3 Bucket Organization**

Recommended structure:

```
s3://my-company-detection-bucket/
│
├── walnuts/                     ← YOLOv5 backend images
│   ├── batch_001/
│   │   ├── walnut_001.jpg
│   │   └── walnut_002.jpg
│   └── batch_002/
│
├── almonds/                     ← D-FINE backend images
│   ├── batch_001/
│   │   ├── almond_001.jpg
│   │   └── almond_002.jpg
│   └── batch_002/
│
└── exports/                     ← Label Studio exports
    ├── walnuts_annotations_2025-01-15.json
    └── almonds_annotations_2025-01-15.json
```

---

## 🔧 **Configuration Checklist**

### ✅ **Before Deployment:**

- [ ] YOLOv5 backend is running on port 9090
- [ ] D-FINE files transferred to `/home/user/ArNaV248/`
- [ ] Dependencies installed: `pip install -r requirements_dfine_backend.txt`
- [ ] Model file exists: `/home/user/ArNaV248/model_2.pt`
- [ ] AWS credentials configured: `~/.aws/credentials`
- [ ] Updated MODEL_PATH in `dfine_labelstudio_backend.py` line 41
- [ ] Tested locally: `./setup_and_test.sh` succeeds
- [ ] Ports 9090 and 9091 are not blocked by firewall

### ✅ **After Deployment:**

- [ ] Both backends running without errors
- [ ] `curl http://localhost:9090/health` returns 200 OK (YOLOv5)
- [ ] `curl http://localhost:9091/health` returns 200 OK (D-FINE)
- [ ] Label Studio shows both backends connected (green checkmarks)
- [ ] Walnut predictions work (YOLOv5)
- [ ] Almond predictions work (D-FINE)
- [ ] Systemd services enabled (auto-restart on reboot)

---

## 🐳 **Docker Deployment (Optional)**

For easier deployment, containerize D-FINE backend:

**File: `/home/user/ArNaV248/Dockerfile`**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements_dfine_backend.txt .
RUN pip install --no-cache-dir -r requirements_dfine_backend.txt

# Copy application files
COPY dfine_labelstudio_backend.py .
COPY model_2.pt .

# Expose port
EXPOSE 9091

# Run backend
CMD ["label-studio-ml", "start", "dfine_labelstudio_backend", "--port", "9091", "--host", "0.0.0.0"]
```

**Build and run:**

```bash
cd /home/user/ArNaV248

# Build image
docker build -t dfine-backend:latest .

# Run container
docker run -d \
  --name dfine-backend \
  --gpus all \
  -p 9091:9091 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=us-east-1 \
  --restart unless-stopped \
  dfine-backend:latest

# Check logs
docker logs -f dfine-backend
```

**Docker Compose (Both Backends):**

**File: `/home/user/docker-compose.yml`**

```yaml
version: '3.8'

services:
  label-studio:
    image: heartexlabs/label-studio:latest
    ports:
      - "8080:8080"
    volumes:
      - label-studio-data:/label-studio/data
    environment:
      - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    restart: unless-stopped

  yolo-backend:
    build: ./walnut_detection
    ports:
      - "9090:9090"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  dfine-backend:
    build: ./ArNaV248
    ports:
      - "9091:9091"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  label-studio-data:
```

**Start all services:**
```bash
docker-compose up -d
```

---

## 📝 **Label Studio Project Setup**

### **Project 1: Walnut Detection (Existing)**

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Walnut_Good" background="green"/>
    <Label value="Walnut_Bad" background="red"/>
    <!-- Other walnut classes -->
  </RectangleLabels>
</View>
```

Connected to: `http://localhost:9090` (YOLOv5)

### **Project 2: Almond Detection (New)**

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="0_Adhering_Skin" background="#FF6B6B"/>
    <Label value="1_Blanched" background="#4ECDC4"/>
    <Label value="2_Broken_Blanched" background="#45B7D1"/>
    <Label value="5_Chip_Scratch_1_4" background="#FFA07A"/>
    <!-- Add all 35 almond classes -->
  </RectangleLabels>
</View>
```

Connected to: `http://localhost:9091` (D-FINE)

---

## 🔍 **Monitoring & Debugging**

### **Check Backend Status:**

```bash
# YOLOv5 backend (walnuts)
curl http://localhost:9090/health
# Expected: {"status": "ok"}

# D-FINE backend (almonds)
curl http://localhost:9091/health
# Expected: {"status": "ok"}
```

### **View Logs:**

```bash
# YOLOv5 logs
journalctl -u yolo-backend -f

# D-FINE logs
journalctl -u dfine-backend -f

# Docker logs
docker logs -f yolo-backend
docker logs -f dfine-backend
```

### **Test Predictions:**

```bash
# Test YOLOv5 backend
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{"tasks":[{"data":{"image":"s3://walnut-bucket/test.jpg"}}]}'

# Test D-FINE backend
curl -X POST http://localhost:9091/predict \
  -H "Content-Type: application/json" \
  -d '{"tasks":[{"data":{"image":"s3://almond-bucket/test.jpg"}}]}'
```

---

## 🎯 **Quick Deployment Commands**

**Full deployment in one go:**

```bash
# 1. SSH to server
ssh user@your-server.com

# 2. Clone D-FINE repo
cd /home/user
git clone https://github.com/ArNaV248/ArNaV248.git
cd ArNaV248

# 3. Run automated setup
./setup_and_test.sh

# 4. Start D-FINE backend (port 9091)
label-studio-ml start dfine_labelstudio_backend --port 9091 &

# 5. Verify both backends
curl http://localhost:9090/health  # YOLOv5
curl http://localhost:9091/health  # D-FINE

# 6. Connect in Label Studio UI
# Settings → Machine Learning → Add Model → http://localhost:9091
```

---

## ✅ **Summary**

### **What You Have Now:**

```
Label Studio (Port 8080)
├── Project 1: Walnuts    → YOLOv5 Backend (Port 9090) ✅
└── Project 2: Almonds    → D-FINE Backend (Port 9091) ⭐ NEW!
```

### **Key Points:**

✅ Run backends on **different ports** (9090, 9091)
✅ Same Label Studio instance serves both projects
✅ Each project connects to its own backend
✅ Files go to: `/home/user/ArNaV248/`
✅ Use systemd for auto-restart in production
✅ Monitor with `curl http://localhost:9091/health`

**You can now detect both walnuts AND almonds! 🎉**
