# D-FINE Almond Detection - Label Studio ML Backend

Production-ready Docker deployment for D-FINE almond quality detection with Label Studio.

## 📋 Overview

This is a containerized ML backend for Label Studio that provides automated almond defect detection using the D-FINE (Deformable DETR) model.

**Features:**
- ✅ Docker-based deployment (production-ready)
- ✅ GPU acceleration support
- ✅ 35 almond defect classes
- ✅ S3 image loading
- ✅ Health checks and monitoring
- ✅ Runs alongside existing YOLOv5 backend

---

## 🏗️ Architecture

```
Label Studio (Port 8080)
├── YOLOv5 Backend (Port 9090) - Walnuts
└── D-FINE Backend (Port 9091) - Almonds ⭐ This backend
```

---

## 📁 Structure

```
dfine-almond-backend/
├── Dockerfile              # Docker image definition
├── .dockerignore           # Files to exclude from image
├── docker-compose.yml      # Docker Compose configuration
├── model.py                # Main ML backend (full D-FINE code embedded)
├── _wsgi.py                # WSGI entry point
├── README.md               # This file
├── requirements-base.txt   # Core dependencies
├── requirements-test.txt   # Testing dependencies
├── requirements.txt        # All dependencies
└── test_api.py             # API test suite
```

---

## 🚀 Quick Start

### Prerequisites

- Docker with GPU support (nvidia-docker2)
- Model file: `model_2.pt` (~150-200 MB)
- AWS credentials (for S3 access)
- Label Studio running

### 1. Prepare Model File

```bash
# Create models directory
mkdir -p dfine-almond-backend/models

# Copy your trained model
cp /path/to/model_2.pt dfine-almond-backend/models/
```

### 2. Configure Environment

Create `.env` file:

```bash
# AWS Credentials (for S3 image access)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1

# Label Studio
LABEL_STUDIO_API_KEY=your-label-studio-api-key
```

### 3. Start with Docker Compose

```bash
cd dfine-almond-backend

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f dfine-backend

# Check health
curl http://localhost:9091/health
```

### 4. Connect to Label Studio

1. Open Label Studio: `http://localhost:8080`
2. Go to: **Settings → Machine Learning**
3. Click: **Add Model**
4. Enter URL: `http://localhost:9091` (or `http://dfine-backend:9091` if using Docker network)
5. Click: **Validate and Save**

---

## 🐳 Docker Commands

### Build Image

```bash
# Build from dfine-almond-backend directory
cd dfine-almond-backend
docker build -t dfine-almond-backend .

# Or build with docker-compose (recommended)
docker-compose build
```

### Run Container

```bash
# Using docker run
docker run -d \
  --name dfine-backend \
  --gpus all \
  -p 9091:9091 \
  -v $(pwd)/models:/app/models:ro \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e MODEL_PATH=/app/models/model_2.pt \
  --restart unless-stopped \
  dfine-almond-backend

# Using docker-compose (recommended)
docker-compose up -d
```

### Manage Container

```bash
# View logs
docker logs -f dfine-backend

# Stop
docker stop dfine-backend

# Restart
docker restart dfine-backend

# Remove
docker rm -f dfine-backend

# With docker-compose
docker-compose logs -f
docker-compose stop
docker-compose restart
docker-compose down
```

---

## 🧪 Testing

### Test API Endpoints

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run test suite
python test_api.py

# Test with S3 image
python test_api.py --s3-url s3://your-bucket/test-image.jpg

# Test different backend URL
python test_api.py --url http://your-server:9091
```

### Test with curl

```bash
# Health check
curl http://localhost:9091/health

# Setup (model info)
curl -X POST http://localhost:9091/setup

# Predict (requires S3 image)
curl -X POST http://localhost:9091/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "data": {
        "image": "s3://bucket/almond.jpg"
      }
    }]
  }'
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model_2.pt | `/app/models/model_2.pt` |
| `LABEL_STUDIO_URL` | Label Studio server URL | `http://label-studio:8080` |
| `LABEL_STUDIO_API_KEY` | API key | Required |
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `CUDA_VISIBLE_DEVICES` | GPU device ID | `0` |

### Model Configuration

Edit `model.py` to change:
- Input size (default: 1280x448)
- Number of classes (default: 35)
- Class names
- Confidence threshold

---

## 📊 Resource Usage

**Memory:**
- RAM: 2-3 GB
- GPU VRAM: 4-6 GB

**Disk:**
- Docker image: ~3 GB
- Model file: ~200 MB

**CPU/GPU:**
- Inference time: ~50-80ms (GPU)
- Supports CUDA 11.x+

---

## 🔧 Troubleshooting

### Container won't start

```bash
# Check logs
docker logs dfine-backend

# Common issues:
# 1. Model file not found
ls dfine-almond-backend/models/model_2.pt

# 2. GPU not available
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 3. Port already in use
lsof -i :9091
```

### Out of GPU memory

```bash
# Use CPU instead
docker run ... -e CUDA_VISIBLE_DEVICES=-1 ...

# Or use different GPU
docker run ... -e CUDA_VISIBLE_DEVICES=1 ...
```

### AWS S3 access denied

```bash
# Test AWS credentials
aws s3 ls s3://your-bucket/

# Update .env file with correct credentials
```

### Predictions not showing in Label Studio

1. Check backend connection: Settings → ML → should show green checkmark
2. Verify backend URL: `http://localhost:9091` or `http://dfine-backend:9091`
3. Check logs: `docker logs dfine-backend`
4. Test manually: `python test_api.py`

---

## 🚀 Production Deployment

### systemd Service

Create `/etc/systemd/system/dfine-backend.service`:

```ini
[Unit]
Description=D-FINE Almond Detection Backend
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/user/ArNaV248/dfine-almond-backend
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable dfine-backend
sudo systemctl start dfine-backend
sudo systemctl status dfine-backend
```

### Scaling

```bash
# Multiple workers (not recommended - GPU conflicts)
# Instead, use nginx load balancing with multiple containers on different GPUs

# Backend 1 on GPU 0
docker run ... -e CUDA_VISIBLE_DEVICES=0 -p 9091:9091 ...

# Backend 2 on GPU 1
docker run ... -e CUDA_VISIBLE_DEVICES=1 -p 9092:9091 ...
```

---

## 📚 Related Documentation

- **Parent Directory:** See `../` for full D-FINE backend code
- **Deployment Guide:** `../MULTI_BACKEND_DEPLOYMENT.md`
- **Workflow Architecture:** `../WORKFLOW_ARCHITECTURE.md`
- **Testing:** `../TEST_INSTRUCTIONS.md`

---

## 🔍 Differences from Parent Directory

This Docker structure differs from the parent directory:

| Aspect | Parent Directory | This Directory |
|--------|------------------|----------------|
| **Deployment** | Manual setup | Docker container |
| **Model** | Full code embedded | Full code embedded |
| **Structure** | Single file | Modular structure |
| **Production** | Development | Production-ready |
| **Dependencies** | requirements.txt | Multi-stage requirements |
| **Testing** | test_dfine_backend_local.py | test_api.py |

---

## ✅ Checklist

Before deploying:

- [ ] Model file exists in `models/model_2.pt`
- [ ] AWS credentials configured in `.env`
- [ ] Docker with GPU support installed
- [ ] Port 9091 available
- [ ] Label Studio running
- [ ] Tested with `python test_api.py`

---

## 📞 Support

**Issues:**
- Check logs: `docker logs dfine-backend`
- Test API: `python test_api.py`
- Verify GPU: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

**Resources:**
- Label Studio Docs: https://labelstud.io/guide/ml.html
- Docker Docs: https://docs.docker.com/
- NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker

---

## 🎉 Quick Recap

```bash
# 1. Prepare
mkdir -p models && cp /path/to/model_2.pt models/

# 2. Configure
cat > .env <<EOF
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
LABEL_STUDIO_API_KEY=your-api-key
EOF

# 3. Deploy
docker-compose up -d

# 4. Test
python test_api.py

# 5. Connect in Label Studio UI
# Settings → ML → Add Model → http://localhost:9091
```

**That's it! Your D-FINE backend is running! 🚀**
