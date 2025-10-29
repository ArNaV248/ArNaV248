# D-FINE Standalone Inference - Setup Guide

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Hardware**:
  - CPU: Any modern CPU (inference will be slower)
  - GPU: NVIDIA GPU with CUDA support (recommended for faster inference)
  - Mac: Apple Silicon M1/M2/M3 with MPS support

---

## Installation

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

This installs:
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `Pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical computing

---

### Step 2: GPU Setup (Optional but Recommended)

#### For NVIDIA GPU (Windows/Linux):

1. **Check if you have CUDA GPU:**
   ```bash
   nvidia-smi
   ```

2. **Uninstall CPU-only PyTorch:**
   ```bash
   pip uninstall torch torchvision
   ```

3. **Install PyTorch with CUDA support:**

   Visit https://pytorch.org/get-started/locally/ to get the correct command for your system.

   Examples:
   ```bash
   # For CUDA 11.8:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify GPU is detected:**
   ```bash
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
   ```

#### For Mac with Apple Silicon (M1/M2/M3):

**Good news!** MPS (Metal Performance Shaders) support is already included in PyTorch 2.0+. No additional setup needed!

**Verify MPS is available:**
```bash
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

---

## Usage

### Basic Usage (Single Image):

```bash
python3 dfine_standalone_inference.py --input image.jpg --conf 0.3
```

### Command Line Options:

```bash
python3 dfine_standalone_inference.py \
  --input /path/to/image.jpg \
  --model /path/to/model_2.pt \
  --output output_folder \
  --conf 0.3 \
  --nms 1.0 \
  --no-vis \
  --no-json
```

**Options:**
- `--input`: Path to image or folder (required)
- `--model`: Path to model file (default: `/Users/borde/arnav/model_2.pt`)
- `--output`: Output directory (default: `output`)
- `--conf`: Confidence threshold 0.0-1.0 (default: 0.3)
- `--nms`: NMS IoU threshold (default: 1.0, disabled)
- `--no-vis`: Skip saving visualization images
- `--no-json`: Skip saving JSON detection files

### Process Multiple Images:

```bash
python3 dfine_standalone_inference.py --input /path/to/image/folder --conf 0.3
```

---

## Performance Comparison

| Hardware | Speed | Notes |
|----------|-------|-------|
| **NVIDIA GPU** | ~10-50ms/image | Fastest, recommended for production |
| **Apple M1/M2/M3 (MPS)** | ~100-200ms/image | Good balance for Mac users |
| **CPU** | ~1-5 seconds/image | Slower, but works everywhere |

---

## Checking Your Setup

Run this diagnostic to check everything:

```bash
python3 << 'EOF'
import torch
import torchvision
from PIL import Image
import numpy as np

print("="*60)
print("D-FINE Setup Diagnostic")
print("="*60)

print(f"\n✓ Python packages installed:")
print(f"  - PyTorch: {torch.__version__}")
print(f"  - TorchVision: {torchvision.__version__}")
print(f"  - Pillow: {Image.__version__}")
print(f"  - NumPy: {np.__version__}")

print(f"\n✓ Hardware acceleration:")
print(f"  - CUDA (NVIDIA GPU): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    GPU Count: {torch.cuda.device_count()}")

print(f"  - MPS (Apple Silicon): {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    print(f"\n✓ Will use: NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    print(f"\n✓ Will use: Apple Silicon (MPS)")
else:
    print(f"\n⚠ Will use: CPU only (slower)")
    print(f"  Consider installing GPU-accelerated PyTorch for better performance")

print("\n" + "="*60)
EOF
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Process images one at a time, or use smaller batches

### Issue: "No GPU detected" but you have NVIDIA GPU
**Solution:**
1. Install NVIDIA drivers: https://www.nvidia.com/drivers
2. Install CUDA toolkit
3. Reinstall PyTorch with CUDA support (see GPU Setup above)

### Issue: Script is very slow
**Solution:**
- Verify GPU is being used (check diagnostic output)
- If no GPU, consider cloud services with GPU (Google Colab, AWS, etc.)

### Issue: "Model checkpoint not found"
**Solution:** Update the model path:
```bash
python3 dfine_standalone_inference.py --model /correct/path/to/model_2.pt --input image.jpg
```

---

## Files and Outputs

After running inference, you'll find:

```
output/
├── visualize/           # Images with bounding boxes drawn
│   └── image.jpg
├── json/               # Detection results in JSON format
│   └── image.json
└── logits_*.pt        # Raw model outputs (for debugging)
```

**JSON format:**
```json
{
  "image": "image.jpg",
  "num_detections": 23,
  "detections": [
    {
      "class": "4_Carmel",
      "class_id": 4,
      "confidence": 0.99,
      "bbox": {"x1": 450.3, "y1": 9.1, "x2": 550.3, "y2": 77.5}
    }
  ]
}
```

---

## Need Help?

1. Check this guide first
2. Run the diagnostic script above
3. Check the GitHub issues: https://github.com/ArNaV248/ArNaV248/issues

---

## License

This standalone script is based on D-FINE architecture for almond defect detection.
