#!/bin/bash
# Quick setup and test script for D-FINE Label Studio backend

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  D-FINE Label Studio Backend - Setup & Test                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Change to script directory
cd "$(dirname "$0")"

echo "📂 Working directory: $(pwd)"
echo ""

# Step 1: Check Python
echo "1️⃣  Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✅ $PYTHON_VERSION"
else
    echo "   ❌ Python3 not found! Please install Python 3.8+"
    exit 1
fi
echo ""

# Step 2: Install dependencies
echo "2️⃣  Installing dependencies..."
echo "   This may take a few minutes..."
pip install -q torch torchvision pillow numpy boto3 label-studio-ml 2>&1 | grep -v "Requirement already satisfied" || true
echo "   ✅ Dependencies installed"
echo ""

# Step 3: Check for model file
echo "3️⃣  Checking for model file..."
MODEL_CURRENT="./model_2.pt"
MODEL_ORIGINAL="/Users/borde/arnav/model_2.pt"

if [ -f "$MODEL_CURRENT" ]; then
    echo "   ✅ Found model at: $MODEL_CURRENT"
    MODEL_PATH="$MODEL_CURRENT"
elif [ -f "$MODEL_ORIGINAL" ]; then
    echo "   ⚠️  Found model at: $MODEL_ORIGINAL"
    read -p "   Copy to current directory? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$MODEL_ORIGINAL" "$MODEL_CURRENT"
        echo "   ✅ Copied model to current directory"
        MODEL_PATH="$MODEL_CURRENT"
    else
        MODEL_PATH="$MODEL_ORIGINAL"
    fi
else
    echo "   ❌ Model file not found!"
    echo "   Please specify the path to your model_2.pt file:"
    read -p "   Model path: " MODEL_PATH
    if [ ! -f "$MODEL_PATH" ]; then
        echo "   ❌ File not found: $MODEL_PATH"
        exit 1
    fi
fi
echo ""

# Step 4: Update model path in script
echo "4️⃣  Updating model path in script..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|MODEL_PATH = \".*\"|MODEL_PATH = \"$MODEL_PATH\"|" dfine_labelstudio_backend.py
else
    # Linux
    sed -i "s|MODEL_PATH = \".*\"|MODEL_PATH = \"$MODEL_PATH\"|" dfine_labelstudio_backend.py
fi
echo "   ✅ Updated to: $MODEL_PATH"
echo ""

# Step 5: Create test image
echo "5️⃣  Creating test image..."
python3 << 'EOF'
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (448, 1280, 3), dtype=np.uint8))
img.save('test_image.jpg')
print("   ✅ Created test_image.jpg")
EOF
echo ""

# Step 6: Run test
echo "6️⃣  Running inference test..."
echo "   This will test the D-FINE model..."
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  TEST OUTPUT                                                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

python3 test_dfine_backend_local.py --image test_image.jpg --conf 0.1

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✅ SETUP COMPLETE!                                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Your D-FINE backend is ready!"
echo ""
echo "📚 Next steps:"
echo "   1. Read: TEST_INSTRUCTIONS.md"
echo "   2. Read: QUICKSTART_DFINE.md"
echo "   3. Start backend: label-studio-ml start dfine_labelstudio_backend --port 9090"
echo ""
echo "🧪 To test with your own image:"
echo "   python test_dfine_backend_local.py --image /path/to/your/image.jpg"
echo ""
