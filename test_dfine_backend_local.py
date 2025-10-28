#!/usr/bin/env python3
"""
Local Test Script for D-FINE Label Studio Backend
==================================================

Tests the backend without needing Label Studio or S3.
Uses a local image file for quick verification.

Usage:
    python test_dfine_backend_local.py --image /path/to/test.jpg
"""

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path
from PIL import Image

def test_backend(image_path: str, conf_threshold: float = 0.3):
    """Test D-FINE backend with a local image"""

    print("="*70)
    print("D-FINE LABEL STUDIO BACKEND - LOCAL TEST")
    print("="*70)

    # 1. Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n✓ Image found: {image_path}")

    # 2. Load image and convert to base64 (simulating S3 download)
    print("\n📥 Loading image...")
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    print(f"   Image size: {img_width} x {img_height}")

    # Convert to base64 (this is what S3 download returns)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(f"   Base64 encoded: {len(img_b64)} chars")

    # 3. Import backend
    print("\n🔧 Importing D-FINE backend...")
    try:
        # Import with mock S3
        import unittest.mock as mock

        # Mock S3 client to avoid AWS dependency
        with mock.patch('boto3.client'):
            from dfine_labelstudio_backend import DFineModel

        print("   ✓ Backend imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import backend: {e}")
        print("\n💡 Make sure dfine_labelstudio_backend.py is in the same directory")
        sys.exit(1)

    # 4. Initialize model
    print("\n🚀 Initializing D-FINE model...")
    try:
        model = DFineModel()
        print("   ✓ Model initialized successfully")
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        print("\n💡 Check:")
        print("   1. Model file exists (MODEL_PATH in script)")
        print("   2. PyTorch is installed: pip install torch torchvision")
        print("   3. GPU drivers if using CUDA")
        sys.exit(1)

    # 5. Create mock task (simulating Label Studio task)
    print("\n📋 Creating mock task...")
    task = {
        'data': {
            'image': f's3://mock-bucket/{Path(image_path).name}'
        }
    }
    print(f"   Task: {task}")

    # 6. Mock S3 download to return our local image
    print("\n🔄 Mocking S3 download (using local image)...")
    original_method = model._get_image_from_s3
    model._get_image_from_s3 = lambda url: img_b64

    # 7. Run prediction
    print("\n🎯 Running inference...")
    print(f"   Confidence threshold: {conf_threshold}")
    try:
        predictions = model.predict([task], context={'confidence_threshold': conf_threshold})
        print("   ✓ Inference completed")
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 8. Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    if not predictions or len(predictions) == 0:
        print("❌ No predictions returned")
        sys.exit(1)

    result = predictions[0]
    detections = result.get('result', [])

    print(f"\n📊 Total detections: {len(detections)}")

    if len(detections) == 0:
        print("\n⚠️  No objects detected!")
        print("\n💡 Try:")
        print(f"   1. Lower confidence: python {sys.argv[0]} --image {image_path} --conf 0.1")
        print("   2. Check if image contains almonds/objects model was trained on")
        print("   3. Verify model file is correct")
    else:
        print("\n✅ Detections found!\n")

        for i, det in enumerate(detections, 1):
            label = det['value']['rectanglelabels'][0]
            score = det['score']
            x = det['value']['x']
            y = det['value']['y']
            w = det['value']['width']
            h = det['value']['height']

            # Convert percentages to pixels
            x_px = int(x * img_width / 100)
            y_px = int(y * img_height / 100)
            w_px = int(w * img_width / 100)
            h_px = int(h * img_height / 100)

            print(f"  {i}. {label}")
            print(f"     Confidence: {score:.3f}")
            print(f"     BBox (pixels): [{x_px}, {y_px}, {x_px+w_px}, {y_px+h_px}]")
            print(f"     BBox (percent): [{x:.1f}%, {y:.1f}%, {w:.1f}%, {h:.1f}%]")
            print()

        # Show confidence distribution
        scores = [d['score'] for d in detections]
        print(f"📈 Confidence Statistics:")
        print(f"   Min:  {min(scores):.3f}")
        print(f"   Max:  {max(scores):.3f}")
        print(f"   Mean: {sum(scores)/len(scores):.3f}")

        # Show class distribution
        from collections import Counter
        classes = [d['value']['rectanglelabels'][0] for d in detections]
        class_counts = Counter(classes)
        print(f"\n🏷️  Detected Classes:")
        for cls, count in class_counts.most_common():
            print(f"   - {cls}: {count} object(s)")

    # 9. Show Label Studio format
    print("\n" + "="*70)
    print("LABEL STUDIO OUTPUT (Sample)")
    print("="*70)
    import json
    print(json.dumps(predictions[0], indent=2)[:1000] + "\n... (truncated)")

    # 10. Success message
    print("\n" + "="*70)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n💡 Next steps:")
    print("   1. Start Label Studio: label-studio start")
    print("   2. Start ML backend: label-studio-ml start dfine_labelstudio_backend --port 9090")
    print("   3. Connect backend in Label Studio UI")
    print("   4. Upload S3 image URLs to Label Studio")
    print("   5. Get predictions!\n")

    # Restore original method
    model._get_image_from_s3 = original_method

def main():
    parser = argparse.ArgumentParser(
        description='Test D-FINE Label Studio backend locally without S3/Label Studio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_dfine_backend_local.py --image /path/to/almond.jpg
    python test_dfine_backend_local.py --image test.jpg --conf 0.1
        """
    )

    parser.add_argument('--image', '-i', required=True, help='Path to test image')
    parser.add_argument('--conf', '-c', type=float, default=0.3, help='Confidence threshold (default: 0.3)')

    args = parser.parse_args()

    test_backend(args.image, args.conf)

if __name__ == '__main__':
    main()
