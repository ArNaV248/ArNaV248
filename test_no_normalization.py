"""
Quick test: What if we DON'T normalize?
This will show what the working script might be doing.
"""

import torch
import torchvision.transforms as T
from PIL import Image

# Test 1: With normalization (current)
image = Image.open("test.jpg").convert('RGB')
image_resized = image.resize((1280, 448), Image.BILINEAR)

transform_with_norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tensor_normalized = transform_with_norm(image_resized)

print("WITH normalization:")
print(f"  Min: {tensor_normalized.min().item():.3f}")
print(f"  Max: {tensor_normalized.max().item():.3f}")
print(f"  Mean: {tensor_normalized.mean().item():.3f}")

# Test 2: Without normalization (old version)
transform_no_norm = T.Compose([
    T.ToTensor()
])
tensor_no_norm = transform_no_norm(image_resized)

print("\nWITHOUT normalization:")
print(f"  Min: {tensor_no_norm.min().item():.3f}")
print(f"  Max: {tensor_no_norm.max().item():.3f}")
print(f"  Mean: {tensor_no_norm.mean().item():.3f}")

print("\n" + "="*50)
print("If working script gives 0.4-0.9 scores,")
print("it probably uses the version that gives HIGHER scores.")
print("="*50)
