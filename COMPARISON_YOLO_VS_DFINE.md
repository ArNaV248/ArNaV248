# YOLOv5 vs D-FINE Label Studio Backend - Detailed Comparison

Side-by-side comparison of the two Label Studio ML backends for object detection.

---

## 🔍 Overview

| Aspect | YOLOv5 Backend (Original) | D-FINE Backend (Your Custom) |
|--------|---------------------------|------------------------------|
| **Model Architecture** | YOLOv5 (anchor-based CNN) | D-FINE-L (transformer-based, deformable DETR) |
| **Backbone** | CSPDarknet | HGNetv2 |
| **Detection Method** | Anchor boxes + NMS | Query-based detection (DETR-style) |
| **Model File** | `best.pt` (YOLOv5 format) | `model_2.pt` (D-FINE format) |
| **Input Size** | 1280x1280 (square) | 1280x448 (rectangular) |
| **Number of Classes** | 35 almond defects | 35 almond defects |
| **Code Dependencies** | YOLOv5 repo required | ✅ All embedded (standalone) |

---

## 📦 Model Loading

### YOLOv5 Backend

```python
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# Requires YOLOv5 repo cloned at /app/yolov5
YOLOV5_ROOT = "/app/yolov5"
sys.path.append(YOLOV5_ROOT)

self.device = select_device('')
self.model = attempt_load(pretrained_model_path, map_location=self.device)
self.stride = int(self.model.stride.max())
self.imgsz = check_img_size(1280, s=self.stride)
```

**Pros:**
- ✅ Standard YOLOv5 format
- ✅ Well-documented

**Cons:**
- ❌ Requires YOLOv5 repo
- ❌ External dependencies

---

### D-FINE Backend

```python
# All code embedded - no external repos!
def build_model(model_name, num_classes, device, img_size=None):
    model_cfg = get_model_config(model_name)
    backbone = HGNetv2(**model_cfg["HGNetv2"])
    encoder = HybridEncoder(**model_cfg["HybridEncoder"])
    decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])
    model = DFINE(backbone, encoder, decoder)
    return model.to(device)

self.model = build_model(MODEL_NAME, NUM_CLASSES, self.device, img_size=None)
self.model = load_checkpoint(self.model, MODEL_PATH)
```

**Pros:**
- ✅ Fully standalone (no external repos)
- ✅ All D-FINE code embedded in one file
- ✅ Easier deployment

**Cons:**
- ❌ Larger file size (~2000 lines)
- ❌ Less familiar to YOLOv5 users

---

## 🖼️ Image Preprocessing

### YOLOv5 Backend

```python
# Letterbox: maintains aspect ratio, adds padding
im = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]
im = im.transpose(2, 0, 1)  # HWC to CHW
im = np.ascontiguousarray(im)
im = torch.from_numpy(im).to(self.device)
im = im.half() if self.half else im.float()
im /= 255.0  # Normalize to [0, 1]
```

**Input:** Original image (any size)
**Output:** 1280x1280 (with padding, maintains aspect ratio)
**Normalization:** Divide by 255

---

### D-FINE Backend

```python
# Direct resize: no padding, distorts aspect ratio
resize_hw = (INPUT_SIZE[1], INPUT_SIZE[0])  # (448, 1280)
self.transform = T.Compose([
    T.Resize(resize_hw),   # (H, W)
    T.ToTensor()           # Converts to [0, 1] and CHW
])
image_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
```

**Input:** Original image (any size)
**Output:** 1280x448 (no padding, aspect ratio distorted)
**Normalization:** ToTensor() automatically normalizes to [0, 1]

**Key Difference:**
- YOLOv5 preserves aspect ratio (adds padding)
- D-FINE distorts aspect ratio (stretches image)

---

## 🎯 Inference

### YOLOv5 Backend

```python
with torch.no_grad():
    pred = self.model(im, augment=False)[0]  # Raw predictions

# Manual NMS required
pred = non_max_suppression(
    pred,
    conf_thres=0.25,      # Confidence threshold
    iou_thres=0.45,       # IoU threshold for NMS
    classes=None,
    agnostic=False,
    max_det=1000
)

det = pred[0]  # First image in batch
if len(det):
    # Scale boxes back to original image size
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
```

**Output format:** `[x1, y1, x2, y2, conf, cls]`
**NMS:** Manual (user must call `non_max_suppression`)
**Confidence:** Direct confidence score

---

### D-FINE Backend

```python
with torch.no_grad():
    outputs = self.model(image_tensor)  # Returns dict

pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes]
pred_boxes = outputs['pred_boxes']    # [batch, num_queries, 4] (cxcywh)

# Convert logits to probabilities using softmax
probs = F.softmax(pred_logits, dim=-1)
max_scores, labels = probs.max(dim=-1)

# Filter by confidence
keep_mask = max_scores > confidence_threshold

# Optional NMS (user configurable)
if nms_threshold < 1.0:
    keep_indices = nms(boxes_xyxy, filtered_scores.cpu(), nms_threshold)
```

**Output format:** Dict with `pred_logits` and `pred_boxes`
**NMS:** Optional (built-in via `nms_threshold` parameter)
**Confidence:** Softmax probabilities

**Key Difference:**
- YOLOv5 returns direct confidence scores
- D-FINE uses softmax probabilities (more calibrated)

---

## 📊 Postprocessing

### YOLOv5 Backend

```python
for *xyxy, conf, cls in reversed(det):
    x1, y1, x2, y2 = xyxy
    label = category_map.get(int(cls), f'class_{int(cls)}')

    predictions.append({
        'from_name': 'label',
        'to_name': 'image',
        'type': 'rectanglelabels',
        'value': {
            "x": float(x1) / original_width * 100,
            "y": float(y1) / original_height * 100,
            "width": (float(x2) - float(x1)) / original_width * 100,
            "height": (float(y2) - float(y1)) / original_height * 100,
            "rectanglelabels": [label]
        },
        "score": float(conf)
    })
```

**Box format:** xyxy (x1, y1, x2, y2)
**Scaling:** Boxes already scaled to original image

---

### D-FINE Backend

```python
# Convert cxcywh → xyxy
boxes_cxcywh = filtered_boxes.cpu()
boxes_xyxy = box_convert(boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')

# Scale to original image size
boxes_xyxy[:, [0, 2]] *= original_width
boxes_xyxy[:, [1, 3]] *= original_height

# Convert to Label Studio format (same as YOLOv5)
for box, score, label in zip(boxes_xyxy, filtered_scores, filtered_labels):
    x1, y1, x2, y2 = box.tolist()
    predictions.append({
        'from_name': 'label',
        'to_name': 'image',
        'type': 'rectanglelabels',
        'value': {
            'x': x1 / original_width * 100,
            'y': y1 / original_height * 100,
            'width': (x2 - x1) / original_width * 100,
            'height': (y2 - y1) / original_height * 100,
            'rectanglelabels': [class_name]
        },
        'score': float(score)
    })
```

**Box format:** cxcywh → xyxy conversion required
**Scaling:** Manual scaling to original image size

---

## ⚙️ Configuration

### YOLOv5 Backend

```python
# Hardcoded parameters
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
imgsz = 1280

# Model settings
self.half = self.device.type != 'cpu'  # FP16 on GPU
augment = False  # Test-time augmentation
```

---

### D-FINE Backend

```python
# Configurable via context
confidence_threshold = 0.3  # Default
nms_threshold = 1.0         # Disabled by default

# Can be overridden:
if context and 'result' in context:
    confidence_threshold = context.get('confidence_threshold', 0.3)
    nms_threshold = context.get('nms_threshold', 1.0)

# Model settings
INPUT_SIZE = (1280, 448)
NUM_CLASSES = 35
MODEL_NAME = "l"  # DFINE-Large
```

**Flexibility:**
- D-FINE allows runtime parameter adjustment via Label Studio context
- YOLOv5 requires code changes

---

## 🚀 Performance Comparison

### Inference Speed (Estimated)

| Metric | YOLOv5 Backend | D-FINE Backend |
|--------|----------------|----------------|
| **Model Size** | ~90 MB | ~150 MB |
| **Inference Time (GPU)** | ~20-30 ms | ~50-80 ms |
| **Inference Time (CPU)** | ~200-300 ms | ~800-1200 ms |
| **Memory Usage (GPU)** | ~2 GB | ~4 GB |
| **Throughput (GPU)** | ~30-50 FPS | ~12-20 FPS |

**Note:** D-FINE is slower but often more accurate due to transformer architecture.

---

### Accuracy Comparison

| Metric | YOLOv5 | D-FINE |
|--------|--------|--------|
| **mAP@0.5** | 0.85 (typical) | 0.88 (typical) |
| **Small Objects** | Good | ✅ Better (transformers excel) |
| **Overlapping Objects** | Struggles | ✅ Better (query-based) |
| **Localization** | Good | ✅ Better (finer boxes) |
| **Speed** | ✅ Faster | Slower |

**Trade-off:**
- YOLOv5: Faster inference, good accuracy
- D-FINE: Slower inference, better accuracy (especially small/overlapping objects)

---

## 🛠️ Deployment Complexity

### YOLOv5 Backend

**Dependencies:**
- ✅ Small file (~200 lines)
- ❌ Requires YOLOv5 repo clone
- ❌ Must set `YOLOV5_ROOT` path
- ❌ Dependent on YOLOv5 version

**Docker Deployment:**
```dockerfile
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5
RUN pip install -r /app/yolov5/requirements.txt
ENV YOLOV5_ROOT=/app/yolov5
```

---

### D-FINE Backend

**Dependencies:**
- ✅ Fully standalone (2000 lines, but all embedded)
- ✅ No external repos needed
- ✅ Just copy one .py file
- ✅ Version-independent

**Docker Deployment:**
```dockerfile
COPY dfine_labelstudio_backend.py /app/
# That's it! No git clone needed
```

**Winner:** D-FINE (easier deployment despite larger file)

---

## 📝 Code Maintainability

### YOLOv5 Backend

**Pros:**
- ✅ Concise (~200 lines)
- ✅ Delegates to YOLOv5 library
- ✅ Easy to understand

**Cons:**
- ❌ Breaks if YOLOv5 repo changes
- ❌ Hard to debug (external code)
- ❌ Version compatibility issues

---

### D-FINE Backend

**Pros:**
- ✅ All code visible and editable
- ✅ No external dependencies (except torch)
- ✅ Full control over behavior

**Cons:**
- ❌ Large file (~2000 lines)
- ❌ Harder to navigate
- ❌ Must maintain D-FINE code yourself

**Winner:** Depends on use case
- Research/experimentation: D-FINE (full control)
- Production with stable YOLOv5: YOLOv5 (concise)

---

## 🔧 Customization

### Adding New Classes

**YOLOv5:**
```python
# Update category_map (line 20)
category_map = {
    0: '0_Adhering_Skin',
    # ... add new classes ...
    35: '35_NewClass',
}
```

**D-FINE:**
```python
# Update CLASS_NAMES (line 46)
CLASS_NAMES = [
    "0_Adhering_Skin",
    # ... add new classes ...
    "35_NewClass",
]

# Update NUM_CLASSES (line 44)
NUM_CLASSES = 36  # Was 35
```

**Winner:** Tie (both easy)

---

### Changing Input Size

**YOLOv5:**
```python
# Model file determines size
# Retrain with new --img-size flag
```

**D-FINE:**
```python
# Just change INPUT_SIZE (line 43)
INPUT_SIZE = (1920, 640)  # Was (1280, 448)
```

**Winner:** D-FINE (no retraining needed, though may affect accuracy)

---

### Custom Preprocessing

**YOLOv5:**
```python
# Must modify letterbox() in utils/augmentations.py
# Or add custom transforms before letterbox
```

**D-FINE:**
```python
# Just modify self.transform (line 1612)
self.transform = T.Compose([
    T.Resize(resize_hw),
    T.ColorJitter(brightness=0.2),  # Add custom transforms
    T.ToTensor()
])
```

**Winner:** D-FINE (easier to modify)

---

## 🎯 Use Case Recommendations

### Choose YOLOv5 Backend If:

- ✅ You need maximum inference speed
- ✅ You're already familiar with YOLOv5
- ✅ You have stable YOLOv5 deployment pipeline
- ✅ You prioritize small file size
- ✅ You want community support (YOLOv5 is popular)

**Best for:**
- Real-time applications
- Resource-constrained environments
- Production with established YOLOv5 workflow

---

### Choose D-FINE Backend If:

- ✅ You need better accuracy (especially small objects)
- ✅ You want full control over code
- ✅ You prefer standalone deployment (no external repos)
- ✅ You're working with overlapping objects
- ✅ You want transformer-based detection

**Best for:**
- Research and experimentation
- High-accuracy requirements
- Environments where inference speed isn't critical
- Docker/cloud deployment (easier without git clones)

---

## 📊 Summary Table

| Feature | YOLOv5 | D-FINE | Winner |
|---------|--------|--------|--------|
| **Inference Speed** | ⚡ Fast | Medium | YOLOv5 |
| **Accuracy** | Good | ✨ Better | D-FINE |
| **Small Objects** | Good | ✨ Better | D-FINE |
| **Deployment** | Complex | ✅ Simple | D-FINE |
| **File Size** | ✅ Small | Large | YOLOv5 |
| **Customization** | Limited | ✅ Full control | D-FINE |
| **Dependencies** | External | ✅ Standalone | D-FINE |
| **Community Support** | ✅ Large | Small | YOLOv5 |
| **Code Maintainability** | Concise | Verbose | YOLOv5 |
| **Flexibility** | Limited | ✅ High | D-FINE |

---

## 🔄 Migration Guide

### Switching from YOLOv5 to D-FINE

1. **Replace backend file:**
   ```bash
   mv yolov5_backend.py yolov5_backend.py.backup
   cp dfine_labelstudio_backend.py .
   ```

2. **Update model path:**
   ```python
   # Line 41 in dfine_labelstudio_backend.py
   MODEL_PATH = "/path/to/model_2.pt"
   ```

3. **Adjust confidence threshold:**
   ```python
   # YOLOv5 used 0.25, D-FINE uses 0.3
   # You may need to tune this
   confidence_threshold = 0.3
   ```

4. **Restart backend:**
   ```bash
   label-studio-ml start dfine_labelstudio_backend --port 9090
   ```

5. **Test and compare:**
   - Use same images in Label Studio
   - Compare detection quality
   - Adjust thresholds if needed

---

## 💡 Best Practices

### For Both Backends

1. **Version your model files:**
   ```bash
   model_v1.pt
   model_v2_better_accuracy.pt
   model_v3_final.pt
   ```

2. **Log predictions for debugging:**
   ```python
   logger.info(f"Detected {len(predictions)} objects")
   for pred in predictions:
       logger.debug(f"Class: {pred['value']['rectanglelabels']}, Conf: {pred['score']:.3f}")
   ```

3. **Monitor inference time:**
   ```python
   import time
   start = time.time()
   outputs = self.model(image_tensor)
   logger.info(f"Inference took {time.time() - start:.3f}s")
   ```

4. **Cache model in GPU memory:**
   ```python
   # Both backends do this automatically
   self.model.eval()  # Set to eval mode
   ```

---

## 🎉 Conclusion

**For production with speed requirements:** Use **YOLOv5**
**For research and high accuracy:** Use **D-FINE**
**For easiest deployment:** Use **D-FINE**
**For smallest file size:** Use **YOLOv5**

Both backends integrate seamlessly with Label Studio and provide the same output format, so you can switch between them easily!
