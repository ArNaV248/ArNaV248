# Logits Comparison Guide

## Goal
Identify why the standalone script produces different confidence scores than the working script for the same objects.

## The Problem
Same bounding boxes getting different confidence scores:
- **Working script**: 7_Doubles at [459, 217, 552, 304] = **0.9978**
- **Standalone script**: 7_Doubles at [458, 218, 554, 309] = **0.8451**

**That's a 15% difference for the SAME object!**

## Diagnostic Approach

We need to capture the **raw model output (logits)** BEFORE any softmax/sigmoid is applied. This will tell us if:

1. **Logits are the same** → Problem is in score calculation (postprocessing)
2. **Logits are different** → Problem is in preprocessing or model loading

---

## Step-by-Step Instructions

### Step 1: Run Standalone Script (DONE)

The standalone script now saves raw logits automatically:

```bash
python3 dfine_standalone_inference.py --input test.jpg --conf 0.1 --nms 1.0
```

This creates: `logits_test.pt` (or whatever your image name is)

---

### Step 2: Modify Your Working Script

**Option A: Use the provided modified script**

1. Open `working_script_with_diagnostics.py`
2. **UNCOMMENT** and fix the model loading code (around line 26-28):
   ```python
   # REPLACE THIS:
   # self.model = build_model(...)

   # WITH YOUR ACTUAL CODE:
   from d_fine import build_model  # Your actual imports
   self.model = build_model(model_name="l", num_classes=len(class_names),
                           device=device, pretrained_model_path=model_path)
   ```

3. Update the model path if needed (line 96)
4. Run it:
   ```bash
   python3 working_script_with_diagnostics.py
   ```

**Option B: Modify your existing working script**

Add this code RIGHT AFTER your model inference (after `preds = self.model(tensor)`):

```python
# DIAGNOSTIC CODE - ADD THIS
pred_logits = preds['pred_logits']
if isinstance(pred_logits, (list, tuple)):
    pred_logits = pred_logits[-1]

pred_logits_cpu = pred_logits[0].cpu()

# Get image name for the file
image_name = "test"  # Replace with actual image name
output_file = f"logits_working_{image_name}.pt"

torch.save({
    'logits': pred_logits_cpu,
    'shape': pred_logits_cpu.shape,
    'first_query_logits': pred_logits_cpu[0, :].tolist(),
    'first_10_queries_first_10_classes': pred_logits_cpu[:10, :10].tolist()
}, output_file)

print(f"\n✅ SAVED RAW LOGITS TO: {output_file}")
# END DIAGNOSTIC CODE
```

---

### Step 3: Compare the Logits

Once you have BOTH files:
- `logits_test.pt` (from standalone script)
- `logits_working_test.pt` (from working script)

Run the comparison tool:

```bash
python3 compare_logits.py logits_test.pt logits_working_test.pt
```

This will show:
- Are the shapes the same?
- Are the values identical?
- Where are the biggest differences?
- **Most importantly**: Is the problem in preprocessing or score calculation?

---

## What the Results Mean

### Case 1: Logits are IDENTICAL
```
✅ LOGITS ARE IDENTICAL!
→ Model outputs are the same
→ Problem is in SCORE CALCULATION (softmax/sigmoid)
```

**Action**: Compare the `postprocess_outputs()` function line by line. The softmax/scoring logic is different.

### Case 2: Logits are DIFFERENT
```
❌ LOGITS ARE DIFFERENT!
→ Model is producing different outputs
→ Problem is in PREPROCESSING or MODEL LOADING
```

**Action**:
- Check preprocessing (resize, normalization, ToTensor)
- Check model loading (checkpoint format, EMA, device)
- Verify both scripts use the exact same model file

---

## Quick Reference

| File | Purpose |
|------|---------|
| `dfine_standalone_inference.py` | Main standalone script (now saves logits) |
| `working_script_with_diagnostics.py` | Your working script modified to save logits |
| `compare_logits.py` | Tool to compare the two logit files |
| `LOGITS_COMPARISON_GUIDE.md` | This guide |

---

## Troubleshooting

**Q: "I don't know how to modify my working script"**
- Use `working_script_with_diagnostics.py` and just fix the model loading line

**Q: "The comparison tool says shapes don't match"**
- The models are configured differently (different num_classes or architecture)

**Q: "How do I share the logit files?"**
- They're small PyTorch tensor files, you can upload them or share via file transfer

**Q: "Can I just copy-paste the logits here?"**
- The comparison script can print them, or you can do:
  ```python
  import torch
  data = torch.load('logits_test.pt')
  print(data['first_10_queries_first_10_classes'])
  ```

---

## Expected Outcome

After this analysis, we will know EXACTLY where to fix the issue:
- ✅ **If preprocessing**: Fix the image transform pipeline
- ✅ **If score calculation**: Fix the softmax/sigmoid logic
- ✅ **If model loading**: Fix the checkpoint loading

No more guessing!
