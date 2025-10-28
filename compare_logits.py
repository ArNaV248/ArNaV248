"""
Compare raw logits from standalone script vs working script

Usage:
    python3 compare_logits.py logits_<image>.pt logits_working_<image>.pt
"""

import torch
import sys
import numpy as np

def compare_logits(standalone_path, working_path):
    print("="*70)
    print("LOGITS COMPARISON TOOL")
    print("="*70)

    # Load both files
    print(f"\n[1/3] Loading logits...")
    print(f"  Standalone: {standalone_path}")
    print(f"  Working:    {working_path}")

    try:
        standalone = torch.load(standalone_path)
        working = torch.load(working_path)
    except Exception as e:
        print(f"\n❌ ERROR loading files: {e}")
        return

    # Extract logits
    standalone_logits = standalone['logits']
    working_logits = working['logits']

    print(f"\n[2/3] Comparing shapes...")
    print(f"  Standalone shape: {standalone_logits.shape}")
    print(f"  Working shape:    {working_logits.shape}")

    if standalone_logits.shape != working_logits.shape:
        print(f"\n❌ SHAPES DON'T MATCH!")
        print(f"   This means the models are configured differently.")
        return

    print(f"  ✅ Shapes match!")

    # Compare actual values
    print(f"\n[3/3] Comparing values...")

    # Calculate differences
    diff = torch.abs(standalone_logits - working_logits)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check if they're identical
    are_identical = torch.allclose(standalone_logits, working_logits, rtol=1e-5, atol=1e-5)

    print(f"\n  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if are_identical:
        print(f"\n  ✅ LOGITS ARE IDENTICAL!")
        print(f"\n  🔍 CONCLUSION:")
        print(f"     → Model outputs are the same")
        print(f"     → Problem is in SCORE CALCULATION (softmax/sigmoid)")
        print(f"     → Check the postprocess_outputs() function")
    elif max_diff < 0.001:
        print(f"\n  ✅ LOGITS ARE VERY CLOSE (numerical precision differences)")
        print(f"\n  🔍 CONCLUSION:")
        print(f"     → Model outputs are essentially the same")
        print(f"     → Problem is in SCORE CALCULATION")
    else:
        print(f"\n  ❌ LOGITS ARE DIFFERENT!")
        print(f"\n  🔍 CONCLUSION:")
        print(f"     → Model is producing different outputs")
        print(f"     → Problem is in PREPROCESSING or MODEL LOADING")

    # Show sample comparisons
    print(f"\n" + "="*70)
    print(f"DETAILED COMPARISON (First query, first 10 classes)")
    print(f"="*70)

    print(f"\nStandalone logits:")
    print(standalone_logits[0, :10].numpy())

    print(f"\nWorking logits:")
    print(working_logits[0, :10].numpy())

    print(f"\nDifference:")
    print(diff[0, :10].numpy())

    # Find queries with biggest differences
    print(f"\n" + "="*70)
    print(f"QUERIES WITH BIGGEST DIFFERENCES (Top 10)")
    print(f"="*70)

    query_diffs = diff.max(dim=1)[0]  # Max diff per query
    top_diff_indices = torch.argsort(query_diffs, descending=True)[:10]

    for i, idx in enumerate(top_diff_indices, 1):
        query_idx = idx.item()
        query_max_diff = query_diffs[idx].item()

        # Find which class has the biggest difference
        class_idx = diff[query_idx].argmax().item()

        standalone_val = standalone_logits[query_idx, class_idx].item()
        working_val = working_logits[query_idx, class_idx].item()

        print(f"{i}. Query {query_idx}, Class {class_idx}:")
        print(f"   Standalone: {standalone_val:.6f}")
        print(f"   Working:    {working_val:.6f}")
        print(f"   Difference: {query_max_diff:.6f}")

    # Compare softmax scores
    print(f"\n" + "="*70)
    print(f"SOFTMAX SCORE COMPARISON (First query)")
    print(f"="*70)

    standalone_probs = torch.softmax(standalone_logits[0], dim=-1)
    working_probs = torch.softmax(working_logits[0], dim=-1)

    standalone_max_score, standalone_max_label = standalone_probs.max(dim=-1)
    working_max_score, working_max_label = working_probs.max(dim=-1)

    print(f"\nStandalone: Class {standalone_max_label.item()}, Score {standalone_max_score.item():.6f}")
    print(f"Working:    Class {working_max_label.item()}, Score {working_max_score.item():.6f}")

    if standalone_max_label != working_max_label:
        print(f"\n⚠️  PREDICTED CLASS IS DIFFERENT!")
    elif abs(standalone_max_score.item() - working_max_score.item()) > 0.01:
        print(f"\n⚠️  SAME CLASS BUT DIFFERENT SCORE!")

    print(f"\n" + "="*70)
    print(f"SUMMARY")
    print(f"="*70)

    if are_identical or max_diff < 0.001:
        print(f"\n✅ The model outputs are the same.")
        print(f"✅ The problem is NOT in preprocessing.")
        print(f"❌ The problem is in the score calculation or filtering.")
        print(f"\n📋 Next steps:")
        print(f"   1. Compare the postprocess_outputs() function line by line")
        print(f"   2. Check if softmax is applied the same way")
        print(f"   3. Check if filtering/NMS is different")
    else:
        print(f"\n❌ The model outputs are different.")
        print(f"❌ The problem is in preprocessing or model loading.")
        print(f"\n📋 Next steps:")
        print(f"   1. Compare preprocessing (resize, normalization)")
        print(f"   2. Check model loading (checkpoint, EMA, etc.)")
        print(f"   3. Verify both use the same model file")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_logits.py <standalone_logits.pt> <working_logits.pt>")
        print("\nExample:")
        print("  python3 compare_logits.py logits_test.pt logits_working_test.pt")
        sys.exit(1)

    standalone_path = sys.argv[1]
    working_path = sys.argv[2]

    compare_logits(standalone_path, working_path)
