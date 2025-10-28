#!/usr/bin/env python3
"""
Quick script to fix CLASS_NAMES in dfine_standalone_inference.py
"""

import re

# Path to your standalone script
SCRIPT_PATH = "/Users/borde/arnav/dfine_standalone_inference.py"

# Correct class names (from working script)
CORRECT_CLASS_NAMES = """CLASS_NAMES = [
    "0_Adhering_Skin", "1_Blanched", "2_Broken_Blanched", "3_Mission", "4_Carmel",
    "5_Chip_Scratch_1_4", "6_Discolor", "7_Doubles", "8_Foreign_Material_Hull",
    "9_NonPareil", "10_OD_Brownspot", "11_SD_Insect_Damage", "12_Specks",
    "13_Split_Broken", "14_LooseSkin_Dust_Particle", "15_OD_Gummy", "16_SD_Pinhole",
    "17_Inshell", "18_Embeddedshell", "19_FM_Other", "20_FM_Rock_Dirtball",
    "21_FM_Pistachio", "22_FM_Walnut", "23_FM_Plastic", "24_FM_Metal",
    "25_FM_Glass", "26_SD_Mold", "27_SD_Decay", "28_SD_Other", "29_SD_Frass",
    "30_OD_Shrivel", "31_OD_Discolor", "32_Chip_Scratch_1_8", "33_California",
    "34_Fold_Deformed",
]"""

def main():
    print("="*70)
    print("CLASS_NAMES FIXER")
    print("="*70)

    # Read the file
    print(f"\n[1/3] Reading file: {SCRIPT_PATH}")
    try:
        with open(SCRIPT_PATH, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {SCRIPT_PATH}")
        return

    # Find and replace CLASS_NAMES
    print(f"\n[2/3] Searching for CLASS_NAMES definition...")

    # Pattern to match CLASS_NAMES = [...] with any content
    pattern = r'CLASS_NAMES\s*=\s*\[[\s\S]*?\]'

    if not re.search(pattern, content):
        print("❌ ERROR: Could not find CLASS_NAMES definition")
        return

    # Show current class names
    current_match = re.search(pattern, content)
    if current_match:
        current_classes = current_match.group(0)
        print(f"\nCurrent CLASS_NAMES (first 100 chars):")
        print(f"   {current_classes[:100]}...")

    # Replace
    new_content = re.sub(pattern, CORRECT_CLASS_NAMES, content)

    # Backup original file
    backup_path = SCRIPT_PATH + ".backup"
    print(f"\n[3/3] Creating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        f.write(content)

    # Write updated file
    print(f"[3/3] Writing updated file: {SCRIPT_PATH}")
    with open(SCRIPT_PATH, 'w') as f:
        f.write(new_content)

    print(f"\n{'='*70}")
    print(f"✅ SUCCESS! CLASS_NAMES updated")
    print(f"{'='*70}")
    print(f"\nNew CLASS_NAMES:")
    print(f"   0_Adhering_Skin, 1_Blanched, 2_Broken_Blanched, 3_Mission, 4_Carmel,")
    print(f"   5_Chip_Scratch_1_4, 6_Discolor, 7_Doubles, 8_Foreign_Material_Hull, ...")
    print(f"\n📋 Next steps:")
    print(f"   1. Re-run standalone script: python3 dfine_standalone_inference.py")
    print(f"   2. Compare logits again: python3 compare_logits.py")
    print(f"\n💾 Backup saved to: {backup_path}")

if __name__ == "__main__":
    main()
