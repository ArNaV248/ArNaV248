#!/usr/bin/env python3
"""
Helper script to collect all D-FINE source files for creating a standalone inference script.
Run this on your Mac to gather all necessary files.
"""

import os
import sys

# Path to your D-FINE source
DFINE_PATH = "/Users/borde/label_studio_local/src/d_fine/"

# Files we need
REQUIRED_FILES = [
    "dfine.py",
    "configs.py",
    "utils.py",
    "matcher.py",
    "arch/hgnetv2.py",
    "arch/hybrid_encoder.py",
    "arch/dfine_decoder.py",
    "__init__.py",
    "arch/__init__.py",
]

def read_file(filepath):
    """Read file and return content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    print("="*70)
    print("D-FINE SOURCE CODE COLLECTOR")
    print("="*70)
    print()

    output_file = "dfine_sources_bundle.txt"

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("# D-FINE Source Code Bundle\n")
        out.write("# Generated for creating standalone inference script\n")
        out.write("="*70 + "\n\n")

        for rel_path in REQUIRED_FILES:
            full_path = os.path.join(DFINE_PATH, rel_path)

            print(f"Reading: {rel_path}...", end=" ")

            if not os.path.exists(full_path):
                print(f"❌ NOT FOUND")
                out.write(f"\n{'='*70}\n")
                out.write(f"FILE: {rel_path}\n")
                out.write(f"STATUS: NOT FOUND at {full_path}\n")
                out.write(f"{'='*70}\n\n")
                continue

            content = read_file(full_path)

            if content is None:
                print(f"❌ ERROR")
                continue

            print(f"✅ ({len(content)} chars)")

            # Write to bundle file
            out.write(f"\n{'='*70}\n")
            out.write(f"FILE: {rel_path}\n")
            out.write(f"{'='*70}\n\n")
            out.write(content)
            out.write("\n\n")

    print()
    print("="*70)
    print(f"✅ All source files collected in: {output_file}")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Open the file:", output_file)
    print("2. Copy the entire contents")
    print("3. Paste it in your response to Claude")
    print()

if __name__ == "__main__":
    main()
