"""
Show Input Files for a Given IDX
=================================

This script shows you exactly which image files are used for any idx
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, required=True, help='Sample index')
    args = parser.parse_args()
    
    # Load metadata
    metadata_path = 'data/BBBC021_v1_image.csv'
    
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"❌ Metadata file not found: {metadata_path}")
        print("   Make sure BBBC021_v1_image.csv is in the data/ folder")
        return
    
    if args.idx >= len(df):
        print(f"❌ idx {args.idx} is out of range (max: {len(df)-1})")
        return
    
    # Get sample info
    sample = df.iloc[args.idx]
    
    print("="*70)
    print(f"📋 SAMPLE INFORMATION FOR idx={args.idx}")
    print("="*70)
    print()
    
    print(f"🧪 Compound: {sample.get('Image_Metadata_Compound', 'Unknown')}")
    print(f"💊 MOA: {sample.get('Image_Metadata_MOA', 'Unknown')}")
    print(f"📊 Concentration: {sample.get('Image_Metadata_Concentration', 'Unknown')}")
    print()
    
    print(f"📅 Experiment Info:")
    print(f"   Week: {sample.get('Image_Week', 'Unknown')}")
    print(f"   Plate: {sample.get('Image_Plate', 'Unknown')}")
    print(f"   Well: {sample.get('Image_Metadata_Well', 'Unknown')}")
    print(f"   Site: {sample.get('Image_Metadata_Site', 'Unknown')}")
    print()
    
    print("="*70)
    print("📸 INPUT IMAGE FILES")
    print("="*70)
    print()
    
    # Get filenames
    dapi = sample.get('Image_FileName_DAPI', 'Not found')
    tubulin = sample.get('Image_FileName_Tubulin', 'Not found')
    actin = sample.get('Image_FileName_Actin', 'Not found')
    
    week = sample.get('Image_Week', '?')
    plate = sample.get('Image_Plate', '?')
    
    print(f"🔵 DAPI (Nucleus - Blue channel):")
    print(f"   Filename: {dapi}")
    print(f"   Path: data/Week{week}/Week{week}_{plate}/{dapi}")
    print()
    
    print(f"🟢 Tubulin (Microtubules - Green channel):")
    print(f"   Filename: {tubulin}")
    print(f"   Path: data/Week{week}/Week{week}_{plate}/{tubulin}")
    print()
    
    print(f"🔴 Actin (Cytoskeleton - Red channel):")
    print(f"   Filename: {actin}")
    print(f"   Path: data/Week{week}/Week{week}_{plate}/{actin}")
    print()
    
    print("="*70)
    print()
    
    print("💡 These are the 3 microscopy images that get loaded when you run:")
    print(f"   python demo_with_gradcam.py --idx {args.idx}")
    print()
    print("="*70)

if __name__ == "__main__":
    main()
