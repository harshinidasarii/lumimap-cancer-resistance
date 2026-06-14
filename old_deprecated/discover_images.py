"""
Image Discovery Script
======================
Scan your data folder and suggest validation images
"""

import os
import glob

def discover_images(data_dir='data'):
    """Discover all TIFF images in data directory"""
    
    print("="*70)
    print("DISCOVERING AVAILABLE IMAGES")
    print("="*70)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("\nPlease make sure your data is in a 'data/' folder")
        return
    
    # Find all TIF files
    tif_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))
    
    if len(tif_files) == 0:
        print(f"❌ No TIFF files found in {data_dir}")
        return
    
    print(f"\n✓ Found {len(tif_files)} TIFF files")
    
    # Organize by channel
    channels = {'w1': [], 'w2': [], 'w4': [], 'w5': [], 'other': []}
    
    for filepath in tif_files:
        filename = os.path.basename(filepath)
        if 'w1' in filename:
            channels['w1'].append(filepath)
        elif 'w2' in filename:
            channels['w2'].append(filepath)
        elif 'w4' in filename:
            channels['w4'].append(filepath)
        elif 'w5' in filename:
            channels['w5'].append(filepath)
        else:
            channels['other'].append(filepath)
    
    print("\nImages by channel:")
    print(f"  w1 (Hoechst/nuclei):  {len(channels['w1'])} images")
    print(f"  w2 (ER):              {len(channels['w2'])} images")
    print(f"  w4 (Actin):           {len(channels['w4'])} images ← Best for morphology!")
    print(f"  w5 (Tubulin):         {len(channels['w5'])} images ← Best for morphology!")
    print(f"  Other:                {len(channels['other'])} images")
    
    # Organize by week
    weeks = {}
    for filepath in tif_files:
        filename = os.path.basename(filepath)
        if 'Week1' in filename:
            week = 'Week1'
        elif 'Week2' in filename:
            week = 'Week2'
        elif 'Week3' in filename:
            week = 'Week3'
        else:
            week = 'Other'
        
        if week not in weeks:
            weeks[week] = []
        weeks[week].append(filepath)
    
    print("\nImages by week:")
    for week in sorted(weeks.keys()):
        print(f"  {week}: {len(weeks[week])} images")
    
    # Suggest 5 validation images
    print("\n" + "="*70)
    print("SUGGESTED VALIDATION IMAGES")
    print("="*70)
    
    # Prefer w4 (actin) channel
    suggested_channel = 'w4' if len(channels['w4']) >= 5 else 'w5'
    if len(channels[suggested_channel]) < 5:
        suggested_channel = 'other'
    
    suggested = channels[suggested_channel][:5]
    
    if len(suggested) < 5:
        print(f"⚠️  Only {len(suggested)} images available")
        print("    Consider using images from multiple channels")
        # Add from other channels
        for ch in ['w4', 'w5', 'w2', 'w1', 'other']:
            while len(suggested) < 5 and len(channels[ch]) > 0:
                img = channels[ch].pop(0)
                if img not in suggested:
                    suggested.append(img)
    
    print("\nAdd these to batch_validation.py:")
    print("\nVALIDATION_IMAGES = [")
    
    for i, filepath in enumerate(suggested[:5], 1):
        filename = os.path.basename(filepath)
        print(f"    # Image {i}")
        print(f"    ('{filepath}',")
        print(f"     'cytochalasin B',  # Update drug name if different")
        print(f"     'Actin disruptors'),  # Update expected MOA if different")
        print()
    
    print("]")
    
    # Show full list of first 10 images
    print("\n" + "="*70)
    print("ALL AVAILABLE IMAGES (first 10):")
    print("="*70)
    
    for i, filepath in enumerate(tif_files[:10], 1):
        print(f"{i:2d}. {filepath}")
    
    if len(tif_files) > 10:
        print(f"    ... and {len(tif_files)-10} more")
    
    print("\n" + "="*70)
    print(f"Total: {len(tif_files)} images found")
    print("="*70)
    
    return suggested

if __name__ == '__main__':
    discover_images()
