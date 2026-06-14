"""
View BBBC021 Metadata CSV
==========================

Shows information from the metadata CSV file
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, help='Show specific sample (optional)')
    parser.add_argument('--head', type=int, default=10, help='Show first N rows')
    args = parser.parse_args()
    
    # Try multiple possible locations
    possible_paths = [
        'data/BBBC021_v1_image.csv',
        '../data/BBBC021_v1_image.csv',
        'BBBC021_v1_image.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            csv_path = path
            break
        except FileNotFoundError:
            continue
    
    if csv_path is None:
        print("❌ Could not find BBBC021_v1_image.csv")
        print()
        print("Looked in:")
        for path in possible_paths:
            print(f"  - {path}")
        print()
        print("💡 The file should be in your data/ folder")
        return
    
    print("="*70)
    print(f"📊 BBBC021 METADATA CSV")
    print("="*70)
    print()
    print(f"✓ Found: {csv_path}")
    print(f"✓ Total samples: {len(df)}")
    print()
    
    if args.idx is not None:
        # Show specific sample
        if args.idx >= len(df):
            print(f"❌ idx {args.idx} out of range (max: {len(df)-1})")
            return
        
        print("="*70)
        print(f"SAMPLE idx={args.idx}")
        print("="*70)
        print()
        
        sample = df.iloc[args.idx]
        
        # Show key fields
        for col in df.columns:
            value = sample[col]
            if pd.notna(value):  # Only show non-null values
                print(f"{col:40s}: {value}")
        
        print()
        print("="*70)
        
    else:
        # Show first N rows summary
        print("="*70)
        print(f"FIRST {args.head} SAMPLES (Summary)")
        print("="*70)
        print()
        
        # Show important columns only
        important_cols = [
            'Image_FileName_DAPI',
            'Image_FileName_Tubulin', 
            'Image_FileName_Actin'
        ]
        
        # Filter to columns that exist
        cols_to_show = [c for c in important_cols if c in df.columns]
        
        if cols_to_show:
            print(df[cols_to_show].head(args.head).to_string())
        else:
            print(df.head(args.head))
        
        print()
        print("="*70)
        print()
        print("💡 To see full details for a specific sample:")
        print("   python view_csv.py --idx 36")
        print()

if __name__ == "__main__":
    main()
