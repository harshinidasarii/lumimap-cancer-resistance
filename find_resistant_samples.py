"""
Find Resistant Samples for Demo - SIMPLE VERSION
=================================================

Just shows which samples are resistant without checking image paths
"""

import pandas as pd
import os

def main():
    print("="*70)
    print("🔍 FINDING RESISTANT SAMPLES FOR DEMO")
    print("="*70)
    print()
    
    # Load results
    results_path = './output/phase2_strategic/phase2_strategic_results.csv'
    alt_path = './output/phase2_strategic/resistance_labels.csv'
    
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
    elif os.path.exists(alt_path):
        df = pd.read_csv(alt_path)
        print(f"✓ Using: {alt_path}")
        print()
    else:
        print(f"❌ Results file not found!")
        print(f"   Looked for: {results_path}")
        print(f"   Also tried: {alt_path}")
        return
    
    # Show distribution
    print("📊 Resistance Distribution:")
    for rtype in ['SENSITIVE', 'PARTIAL_RESISTANCE', 'CROSS_RESISTANCE', 'PRIMARY_RESISTANCE']:
        count = (df['resistance_type'] == rtype).sum()
        pct = 100 * count / len(df)
        print(f"   {rtype:25s}: {count:4d} ({pct:5.1f}%)")
    
    print()
    print("="*70)
    print("🎯 GOOD SAMPLES FOR DEMO")
    print("="*70)
    print()
    
    # Check each resistance type
    for rtype in ['PARTIAL_RESISTANCE', 'CROSS_RESISTANCE', 'PRIMARY_RESISTANCE']:
        resistant = df[df['resistance_type'] == rtype].copy()
        
        if len(resistant) == 0:
            continue
            
        print(f"{rtype}:")
        print(f"   Total: {len(resistant)} samples")
        print(f"   Good examples (idx to use for demo):")
        
        # Show top 5
        for idx in resistant.index[:5]:
            row = resistant.loc[idx]
            
            # Get compound and MOA if available
            compound = row.get('compound_name', 'Unknown')
            moa = row.get('moa', 'Unknown')
            sim_dmso = row.get('dmso_similarity', row.get('sim_dmso', 0))
            sim_moa = row.get('moa_similarity', row.get('sim_moa', 0))
            
            print(f"      idx={idx:4d}  |  {compound:22s} |  MOA: {moa:35s}")
            print(f"                DMSO sim: {sim_dmso:.3f}  |  MOA sim: {sim_moa:.3f}")
        
        print()
    
    print("="*70)
    print("🚀 HOW TO RUN DEMO")
    print("="*70)
    print()
    print("Try these commands:")
    print()
    
    # Find first of each type
    for rtype in ['PARTIAL_RESISTANCE', 'CROSS_RESISTANCE', 'PRIMARY_RESISTANCE']:
        resistant = df[df['resistance_type'] == rtype]
        
        if len(resistant) > 0:
            idx = resistant.index[0]
            print(f"# {rtype}")
            print(f"python demo_with_gradcam.py --idx {idx}")
            print()
    
    print("="*70)
    print()
    print("💡 TIP: These samples worked earlier:")
    print("   python demo_with_gradcam.py --idx 36  # CROSS_RESISTANCE")
    print("   python demo_with_gradcam.py --idx 41  # PARTIAL_RESISTANCE")
    print()
    print("="*70)
    print()
    
    # Summary
    print("📋 WHAT THE DEMO SHOWS:")
    print()
    print("🔬 Images:")
    print("   • 3-channel fluorescence microscopy")
    print("   • DAPI (blue) = nucleus")
    print("   • Tubulin (green) = microtubules")
    print("   • Actin (red) = cytoskeleton")
    print()
    print("🎨 GradCAM:")
    print("   • Heatmap showing WHERE AI looks")
    print("   • Proves AI learned biology!")
    print()
    print("💊 Classification:")
    print("   • SENSITIVE - Drug working")
    print("   • PARTIAL - Some resistance")
    print("   • CROSS - Different drug would work")
    print("   • PRIMARY - Multi-resistant")
    print()
    print("="*70)

if __name__ == "__main__":
    main()