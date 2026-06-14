"""
Find ALL Resistance Types - Including SENSITIVE and PRIMARY
============================================================
"""

import pandas as pd
import os

def main():
    # Load results
    results_path = './output/phase2_strategic/phase2_strategic_results.csv'
    alt_path = './output/phase2_strategic/resistance_labels.csv'
    
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
    elif os.path.exists(alt_path):
        df = pd.read_csv(alt_path)
    else:
        print("❌ Results file not found!")
        return
    
    print("="*70)
    print("🔬 ALL 4 RESISTANCE TYPES - DEMO COMMANDS")
    print("="*70)
    print()
    
    # For each type, find first few examples
    types = {
        'SENSITIVE': '✅ Drug Working (Expected Response)',
        'PARTIAL_RESISTANCE': '⚠️  Partial Response (Some Resistance)',
        'CROSS_RESISTANCE': '🔄 Cross-Resistance (Different Drug Works)',
        'PRIMARY_RESISTANCE': '❌ Multi-Drug Resistant'
    }
    
    for rtype, description in types.items():
        resistant = df[df['resistance_type'] == rtype]
        count = len(resistant)
        
        print(f"{rtype}:")
        print(f"  {description}")
        print(f"  Total: {count} samples")
        print()
        
        if count > 0:
            print(f"  📋 Example indices:")
            # Show first 5
            for i, idx in enumerate(resistant.index[:5]):
                row = resistant.loc[idx]
                compound = row.get('compound_name', 'Unknown')
                moa = row.get('moa', 'Unknown')
                sim_dmso = row.get('dmso_similarity', row.get('sim_dmso', 0))
                sim_moa = row.get('moa_similarity', row.get('sim_moa', 0))
                
                print(f"     idx={idx:4d}  |  {compound[:20]:20s}  |  DMSO:{sim_dmso:.3f} MOA:{sim_moa:.3f}")
            
            print()
            print(f"  🚀 Demo command:")
            first_idx = resistant.index[0]
            print(f"     python demo_with_gradcam.py --idx {first_idx}")
            print()
        
        print("-"*70)
        print()
    
    print()
    print("="*70)
    print("💡 RECOMMENDED DEMO SEQUENCE")
    print("="*70)
    print()
    
    # Find one of each type
    for rtype in ['SENSITIVE', 'PARTIAL_RESISTANCE', 'CROSS_RESISTANCE', 'PRIMARY_RESISTANCE']:
        resistant = df[df['resistance_type'] == rtype]
        if len(resistant) > 0:
            idx = resistant.index[0]
            emoji = {'SENSITIVE': '✅', 'PARTIAL_RESISTANCE': '⚠️', 
                    'CROSS_RESISTANCE': '🔄', 'PRIMARY_RESISTANCE': '❌'}[rtype]
            print(f"{emoji} {rtype}:")
            print(f"   python demo_with_gradcam.py --idx {idx}")
            print()
    
    print("="*70)
    print()
    print("📊 WHAT EACH TYPE SHOWS:")
    print()
    print("✅ SENSITIVE:")
    print("   • Cell died from drug (as expected)")
    print("   • High similarity to expected response")
    print("   • Recommendation: Continue treatment")
    print()
    print("⚠️  PARTIAL_RESISTANCE:")
    print("   • Some cells responding, some not")
    print("   • Medium similarity to expected response")
    print("   • Recommendation: Increase dose or add combination")
    print()
    print("🔄 CROSS_RESISTANCE:")
    print("   • NOT responding to current drug")
    print("   • HIGH similarity to DIFFERENT drug mechanism")
    print("   • Recommendation: Switch to that specific drug!")
    print()
    print("❌ PRIMARY_RESISTANCE:")
    print("   • Not responding to ANY drug mechanism")
    print("   • Low similarity across all mechanisms")
    print("   • Recommendation: Try alternatives or clinical trial")
    print()
    print("="*70)

if __name__ == "__main__":
    main()