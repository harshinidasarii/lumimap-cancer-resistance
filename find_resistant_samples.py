"""
Find Resistant Samples
======================

Finds samples with different resistance types:
- PARTIAL_RESISTANCE
- CROSS_RESISTANCE  
- PRIMARY_RESISTANCE

Shows indices you can use for demo!
"""

import pandas as pd
import random

# Load Phase 2 labels
labels_df = pd.read_csv('./output/phase2_strategic/resistance_labels.csv')

print("="*70)
print("🔍 FINDING RESISTANT SAMPLES FOR DEMO")
print("="*70)

# Count resistance types
resistance_counts = labels_df['resistance_type'].value_counts()

print("\n📊 Resistance Distribution:")
for rtype, count in resistance_counts.items():
    pct = (count / len(labels_df)) * 100
    print(f"   {rtype:25s}: {count:4d} ({pct:5.1f}%)")

print("\n" + "="*70)
print("🎯 GOOD SAMPLES FOR DEMO")
print("="*70)

# Find examples of each resistance type
resistance_types = ['PARTIAL_RESISTANCE', 'CROSS_RESISTANCE', 'PRIMARY_RESISTANCE']

for rtype in resistance_types:
    samples = labels_df[labels_df['resistance_type'] == rtype]
    
    if len(samples) > 0:
        print(f"\n{rtype}:")
        print(f"   Total: {len(samples)} samples")
        
        # Pick 5 good examples (high confidence)
        if rtype == 'PARTIAL_RESISTANCE':
            # Medium MOA similarity
            good_samples = samples[(samples['moa_similarity'] > 0.65) & (samples['moa_similarity'] < 0.80)]
        elif rtype == 'CROSS_RESISTANCE':
            # High cross-MOA similarity
            good_samples = samples[samples.get('cross_moa_similarity', 0) > 0.75] if 'cross_moa_similarity' in samples.columns else samples
        else:  # PRIMARY_RESISTANCE
            # Low all similarities
            good_samples = samples[samples['moa_similarity'] < 0.65]
        
        if len(good_samples) == 0:
            good_samples = samples
        
        # Show up to 5 examples
        examples = good_samples.head(5)
        
        print(f"   Good examples (idx to use for demo):")
        for _, row in examples.iterrows():
            print(f"      idx={row['idx']:5d}  |  {row['compound']:20s}  |  MOA: {row['moa']:30s}")
            print(f"                DMSO sim: {row['dmso_similarity']:.3f}  |  MOA sim: {row['moa_similarity']:.3f}")
    else:
        print(f"\n{rtype}: No samples found")

print("\n" + "="*70)
print("🚀 HOW TO RUN DEMO ON RESISTANT SAMPLES")
print("="*70)

# Pick one of each type
print("\nTry these commands:\n")

for rtype in resistance_types:
    samples = labels_df[labels_df['resistance_type'] == rtype]
    if len(samples) > 0:
        idx = samples.iloc[0]['idx']
        print(f"# {rtype}")
        print(f"python demo_with_gradcam.py --idx {idx}\n")

print("="*70)
