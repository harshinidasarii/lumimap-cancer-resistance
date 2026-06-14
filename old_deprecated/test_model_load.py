"""
Quick test script to verify the model loads correctly
"""

import torch

print("=" * 60)
print("MODEL LOADING TEST")
print("=" * 60)

# Load the checkpoint
checkpoint_path = './outputs/best_model.pth'
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check architecture
state_dict = checkpoint['model_state_dict']
first_key = list(state_dict.keys())[0]
last_key = list(state_dict.keys())[-1]

# Get validation accuracy with fallback
val_acc = checkpoint.get('val_acc') or checkpoint.get('best_val_acc') or checkpoint.get('accuracy')

print(f"\n✓ Checkpoint loaded!")
if val_acc:
    if isinstance(val_acc, float):
        print(f"  Validation accuracy: {val_acc:.4f}")
    else:
        print(f"  Validation accuracy: {val_acc}")
else:
    print(f"  Validation accuracy: Not found in checkpoint")
print(f"  First key: {first_key}")
print(f"  Last key: {last_key}")

# Detect num classes
if 'classifier.6.weight' in state_dict:
    num_classes = state_dict['classifier.6.weight'].shape[0]
    print(f"  Architecture: WRAPPED (backbone + classifier)")
    print(f"  Num classes: {num_classes}")
elif 'fc.4.weight' in state_dict:
    num_classes = state_dict['fc.4.weight'].shape[0]
    print(f"  Architecture: STANDARD (ResNet + fc)")
    print(f"  Num classes: {num_classes}")
else:
    print(f"  Architecture: UNKNOWN")
    print(f"  Available keys: {list(state_dict.keys())[:5]}...")

print("\n" + "=" * 60)
print("Now run: python resistance_detector_inference.py")
print("=" * 60)
