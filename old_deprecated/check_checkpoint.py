"""
Diagnostic script to check checkpoint file contents
"""

import torch

checkpoint_path = './outputs/best_model.pth'

print("=" * 60)
print("CHECKPOINT DIAGNOSTIC")
print("=" * 60)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\n1. CHECKPOINT KEYS:")
print(f"   {list(checkpoint.keys())}")

print(f"\n2. CHECKPOINT INFO:")
for key in checkpoint.keys():
    if key != 'model_state_dict':
        print(f"   {key}: {checkpoint[key]}")

print(f"\n3. MODEL STATE_DICT KEYS:")
state_dict = checkpoint['model_state_dict']
print(f"   Total keys: {len(state_dict.keys())}")
print(f"\n   First 20 keys:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    tensor_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"   {i+1}. {key}: {tensor_shape}")

print(f"\n   Last 10 keys:")
for i, key in enumerate(list(state_dict.keys())[-10:]):
    tensor_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"   {i+1}. {key}: {tensor_shape}")

# Check for specific keys
print(f"\n4. CHECKING FOR EXPECTED KEYS:")
expected_keys = [
    'conv1.weight',
    'layer1.0.conv1.weight',
    'layer4.0.conv1.weight',
    'fc.0.weight',  # Our custom classifier
    'fc.2.weight',
    'fc.4.weight'
]

for key in expected_keys:
    exists = key in state_dict
    symbol = "✓" if exists else "✗"
    print(f"   {symbol} {key}")

print("\n" + "=" * 60)
