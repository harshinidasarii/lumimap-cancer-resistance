"""
Diagnostic: Compare image preprocessing between resistance detector and gradcam
"""

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from moa_classifier_train import Config

image_path = 'data/Week2/Week2_24141/Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif'

print("="*70)
print("PREPROCESSING COMPARISON")
print("="*70)

# Method 1: Resistance Detector Style
print("\n1. RESISTANCE DETECTOR PREPROCESSING:")
# Load TIFF
image = Image.open(image_path)
image_array = np.array(image)
print(f"   Original: shape={image_array.shape}, dtype={image_array.dtype}")
print(f"   Min={image_array.min()}, Max={image_array.max()}")

# Normalize if 16-bit
if image_array.dtype in [np.uint16, np.int16, np.float32, np.float64]:
    image_array = ((image_array - image_array.min()) / 
                  (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    print(f"   After 16-bit norm: Min={image_array.min()}, Max={image_array.max()}")

# Ensure RGB
if len(image_array.shape) == 2:
    image_array = np.stack([image_array] * 3, axis=-1)
elif image_array.shape[2] > 3:
    image_array = image_array[:, :, :3]

image_array = np.ascontiguousarray(image_array.astype(np.uint8))
print(f"   After RGB: shape={image_array.shape}")

# Resize
if isinstance(Config.IMG_SIZE, (list, tuple)):
    target_size = (int(Config.IMG_SIZE[0]), int(Config.IMG_SIZE[1]))
else:
    target_size = (int(Config.IMG_SIZE), int(Config.IMG_SIZE))

image_array_resized_v1 = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
print(f"   After resize: shape={image_array_resized_v1.shape}")

# Convert to PIL and apply transforms
image_pil_v1 = Image.fromarray(image_array_resized_v1.astype(np.uint8))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor_v1 = transform(image_pil_v1).unsqueeze(0)
print(f"   Tensor: shape={image_tensor_v1.shape}")
print(f"   Tensor stats: min={image_tensor_v1.min():.4f}, max={image_tensor_v1.max():.4f}")
print(f"   Tensor stats: mean={image_tensor_v1.mean():.4f}, std={image_tensor_v1.std():.4f}")

# Method 2: Grad-CAM Style  
print("\n2. GRAD-CAM PREPROCESSING:")
# Load TIFF
image2 = Image.open(image_path)
image_array2 = np.array(image2)
print(f"   Original: shape={image_array2.shape}, dtype={image_array2.dtype}")

# Normalize if 16-bit
if image_array2.dtype in [np.uint16, np.int16, np.float32, np.float64]:
    image_array2 = ((image_array2 - image_array2.min()) / 
                   (image_array2.max() - image_array2.min()) * 255).astype(np.uint8)

# Ensure RGB
if len(image_array2.shape) == 2:
    image_array2 = np.stack([image_array2] * 3, axis=-1)
elif image_array2.shape[2] > 3:
    image_array2 = image_array2[:, :, :3]

image_array2 = np.ascontiguousarray(image_array2.astype(np.uint8))

# Resize
if isinstance(Config.IMG_SIZE, (list, tuple)):
    target_size2 = int(Config.IMG_SIZE[0])
else:
    target_size2 = int(Config.IMG_SIZE)

image_array_resized_v2 = cv2.resize(image_array2, (target_size2, target_size2), interpolation=cv2.INTER_LINEAR)
print(f"   After resize: shape={image_array_resized_v2.shape}")

# Convert to PIL and apply transforms
image_pil_v2 = Image.fromarray(image_array_resized_v2.astype(np.uint8))
image_tensor_v2 = transform(image_pil_v2).unsqueeze(0)
print(f"   Tensor: shape={image_tensor_v2.shape}")
print(f"   Tensor stats: min={image_tensor_v2.min():.4f}, max={image_tensor_v2.max():.4f}")
print(f"   Tensor stats: mean={image_tensor_v2.mean():.4f}, std={image_tensor_v2.std():.4f}")

# Compare
print("\n3. COMPARISON:")
tensor_diff = (image_tensor_v1 - image_tensor_v2).abs()
print(f"   Tensors identical? {torch.allclose(image_tensor_v1, image_tensor_v2)}")
print(f"   Max difference: {tensor_diff.max():.6f}")
print(f"   Mean difference: {tensor_diff.mean():.6f}")

if torch.allclose(image_tensor_v1, image_tensor_v2):
    print("\n✓ Preprocessing is IDENTICAL")
else:
    print("\n✗ Preprocessing is DIFFERENT!")
    print("   This explains why predictions differ!")

print("="*70)
