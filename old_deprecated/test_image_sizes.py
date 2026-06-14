"""
Test if image size is the issue
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Load model
checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

CORRECT_13_CLASSES = [
    'Actin disruptors',
    'Aurora kinase inhibitors',
    'Cholesterol-lowering',
    'DMSO',
    'DNA damage',
    'DNA replication',
    'Eg5 inhibitors',
    'Kinase inhibitors',
    'Microtubule destabilizers',
    'Microtubule stabilizers',
    'PKC activators',
    'Protein degradation',
    'Protein synthesis'
]

class MOAClassifierModel(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

model = MOAClassifierModel(num_classes=13)
model.load_state_dict(state_dict)
model.eval()

# Load test image
image_path = 'data/Week2/Week2_24141/Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif'
image = Image.open(image_path)
image_array = np.array(image)

# Normalize 16-bit
if image_array.dtype in [np.uint16, np.int16]:
    image_array = ((image_array - image_array.min()) / 
                  (image_array.max() - image_array.min()) * 255).astype(np.uint8)

# Ensure RGB
if len(image_array.shape) == 2:
    image_array = np.stack([image_array] * 3, axis=-1)

print("="*70)
print("TESTING DIFFERENT IMAGE SIZES")
print("="*70)

for size in [224, 256, 384, 512]:
    print(f"\n{size}×{size}:")
    
    # Resize
    resized = cv2.resize(image_array, (size, size), interpolation=cv2.INTER_LINEAR)
    
    # To tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_img = Image.fromarray(resized.astype(np.uint8))
    tensor = transform(pil_img).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Top 3
    top3_probs, top3_indices = probabilities.topk(3)
    for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs), 1):
        print(f"  {i}. {CORRECT_13_CLASSES[idx]:30s} {prob.item()*100:6.2f}%")

print("\n" + "="*70)
print("CONCLUSION:")
print("If predictions change dramatically with size, that's the bug!")
print("="*70)
