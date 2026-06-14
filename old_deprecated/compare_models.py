"""
Compare the actual models loaded by both scripts
"""

import torch
import torch.nn as nn
from torchvision import models
from moa_classifier_train import Config

print("="*70)
print("MODEL COMPARISON")
print("="*70)

# Load checkpoint
checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

# Define correct 13 classes (same as both scripts now use)
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

Config.MOA_CLASSES = CORRECT_13_CLASSES
Config.NUM_CLASSES = 13

# Create model (same way both scripts do)
class MOAClassifierModel(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        # Create full ResNet-50 as backbone
        self.backbone = models.resnet50(pretrained=False)
        
        # Remove the fc layer from backbone
        self.backbone.fc = nn.Identity()
        
        # Custom classifier
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

# Create two instances
print("\n1. Creating Model 1...")
model1 = MOAClassifierModel(num_classes=13)
model1.load_state_dict(state_dict)
model1.eval()

print("2. Creating Model 2...")
model2 = MOAClassifierModel(num_classes=13)
model2.load_state_dict(state_dict)
model2.eval()

# Compare state dicts
print("\n3. Comparing model weights...")
model1_dict = model1.state_dict()
model2_dict = model2.state_dict()

all_match = True
for key in model1_dict.keys():
    if key in model2_dict:
        diff = (model1_dict[key] - model2_dict[key]).abs().max().item()
        if diff > 0:
            print(f"   DIFFERENT: {key} - max diff: {diff}")
            all_match = False
    else:
        print(f"   MISSING in model2: {key}")
        all_match = False

if all_match:
    print("   ✓ All weights are identical!")
else:
    print("   ✗ Models have different weights!")

# Test with same input
print("\n4. Testing with dummy input...")
dummy_input = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    output1 = model1(dummy_input)
    output2 = model2(dummy_input)

output_diff = (output1 - output2).abs().max().item()
print(f"   Output difference: {output_diff}")

if output_diff < 1e-6:
    print("   ✓ Outputs are identical!")
else:
    print("   ✗ Outputs are DIFFERENT!")
    print(f"   Model1 prediction: {output1.argmax().item()}")
    print(f"   Model2 prediction: {output2.argmax().item()}")

# Check if models are in eval mode
print("\n5. Checking model modes...")
print(f"   Model1 training mode: {model1.training}")
print(f"   Model2 training mode: {model2.training}")

# Check dropout states
print("\n6. Checking dropout layers...")
for name, module in model1.named_modules():
    if isinstance(module, nn.Dropout):
        print(f"   Model1 - {name}: p={module.p}, training={module.training}")

for name, module in model2.named_modules():
    if isinstance(module, nn.Dropout):
        print(f"   Model2 - {name}: p={module.p}, training={module.training}")

print("\n" + "="*70)
