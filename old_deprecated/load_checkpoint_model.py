"""
Model loader that matches the saved checkpoint architecture
"""

import torch
import torch.nn as nn
from torchvision import models

class MOAClassifierModel(nn.Module):
    """Model architecture that matches the checkpoint"""
    def __init__(self, num_classes=13):
        super().__init__()
        
        # Load ResNet-50 as backbone
        resnet = models.resnet50(pretrained=False)
        
        # Use all of ResNet except the final fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom classifier (matches checkpoint: 2048 -> 512 -> 256 -> 13)
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
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def load_checkpoint_model(checkpoint_path):
    """Load the model with the checkpoint architecture"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check num classes from checkpoint
    num_classes = checkpoint['model_state_dict']['classifier.6.weight'].shape[0]
    print(f"Detected {num_classes} classes in checkpoint")
    
    # Create model with correct architecture
    model = MOAClassifierModel(num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded successfully!")
    print(f"  Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, num_classes

if __name__ == '__main__':
    model, num_classes = load_checkpoint_model('./outputs/best_model.pth')
    print(f"\nModel loaded with {num_classes} classes")
    print(f"Architecture: 2048 -> 512 -> 256 -> {num_classes}")
