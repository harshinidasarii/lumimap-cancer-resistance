"""
LUMIMAP - Grad-CAM Explainer
Generates visual explanations showing which cellular features the AI examines
Provides explainability and transparency for clinical use
NOW WITH: Spatial nanoparticle targeting based on resistance hotspots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Import Config from training script
from moa_classifier_train import Config

# Import centralized test configuration
from test_config import TEST_IMAGE, DRUG_NAME, DRUG_CONCENTRATION, EXPECTED_MOA

# Import spatial nanoparticle targeting
import sys
sys.path.insert(0, '.')
try:
    from spatial_nanoparticle_targeting import SpatialNanoparticleTargeting
except ImportError:
    print("Warning: Spatial targeting module not found, continuing without it")
    SpatialNanoparticleTargeting = None

# ============================================
# GRAD-CAM IMPLEMENTATION
# ============================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = model_output[:, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), target_class

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def apply_colormap(cam, original_image):
    """Apply heatmap to original image"""
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert image to RGB if grayscale
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Superimpose
    overlay = heatmap * 0.4 + original_image * 0.6
    overlay = overlay.astype(np.uint8)
    
    return heatmap, overlay

def visualize_gradcam(image_path, model, drug_info=None, save_path='gradcam_explanation.png'):
    """Generate comprehensive Grad-CAM visualization"""
    
    # Load and preprocess image using same method as resistance detector
    # Load image
    if image_path.endswith('.tif') or image_path.endswith('.tiff'):
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Handle different bit depths
        if image_array.dtype in [np.uint16, np.int16, np.float32, np.float64]:
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
        # Ensure RGB
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] > 3:
            image_array = image_array[:, :, :3]
        
        image_array = np.ascontiguousarray(image_array.astype(np.uint8))
        original_image = Image.fromarray(image_array)
    else:
        original_image = Image.open(image_path).convert('RGB')
        image_array = np.array(original_image)
    
    # Resize using cv2
    # CRITICAL: Model was trained on 224x224 images, ALWAYS use this size!
    TARGET_SIZE = 224
    print(f"  Resizing to {TARGET_SIZE}×{TARGET_SIZE} (model training size)")
    image_array_resized = cv2.resize(image_array, (TARGET_SIZE, TARGET_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
    
    # Convert to PIL for transforms
    image_pil = Image.fromarray(image_array_resized.astype(np.uint8))
    
    # Apply transforms (just ToTensor and Normalize, no resize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_pil).unsqueeze(0).to(Config.DEVICE)
    original_np = image_array_resized
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # DEBUG: Print all predictions
        print("\n" + "="*70)
        print("DEBUG - ALL PREDICTIONS (Grad-CAM):")
        print("="*70)
        sorted_indices = probabilities.argsort(descending=True)
        for i in range(len(Config.MOA_CLASSES)):
            idx = sorted_indices[i]
            print(f"{i+1:2d}. {Config.MOA_CLASSES[idx]:35s} {probabilities[idx].item()*100:6.2f}%")
        print("="*70 + "\n")
        
        predicted_idx = probabilities.argmax().item()
        predicted_moa = Config.MOA_CLASSES[predicted_idx]
        predicted_conf = probabilities[predicted_idx].item()
    
    # Initialize Grad-CAM - handle wrapped vs standard model
    if hasattr(model, 'backbone'):
        # Wrapped model - backbone is a full ResNet-50, so it has layer4
        target_layer = model.backbone.layer4
    else:
        # Standard model
        target_layer = model.layer4
    
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM for predicted class
    cam_predicted, _ = gradcam.generate_cam(image_tensor, predicted_idx)
    heatmap_pred, overlay_pred = apply_colormap(cam_predicted, original_np)
    
    # Generate CAM for expected class if provided
    if drug_info:
        expected_moa = drug_info['expected_moa']
        expected_idx = Config.MOA_CLASSES.index(expected_moa)
        expected_prob = probabilities[expected_idx].item()
        
        cam_expected, _ = gradcam.generate_cam(image_tensor, expected_idx)
        heatmap_exp, overlay_exp = apply_colormap(cam_expected, original_np)
        
        # Calculate resistance
        resistance_score = (1 - expected_prob) * 100
    
    # Create visualization
    if drug_info:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Determine resistance status
        if resistance_score >= 75:
            status_color = 'red'
            status_text = 'RESISTANT'
        elif resistance_score >= 25:
            status_color = 'orange'
            status_text = 'PARTIAL RESISTANCE'
        else:
            status_color = 'green'
            status_text = 'SENSITIVE'
        
        fig.suptitle(
            f'⚠️ {status_text} (Resistance Score: {resistance_score:.2f}%)\n'
            f'Drug: {drug_info["drug"]}',
            fontsize=16,
            fontweight='bold',
            color=status_color
        )
        
        # Row 1: Predicted MOA
        axes[0, 0].imshow(original_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(heatmap_pred)
        axes[0, 1].set_title(f'Predicted: {predicted_moa}\nConfidence: {predicted_conf*100:.2f}%')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(overlay_pred)
        axes[0, 2].set_title('Grad-CAM Overlay')
        axes[0, 2].axis('off')
        
        # Row 2: Expected MOA
        axes[1, 0].imshow(original_np)
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(heatmap_exp)
        axes[1, 1].set_title(f'Expected: {expected_moa}\nProbability: {expected_prob*100:.2f}%')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(overlay_exp)
        axes[1, 2].set_title('Grad-CAM Overlay')
        axes[1, 2].axis('off')
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Predicted: {predicted_moa} ({predicted_conf*100:.2f}%)', 
                     fontsize=14, fontweight='bold')
        
        axes[0].imshow(original_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_pred)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay_pred)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Explanation saved to {save_path}")
    
    # Generate spatial nanoparticle targeting report
    if SpatialNanoparticleTargeting is not None and drug_info:
        print("\nAnalyzing spatial nanoparticle targeting...")
        targeting_system = SpatialNanoparticleTargeting()
        
        # Use the predicted CAM for targeting analysis
        targeting_report = targeting_system.generate_targeting_report(
            cam_predicted,
            resistance_mechanism=f"Resistance via {predicted_moa} morphology",
            predicted_moa=predicted_moa,
            recommended_drug=drug_info['drug']
        )
        
        print(targeting_report)
        
        # Save targeting report to file
        with open('spatial_targeting_report.txt', 'w') as f:
            f.write(targeting_report)
        print("✓ Spatial targeting report saved to spatial_targeting_report.txt")

# ============================================
# MAIN
# ============================================

def main():
    print("Loading trained model...")
    
    # Load checkpoint first to detect architecture
    checkpoint = torch.load('./outputs/best_model.pth', map_location=Config.DEVICE)
    state_dict = checkpoint['model_state_dict']
    first_key = list(state_dict.keys())[0]
    
    # CRITICAL: Set correct MOA_CLASSES BEFORE creating model
    # The checkpoint has 13 classes, so we need the correct 13-class list
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
    
    # Check if wrapped model
    if 'backbone' in first_key:
        print("Detected WRAPPED model (backbone + classifier)")
        
        # Use MOAClassifierModel architecture
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
        
        # Get num classes
        num_classes = state_dict['classifier.6.weight'].shape[0]
        print(f"  Num classes: {num_classes}")
        
        # ALWAYS use the correct 13-class list
        Config.MOA_CLASSES = CORRECT_13_CLASSES
        Config.NUM_CLASSES = 13
        print(f"  ✓ Using correct 13-class MOA list")
        
        model = MOAClassifierModel(num_classes=num_classes)
        model.load_state_dict(state_dict)
        
    else:
        print("Detected STANDARD model")
        # Standard ResNet-50
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, Config.NUM_CLASSES)
        )
        model.load_state_dict(state_dict)
    
    model.to(Config.DEVICE)
    model.eval()
    print("✓ Model loaded")
    
    # Test configuration (configured in test_config.py)
    image_path = TEST_IMAGE
    print(f"✓ Using test configuration:")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Drug: {DRUG_NAME} ({DRUG_CONCENTRATION})")
    print(f"  Expected MOA: {EXPECTED_MOA}")
    
    # Drug info for resistance detection
    drug_info = {
        'drug': DRUG_NAME,
        'expected_moa': EXPECTED_MOA
    }
    
    print(f"\nLoading image: {image_path}")
    print("Generating Grad-CAM explanation...")
    visualize_gradcam(image_path, model, drug_info)

if __name__ == '__main__':
    main()
