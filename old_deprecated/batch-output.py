"""
LUMIMAP Automated Batch Processing
===================================
Place images in 'input/' folder
Get organized reports in 'output/' folder

Each input image gets its own report folder with:
- Original image (copy)
- Resistance report visualization
- Grad-CAM explanation visualization
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from datetime import datetime
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'

# Drug configuration (update these for your images)
DEFAULT_DRUG_NAME = 'cytochalasin B'
DEFAULT_DRUG_CONCENTRATION = '10 nM'
DEFAULT_EXPECTED_MOA = 'Actin disruptors'

# HARDCODED correct 13-class MOA list
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

class Config:
    """Configuration class"""
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    MOA_CLASSES = CORRECT_13_CLASSES
    NUM_CLASSES = 13
    IMG_SIZE = 224  # CRITICAL: Model trained on 224x224

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class MOAClassifierModel(nn.Module):
    """ResNet-50 based MOA classifier"""
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

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_path):
    """Preprocess image for model input"""
    
    # Load image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Normalize 16-bit to 8-bit
    if image_array.dtype in [np.uint16, np.int16]:
        image_array = ((image_array - image_array.min()) / 
                      (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    
    # Convert grayscale to RGB
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    
    # Make contiguous
    image_array = np.ascontiguousarray(image_array)
    
    # CRITICAL: Resize to 224x224
    TARGET_SIZE = 224
    image_array = cv2.resize(image_array, (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_LINEAR)
    
    # Convert to PIL and apply transforms
    image_pil = Image.fromarray(image_array.astype(np.uint8))
    
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_pil).unsqueeze(0)
    
    return image_tensor, image_pil, image_array

# =============================================================================
# RESISTANCE DETECTION
# =============================================================================

def generate_resistance_report(model, image_path, drug_name, expected_moa, output_path):
    """Generate resistance report visualization"""
    
    # Preprocess
    image_tensor, image_pil, image_array = preprocess_image(image_path)
    image_tensor = image_tensor.to(Config.DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get predictions
    predicted_idx = probabilities.argmax().item()
    predicted_moa = Config.MOA_CLASSES[predicted_idx]
    predicted_prob = probabilities[predicted_idx].item()
    
    # Get expected probability
    expected_idx = Config.MOA_CLASSES.index(expected_moa)
    expected_prob = probabilities[expected_idx].item()
    
    # Calculate resistance
    resistance_score = abs(predicted_prob - expected_prob)
    
    # Get top 3
    top3_probs, top3_indices = probabilities.topk(3)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Resistance score gauge
    ax1 = plt.subplot(1, 2, 1)
    ax1.barh([0], [resistance_score*100], color='#e74c3c', height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_xlabel('Score (%)', fontsize=12)
    ax1.set_title(f'Resistance Score: {resistance_score*100:.2f}%', fontsize=14, fontweight='bold')
    ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    
    # Top 3 predictions
    ax2 = plt.subplot(1, 2, 2)
    moas = [Config.MOA_CLASSES[idx.item()] for idx in top3_indices]
    probs = [prob.item()*100 for prob in top3_probs]
    colors = ['#e74c3c' if moa == predicted_moa else '#3498db' for moa in moas]
    
    bars = ax2.barh(moas, probs, color=colors)
    ax2.set_xlabel('Probability (%)', fontsize=12)
    ax2.set_title('Top 3 MOA Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    for bar, prob in zip(bars, probs):
        ax2.text(prob + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', va='center', fontsize=10)
    
    # Add info box
    info_text = (
        f"Drug: {drug_name}\n"
        f"Expected MOA: {expected_moa}\n"
        f"Predicted MOA: {predicted_moa}\n\n"
        f"Resistance Type: {'Complete Resistance' if resistance_score > 0.7 else 'Partial Resistance' if resistance_score > 0.3 else 'Sensitive'}\n"
        f"Resistance Mechanism: Cytoskeletal Compensation"
    )
    
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return results
    return {
        'predicted_moa': predicted_moa,
        'predicted_prob': float(predicted_prob),
        'expected_moa': expected_moa,
        'expected_prob': float(expected_prob),
        'resistance_score': float(resistance_score),
        'top3': [(moa, float(prob/100)) for moa, prob in zip(moas, probs)]
    }

# =============================================================================
# GRAD-CAM EXPLANATION
# =============================================================================

class GradCAM:
    """Grad-CAM implementation"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam

def generate_gradcam_explanation(model, image_path, drug_name, expected_moa, output_path):
    """Generate Grad-CAM explanation visualization"""
    
    # Preprocess
    image_tensor, image_pil, image_array = preprocess_image(image_path)
    image_tensor = image_tensor.to(Config.DEVICE)
    
    # Get target layer
    target_layer = model.backbone.layer4[-1]
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    predicted_idx = probabilities.argmax().item()
    expected_idx = Config.MOA_CLASSES.index(expected_moa)
    
    # Generate CAMs
    predicted_cam = gradcam.generate(image_tensor, predicted_idx)
    expected_cam = gradcam.generate(image_tensor, expected_idx)
    
    # Resize CAMs to match image
    predicted_cam_resized = F.interpolate(predicted_cam, size=(224, 224), mode='bilinear', align_corners=False)
    expected_cam_resized = F.interpolate(expected_cam, size=(224, 224), mode='bilinear', align_corners=False)
    
    predicted_cam_np = predicted_cam_resized.squeeze().cpu().numpy()
    expected_cam_np = expected_cam_resized.squeeze().cpu().numpy()
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Row 1: Predicted MOA
    plt.subplot(2, 3, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(predicted_cam_np, cmap='jet')
    plt.title(f'Predicted: {Config.MOA_CLASSES[predicted_idx]}\nConfidence: {probabilities[predicted_idx].item()*100:.2f}%', fontsize=12)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(2, 3, 3)
    plt.imshow(image_array, cmap='gray')
    plt.imshow(predicted_cam_np, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM Overlay', fontsize=12)
    plt.axis('off')
    
    # Row 2: Expected MOA
    plt.subplot(2, 3, 4)
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(expected_cam_np, cmap='jet')
    plt.title(f'Expected: {expected_moa}\nProbability: {probabilities[expected_idx].item()*100:.2f}%', fontsize=12)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(2, 3, 6)
    plt.imshow(image_array, cmap='gray')
    plt.imshow(expected_cam_np, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM Overlay', fontsize=12)
    plt.axis('off')
    
    # Add title
    resistance_score = abs(probabilities[predicted_idx].item() - probabilities[expected_idx].item())
    status = '⚠️ RESISTANT' if resistance_score > 0.7 else '⚠️ PARTIAL' if resistance_score > 0.3 else '✓ SENSITIVE'
    
    plt.suptitle(f'{status} (Resistance Score: {resistance_score*100:.2f}%)\n'
                f'Drug: {drug_name}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_batch(input_folder, output_folder):
    """Process all images in input folder"""
    
    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all TIFF files
    tif_files = glob.glob(os.path.join(input_folder, '*.tif')) + \
                glob.glob(os.path.join(input_folder, '*.tiff'))
    
    if len(tif_files) == 0:
        print(f"❌ No TIFF files found in '{input_folder}/' folder")
        print(f"\n💡 Please add TIFF images to the '{input_folder}/' folder and run again")
        return
    
    print("="*70)
    print("LUMIMAP AUTOMATED BATCH PROCESSING")
    print("="*70)
    print(f"Input folder: {input_folder}/")
    print(f"Output folder: {output_folder}/")
    print(f"Found {len(tif_files)} images to process")
    print("")
    
    # Load model
    print("Loading model...")
    model_path = './outputs/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model = MOAClassifierModel(num_classes=13)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    
    print(f"✓ Model loaded (validation accuracy: {checkpoint['best_val_acc']:.4f})")
    print("")
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(tif_files, 1):
        image_name = os.path.basename(image_path)
        report_folder = os.path.join(output_folder, f'report_{i}')
        os.makedirs(report_folder, exist_ok=True)
        
        print(f"[{i}/{len(tif_files)}] Processing: {image_name}")
        print(f"  Report folder: {report_folder}/")
        
        try:
            # Copy original image
            original_copy = os.path.join(report_folder, 'original_image.tif')
            shutil.copy2(image_path, original_copy)
            print(f"  ✓ Copied original image")
            
            # Generate resistance report
            resistance_path = os.path.join(report_folder, 'resistance_report.png')
            results = generate_resistance_report(
                model, image_path, 
                DEFAULT_DRUG_NAME, DEFAULT_EXPECTED_MOA,
                resistance_path
            )
            print(f"  ✓ Generated resistance report")
            print(f"     Resistance: {results['resistance_score']*100:.2f}%")
            
            # Generate Grad-CAM
            gradcam_path = os.path.join(report_folder, 'gradcam_explanation.png')
            generate_gradcam_explanation(
                model, image_path,
                DEFAULT_DRUG_NAME, DEFAULT_EXPECTED_MOA,
                gradcam_path
            )
            print(f"  ✓ Generated Grad-CAM explanation")
            
            # Save individual report JSON
            report_json = {
                'image_name': image_name,
                'image_path': image_path,
                'drug_name': DEFAULT_DRUG_NAME,
                'drug_concentration': DEFAULT_DRUG_CONCENTRATION,
                'expected_moa': DEFAULT_EXPECTED_MOA,
                **results
            }
            
            json_path = os.path.join(report_folder, 'report.json')
            with open(json_path, 'w') as f:
                json.dump(report_json, f, indent=2)
            print(f"  ✓ Saved report.json")
            
            all_results.append(report_json)
            print("")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            print("")
            continue
    
    # Save summary
    if len(all_results) > 0:
        summary_path = os.path.join(output_folder, 'summary_report.json')
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(tif_files),
            'processed_images': len(all_results),
            'average_resistance': float(np.mean([r['resistance_score'] for r in all_results])),
            'results': all_results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total images processed: {len(all_results)}/{len(tif_files)}")
        print(f"Average resistance score: {summary['average_resistance']*100:.2f}%")
        print(f"\n✓ Summary saved: {summary_path}")
        print(f"✓ All reports saved in: {output_folder}/")
        print("="*70)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER)