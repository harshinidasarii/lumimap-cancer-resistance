"""
Batch Validation Script
========================
Process multiple test images and generate comprehensive validation report
Compare predictions across different drugs and MOA classes
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import from main scripts
import sys
sys.path.insert(0, '.')

# =============================================================================
# CONFIGURATION
# =============================================================================

# HARDCODED correct 13-class MOA list (same as main scripts)
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
# TEST IMAGES CONFIGURATION
# =============================================================================

# Define your 5 validation images here
# Format: (image_path, drug_name, expected_moa)
VALIDATION_IMAGES = [
    # Example 1: Week2 cytochalasin B
    ('data/Week2/Week2_24141/Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    # Example 2: Week1 image
    ('data/Week1/Week1_22123/Week1_150607_B02_s4_w2EE226363-0BAC-443F-A41C-16C9C89FDDFA.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    # Add 3 more images here - just update the paths:
    # ('path/to/your/image3.tif', 'drug_name', 'expected_MOA'),
    # ('path/to/your/image4.tif', 'drug_name', 'expected_MOA'),
    # ('path/to/your/image5.tif', 'drug_name', 'expected_MOA'),
]

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
    
    # Make contiguous for cv2
    image_array = np.ascontiguousarray(image_array)
    
    # CRITICAL: Model was trained on 224x224 images, ALWAYS use this size!
    TARGET_SIZE = 224
    
    # Resize using cv2
    try:
        import cv2
        image_array = cv2.resize(image_array, (TARGET_SIZE, TARGET_SIZE),
                                interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Fallback to scipy
        from scipy.ndimage import zoom
        h, w = image_array.shape[:2]
        zoom_h = TARGET_SIZE / h
        zoom_w = TARGET_SIZE / w
        if len(image_array.shape) == 3:
            image_array = zoom(image_array, (zoom_h, zoom_w, 1), order=1)
        else:
            image_array = zoom(image_array, (zoom_h, zoom_w), order=1)
    
    # Convert to PIL and apply transforms
    image_pil = Image.fromarray(image_array.astype(np.uint8))
    
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_pil).unsqueeze(0)
    
    return image_tensor, image_pil

# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_image(model, image_path, drug_name, expected_moa):
    """Predict on a single image and return results"""
    
    # Preprocess
    image_tensor, image_pil = preprocess_image(image_path)
    image_tensor = image_tensor.to(Config.DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get top prediction
    predicted_idx = probabilities.argmax().item()
    predicted_moa = Config.MOA_CLASSES[predicted_idx]
    predicted_prob = probabilities[predicted_idx].item()
    
    # Get expected MOA probability
    expected_idx = Config.MOA_CLASSES.index(expected_moa)
    expected_prob = probabilities[expected_idx].item()
    
    # Calculate resistance score
    resistance_score = abs(predicted_prob - expected_prob) * 100
    
    # Get top 3 predictions
    top3_probs, top3_indices = probabilities.topk(3)
    top3 = [(Config.MOA_CLASSES[idx.item()], prob.item()) 
            for idx, prob in zip(top3_indices, top3_probs)]
    
    return {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'drug_name': drug_name,
        'expected_moa': expected_moa,
        'expected_prob': expected_prob,
        'predicted_moa': predicted_moa,
        'predicted_prob': predicted_prob,
        'resistance_score': resistance_score,
        'top3_predictions': top3,
        'all_probabilities': {Config.MOA_CLASSES[i]: probabilities[i].item() 
                             for i in range(len(Config.MOA_CLASSES))}
    }

# =============================================================================
# BATCH PROCESSING
# =============================================================================

def validate_batch(model, validation_images):
    """Process batch of validation images"""
    
    print("="*70)
    print("BATCH VALIDATION REPORT")
    print("="*70)
    print(f"Number of images: {len(validation_images)}")
    print(f"Model: ResNet-50 (13 MOA classes)")
    print(f"Device: {Config.DEVICE}")
    print("")
    
    results = []
    
    for i, (image_path, drug_name, expected_moa) in enumerate(validation_images, 1):
        print(f"\n[{i}/{len(validation_images)}] Processing: {os.path.basename(image_path)}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"  ⚠️  File not found: {image_path}")
            continue
        
        try:
            # Predict
            result = predict_single_image(model, image_path, drug_name, expected_moa)
            results.append(result)
            
            # Print summary
            print(f"  Drug: {drug_name}")
            print(f"  Expected: {expected_moa} ({result['expected_prob']*100:.2f}%)")
            print(f"  Predicted: {result['predicted_moa']} ({result['predicted_prob']*100:.2f}%)")
            print(f"  Resistance: {result['resistance_score']:.2f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_batch_report(results):
    """Create visual comparison report"""
    
    if len(results) == 0:
        print("No results to visualize")
        return
    
    n_images = len(results)
    
    # Create figure
    fig = plt.figure(figsize=(16, 4*n_images))
    
    for i, result in enumerate(results):
        # Top 3 predictions bar chart
        ax = plt.subplot(n_images, 2, i*2 + 1)
        moas = [m for m, p in result['top3_predictions']]
        probs = [p*100 for m, p in result['top3_predictions']]
        colors = ['#2ecc71' if m == result['predicted_moa'] else '#3498db' 
                 for m in moas]
        
        bars = ax.barh(moas, probs, color=colors)
        ax.set_xlabel('Probability (%)')
        ax.set_title(f'{result["image_name"][:30]}...\nTop 3 Predictions')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            ax.text(prob + 1, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.1f}%', va='center')
        
        # Resistance score
        ax2 = plt.subplot(n_images, 2, i*2 + 2)
        resistance = result['resistance_score']
        
        # Color based on resistance level
        if resistance < 30:
            color = '#2ecc71'  # Green - sensitive
            status = 'SENSITIVE'
        elif resistance < 70:
            color = '#f39c12'  # Orange - partial
            status = 'PARTIAL RESISTANCE'
        else:
            color = '#e74c3c'  # Red - resistant
            status = 'RESISTANT'
        
        ax2.barh([0], [resistance], color=color, height=0.5)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.set_xlabel('Resistance Score (%)')
        ax2.set_title(f'{status}\nResistance: {resistance:.1f}%')
        ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        
        # Add text info
        info_text = (f"Drug: {result['drug_name']}\n"
                    f"Expected: {result['expected_moa']}\n"
                    f"Predicted: {result['predicted_moa']}")
        ax2.text(5, 0, info_text, va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = f'batch_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visual report saved: {output_path}")
    
    return output_path

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(results):
    """Print summary statistics"""
    
    if len(results) == 0:
        return
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Overall accuracy (top-1)
    correct = sum(1 for r in results if r['predicted_moa'] == r['expected_moa'])
    accuracy = correct / len(results) * 100
    print(f"\nTop-1 Accuracy: {accuracy:.1f}% ({correct}/{len(results)})")
    
    # Average resistance score
    avg_resistance = np.mean([r['resistance_score'] for r in results])
    print(f"Average Resistance Score: {avg_resistance:.2f}%")
    
    # Resistance distribution
    sensitive = sum(1 for r in results if r['resistance_score'] < 30)
    partial = sum(1 for r in results if 30 <= r['resistance_score'] < 70)
    resistant = sum(1 for r in results if r['resistance_score'] >= 70)
    
    print(f"\nResistance Distribution:")
    print(f"  Sensitive (<30%):        {sensitive} ({sensitive/len(results)*100:.1f}%)")
    print(f"  Partial (30-70%):        {partial} ({partial/len(results)*100:.1f}%)")
    print(f"  Resistant (>70%):        {resistant} ({resistant/len(results)*100:.1f}%)")
    
    # Per-drug breakdown
    print(f"\nPer-Drug Breakdown:")
    drugs = {}
    for r in results:
        drug = r['drug_name']
        if drug not in drugs:
            drugs[drug] = []
        drugs[drug].append(r)
    
    for drug, drug_results in drugs.items():
        avg_res = np.mean([r['resistance_score'] for r in drug_results])
        print(f"  {drug}: {len(drug_results)} images, avg resistance {avg_res:.1f}%")
    
    print("="*70)

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main batch validation function"""
    
    print("\n" + "="*70)
    print("LUMIMAP BATCH VALIDATION")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
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
    
    # Validate images
    print(f"\n{'='*70}")
    print(f"Processing {len(VALIDATION_IMAGES)} validation images...")
    print(f"{'='*70}")
    
    results = validate_batch(model, VALIDATION_IMAGES)
    
    # Print summary
    print_summary_statistics(results)
    
    # Create visualization
    if len(results) > 0:
        print("\nCreating visual report...")
        output_path = create_batch_report(results)
    
    # Save JSON
    if len(results) > 0:
        json_path = f'batch_validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ JSON results saved: {json_path}")
    
    print("\n" + "="*70)
    print("BATCH VALIDATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
