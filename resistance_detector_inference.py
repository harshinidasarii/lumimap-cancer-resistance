"""
LUMIMAP - Resistance Detector
Detects drug resistance by comparing predicted vs expected MOA
Provides quantitative 0-100% resistance scoring and treatment recommendations
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

# Import centralized test configuration
from test_config import TEST_IMAGE, DRUG_NAME, DRUG_CONCENTRATION, EXPECTED_MOA

# Import morphology analysis
from morphology_analyzer import analyze_cell_morphology, classify_resistance_with_morphology

# Import centralized drug database
from drug_database import DRUG_DATABASE, get_drug_info, list_drugs_by_moa

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for resistance detection"""
    # Image settings
    IMG_SIZE = 224  # Single integer, not tuple!
    
    # MOA Classes (13 classes from your model)
    MOA_CLASSES = [
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
    
    NUM_CLASSES = len(MOA_CLASSES)
    
    # Device
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    OUTPUT_DIR = './outputs'

# ============================================
# DRUG DATABASE
# ============================================
# Now imported from centralized drug_database.py
# Contains ~70 drugs organized by MOA class
# To add custom drugs: use drug_database.add_custom_drug()

ALTERNATIVE_DRUGS = {
    'Actin disruptors': [
        ('cisplatin', 'DNA damage'),
        ('paclitaxel', 'Microtubule stabilizers'),
        ('doxorubicin', 'DNA damage'),
        ('gemcitabine', 'DNA replication'),
        ('bortezomib', 'Protein degradation')
    ],
    'Microtubule stabilizers': [
        ('doxorubicin', 'DNA damage'),
        ('cisplatin', 'DNA damage'),
        ('gemcitabine', 'DNA replication'),
        ('cytarabine', 'DNA replication'),
        ('bortezomib', 'Protein degradation')
    ],
    'DNA damage': [
        ('paclitaxel', 'Microtubule stabilizers'),
        ('vincristine', 'Microtubule destabilizers'),
        ('bortezomib', 'Protein degradation'),
        ('cytarabine', 'DNA replication'),
        ('monastrol', 'Eg5 inhibitors')
    ],
    'DMSO': [
        ('paclitaxel', 'Microtubule stabilizers'),
        ('doxorubicin', 'DNA damage'),
        ('cytochalasin B', 'Actin disruptors'),
        ('nocodazole', 'Microtubule destabilizers'),
        ('bortezomib', 'Protein degradation')
    ]
}

# ============================================
# MODEL LOADING
# ============================================

def load_model(model_path):
    """Load trained model - handles checkpoint architecture automatically"""
    import sys
    
    # Check what's in the checkpoint first
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    print(f"  Checkpoint info:")
    print(f"    Keys: {list(checkpoint.keys())}")
    
    # Try different possible validation accuracy keys
    val_acc = checkpoint.get('val_acc') or checkpoint.get('best_val_acc') or checkpoint.get('accuracy')
    if val_acc and isinstance(val_acc, (int, float)):
        print(f"    Val accuracy: {val_acc:.4f}")
    else:
        print(f"    Val accuracy: {val_acc if val_acc else 'Not found'}")
    
    # Check if it's the new format (with backbone/classifier) or old format
    state_dict = checkpoint['model_state_dict']
    first_key = list(state_dict.keys())[0]
    
    if 'backbone' in first_key:
        print(f"  Detected WRAPPED model (backbone + classifier)")
        
        # This is the wrapped model
        # backbone is a direct ResNet-50 model (not Sequential)
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
        
        # Get num classes from checkpoint
        num_classes = state_dict['classifier.6.weight'].shape[0]
        print(f"    Num classes in checkpoint: {num_classes}")
        
        # ALWAYS use the correct 13-class list for this model
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
        
        if num_classes == 13:
            Config.MOA_CLASSES = CORRECT_13_CLASSES
            Config.NUM_CLASSES = 13
            print(f"  ✓ Using correct 13-class MOA list")
        
        model = MOAClassifierModel(num_classes=num_classes)
        
        # Try loading - use strict=False to see what's missing
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"  ✓ Loaded with strict=True")
        except RuntimeError as e:
            print(f"  ⚠️ Strict loading failed, trying strict=False...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys (first 5): {list(missing)[:5]}")
            if unexpected:
                print(f"  Unexpected keys (first 5): {list(unexpected)[:5]}")
            print(f"  ⚠️ Loaded with strict=False (some keys may not match)")
        
    else:
        print(f"  Detected STANDARD model (direct ResNet)")
        
        # Standard ResNet-50 model
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=None)
        except ImportError:
            model = models.resnet50(pretrained=False)
        
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, Config.NUM_CLASSES)
        )
        
        model.load_state_dict(state_dict, strict=False)
    
    model.to(Config.DEVICE)
    model.eval()
    
    print(f"✓ Model loaded!")
    print(f"  Active MOA classes ({len(Config.MOA_CLASSES)}): {Config.MOA_CLASSES[:3]}...{Config.MOA_CLASSES[-1]}")
    
    return model

# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load image - handle TIFF files specially
    if image_path.endswith('.tif') or image_path.endswith('.tiff'):
        # TIFF files can be multi-channel, need special handling
        image = Image.open(image_path)
        
        # Convert to numpy array first
        image_array = np.array(image)
        
        # Handle different bit depths
        if image_array.dtype in [np.uint16, np.int16, np.float32, np.float64]:
            # Normalize to 8-bit
            image_array = ((image_array - image_array.min()) / 
                          (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
        # Ensure RGB (3 channels)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] > 3:  # More than 3 channels
            image_array = image_array[:, :, :3]
        
        # Ensure uint8 type for cv2
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Make contiguous for cv2
        image_array = np.ascontiguousarray(image_array)
        
        # CRITICAL: Model was trained on 224x224 images, ALWAYS use this size!
        TARGET_SIZE = 224
        print(f"  Resizing to {TARGET_SIZE}×{TARGET_SIZE} (model training size)")
        
        # Resize using cv2 if available, otherwise scipy
        try:
            import cv2
            image_array = cv2.resize(image_array, (TARGET_SIZE, TARGET_SIZE), 
                                    interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback to scipy
            from scipy.ndimage import zoom
            h, w = image_array.shape[:2]
            TARGET_SIZE = 224
            zoom_h = TARGET_SIZE / h
            zoom_w = TARGET_SIZE / w
            if len(image_array.shape) == 3:
                image_array = zoom(image_array, (zoom_h, zoom_w, 1), order=1)
            else:
                image_array = zoom(image_array, (zoom_h, zoom_w), order=1)
        
        # Convert back to PIL for transforms
        image = Image.fromarray(image_array.astype(np.uint8))
        
    else:
        # Regular image files - load as numpy array
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Ensure RGB
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        # Ensure uint8
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Make contiguous
        image_array = np.ascontiguousarray(image_array)
        
        # Resize with cv2
        try:
            import cv2
            # Handle IMG_SIZE whether it's int or tuple
            if isinstance(Config.IMG_SIZE, (list, tuple)):
                target_size = (int(Config.IMG_SIZE[0]), int(Config.IMG_SIZE[1]))
            else:
                target_size = (int(Config.IMG_SIZE), int(Config.IMG_SIZE))
            image_array = cv2.resize(image_array, target_size,
                                    interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback: try PIL one more time with basic parameters
            image = Image.fromarray(image_array)
            if isinstance(Config.IMG_SIZE, (list, tuple)):
                target_size = (int(Config.IMG_SIZE[0]), int(Config.IMG_SIZE[1]))
            else:
                target_size = (int(Config.IMG_SIZE), int(Config.IMG_SIZE))
            image = image.resize(target_size)
            image_array = np.array(image)
        
        # Convert to PIL
        image = Image.fromarray(image_array.astype(np.uint8))
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image

# ============================================
# RESISTANCE DETECTION
# ============================================

def detect_resistance(model, image_path, drug_name, expected_moa=None, concentration=None):
    """Detect resistance and generate report
    
    Args:
        model: Trained model
        image_path: Path to cell image
        drug_name: Name of the drug
        expected_moa: (Optional) Expected MOA, overrides database lookup
        concentration: (Optional) Drug concentration, overrides database lookup
    """
    
    # Get drug info from database or use provided values
    if drug_name in DRUG_DATABASE:
        drug_info = DRUG_DATABASE[drug_name]
        if expected_moa is None:
            expected_moa = drug_info['moa']
        if concentration is None:
            concentration = drug_info['concentration']
    else:
        # Use provided values if drug not in database
        if expected_moa is None or concentration is None:
            raise ValueError(
                f"Drug '{drug_name}' not in database. "
                f"Please provide expected_moa and concentration parameters."
            )
    
    # Get expected MOA index
    expected_moa_idx = Config.MOA_CLASSES.index(expected_moa)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(Config.DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # DEBUG: Print all predictions
        print("\n" + "="*70)
        print("DEBUG - ALL PREDICTIONS (Resistance Detector):")
        print("="*70)
        sorted_indices = probabilities.argsort(descending=True)
        for i in range(len(Config.MOA_CLASSES)):
            idx = sorted_indices[i]
            print(f"{i+1:2d}. {Config.MOA_CLASSES[idx]:35s} {probabilities[idx].item()*100:6.2f}%")
        print("="*70 + "\n")
        
        predicted_idx = probabilities.argmax().item()
        predicted_moa = Config.MOA_CLASSES[predicted_idx]
    
    # Calculate resistance score
    expected_prob = probabilities[expected_moa_idx].item()
    resistance_score = (1 - expected_prob) * 100
    
    # Analyze cell morphology for more accurate resistance mechanism classification
    print("Analyzing cell morphology...")
    morphology_features = analyze_cell_morphology(image_path)
    print(f"  Morphology type: {morphology_features['morphology_type'].upper()}")
    print(f"  Elongation: {morphology_features['elongation']:.3f} (0=round, 1=spindle)")
    print(f"  Circularity: {morphology_features['circularity']:.3f} (1=perfect circle)")
    print(f"  Confidence: {morphology_features['morphology_confidence']:.1%}\n")
    
    # Determine resistance type and mechanism (using morphology features)
    if resistance_score < 25:
        resistance_type = "Sensitive"
        resistance_mechanism = "None - Drug Working"
        explanation = "Cells show normal drug response with expected morphological changes"
    elif resistance_score < 75:
        resistance_type = "Partial Resistance"
        resistance_mechanism = classify_resistance_mechanism(
            predicted_moa, expected_moa, resistance_score, "partial", morphology_features
        )
        explanation = f"Cells show partial drug response - {resistance_mechanism} developing"
    else:
        resistance_type = "Complete Resistance"
        resistance_mechanism = classify_resistance_mechanism(
            predicted_moa, expected_moa, resistance_score, "complete", morphology_features
        )
        explanation = f"Cells display {resistance_mechanism} - morphological analysis indicates specific resistance pathway"
    
    # Get top 3 predictions
    top3_probs, top3_indices = probabilities.topk(3)
    top3_predictions = [
        (Config.MOA_CLASSES[idx], prob.item())
        for idx, prob in zip(top3_indices, top3_probs)
    ]
    
    # Generate recommendations
    recommendations = get_alternative_drugs(expected_moa, resistance_score)
    
    # Create report
    report = {
        'drug': drug_name,
        'concentration': concentration,
        'expected_moa': expected_moa,
        'predicted_moa': predicted_moa,
        'predicted_confidence': probabilities[predicted_idx].item(),
        'expected_moa_probability': expected_prob,
        'resistance_score': resistance_score,
        'resistance_type': resistance_type,
        'resistance_mechanism': resistance_mechanism,
        'explanation': explanation,
        'top3_predictions': top3_predictions,
        'recommendations': recommendations,
        'morphology_features': morphology_features  # Include morphology analysis
    }
    
    return report

def get_alternative_drugs(expected_moa, resistance_score):
    """Get alternative drug recommendations"""
    if resistance_score < 25:
        return []
    
    alternatives = ALTERNATIVE_DRUGS.get(expected_moa, [])
    
    recommendations = []
    for drug, moa in alternatives:
        recommendations.append({
            'drug': drug,
            'moa': moa,
            'score': 1.0,
            'rationale': f"Different mechanism from {expected_moa}"
        })
    
    return recommendations[:5]

def classify_resistance_mechanism(predicted_moa, expected_moa, resistance_score, severity, morphology_features=None):
    """
    Classify specific resistance mechanism based on morphological patterns
    Maps predicted MOA to biological resistance types
    
    Args:
        predicted_moa: Predicted MOA from model
        expected_moa: Expected MOA from drug
        resistance_score: Resistance score (0-100)
        severity: "partial" or "complete"
        morphology_features: (Optional) Dict from analyze_cell_morphology()
    """
    
    # Map MOA predictions to resistance mechanisms based on morphological features
    
    # Drug Efflux: Cells appear completely untreated (DMSO-like)
    if predicted_moa == 'DMSO':
        if severity == "complete":
            return "Drug Efflux Resistance (P-glycoprotein overexpression)"
        else:
            return "Early Drug Efflux (increased membrane transporter activity)"
    
    # Apoptosis Resistance: Shows DNA damage but cells don't die
    elif predicted_moa == 'DNA damage' and expected_moa != 'DNA damage':
        if severity == "complete":
            return "Apoptosis Pathway Resistance (Bcl-2 family dysregulation)"
        else:
            return "Emerging Apoptosis Resistance (mitochondrial dysfunction)"
    
    # Metabolic Rewiring: Unexpected metabolic/replication patterns
    elif predicted_moa == 'DNA replication' and expected_moa != 'DNA replication':
        if severity == "complete":
            return "Metabolic Rewiring Resistance (altered energy metabolism)"
        else:
            return "Metabolic Adaptation (mitochondrial compensation)"
    
    # Protein degradation pathway resistance
    elif predicted_moa == 'Protein degradation' and expected_moa != 'Protein degradation':
        if severity == "complete":
            return "Protein Degradation Pathway Resistance (proteasome dysregulation)"
        else:
            return "Protein Homeostasis Alteration (UPR activation)"
    
    # Cytoskeletal resistance: Check ACTUAL morphology if available
    elif predicted_moa in ['Actin disruptors', 'Microtubule destabilizers', 'Microtubule stabilizers']:
        if predicted_moa != expected_moa:
            # Use morphology features if available
            if morphology_features:
                morph_type = morphology_features.get('morphology_type', 'unknown')
                
                if morph_type == 'mesenchymal':
                    # True EMT - cells are spindle-shaped
                    if severity == "complete":
                        return "EMT-like Resistance (cytoskeletal reorganization - mesenchymal)"
                    else:
                        return "Early EMT Transition (cytoskeletal plasticity)"
                        
                elif morph_type == 'epithelial':
                    # Epithelial resistance - cells stay round
                    if severity == "complete":
                        return "Epithelial Resistance (maintaining epithelial phenotype)"
                    else:
                        return "Epithelial Adaptation (cytoskeletal compensation)"
                        
                else:  # intermediate
                    if severity == "complete":
                        return "Cytoskeletal Remodeling Resistance (transitional state)"
                    else:
                        return "Cytoskeletal Plasticity (adaptive remodeling)"
            else:
                # Fallback if no morphology data
                if severity == "complete":
                    return "Cytoskeletal Resistance (EMT-like pattern - needs morphology validation)"
                else:
                    return "Cytoskeletal Plasticity (adaptive remodeling)"
    
    # Targeted therapy resistance: Eg5, Aurora kinase pathways
    elif predicted_moa in ['Eg5 inhibitors', 'Aurora kinase inhibitors']:
        if predicted_moa != expected_moa:
            if severity == "complete":
                return "Targeted Therapy Resistance (compensatory pathway activation)"
            else:
                return "Pathway Compensation (kinase network rewiring)"
    
    # ncRNA-mediated resistance: Protein synthesis changes
    elif predicted_moa == 'Protein synthesis' and expected_moa != 'Protein synthesis':
        if severity == "complete":
            return "ncRNA-mediated Resistance (altered gene expression)"
        else:
            return "Translational Reprogramming (ribosome modification)"
    
    # Endocrine/Hormone resistance (if applicable)
    elif 'DMSO' in predicted_moa and resistance_score > 80:
        return "Hormone Receptor Resistance (signaling pathway bypass)"
    
    # Generic resistance if no specific pattern matches
    else:
        if severity == "complete":
            return "Multi-mechanism Resistance (target mutation or efflux pump)"
        else:
            return "Developing Resistance (multiple pathways)"

# ============================================
# VISUALIZATION
# ============================================

def visualize_resistance_report(report, save_path='resistance_report.png'):
    """Create visual resistance report"""
    # Taller figure to accommodate morphology info
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, height_ratios=[1, 1, 1.5])
    
    # Resistance score gauge
    ax1 = fig.add_subplot(gs[0, 0])
    plot_resistance_gauge(ax1, report['resistance_score'])
    
    # Top 3 MOA predictions
    ax2 = fig.add_subplot(gs[0, 1])
    plot_top3_predictions(ax2, report['top3_predictions'])
    
    # Summary text - now spans bottom section with more height
    ax3 = fig.add_subplot(gs[1:, :])
    plot_summary_text(ax3, report)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visual report saved to {save_path}")

def plot_resistance_gauge(ax, resistance_score):
    """Plot resistance score gauge"""
    # Determine color
    if resistance_score < 25:
        color = 'green'
    elif resistance_score < 75:
        color = 'orange'
    else:
        color = 'red'
    
    ax.barh(0, resistance_score/100, height=0.3, color=color, alpha=0.7)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Threshold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Score')
    ax.set_title(f'Resistance Score: {resistance_score:.2f}%', fontweight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)

def plot_top3_predictions(ax, top3_predictions):
    """Plot top 3 MOA predictions"""
    moas = [pred[0] for pred in top3_predictions]
    probs = [pred[1] for pred in top3_predictions]
    
    colors = ['red' if i == 0 else 'blue' for i in range(len(moas))]
    ax.barh(moas, probs, color=colors, alpha=0.7)
    ax.set_xlabel('Probability')
    ax.set_title('Top 3 MOA Predictions', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

def plot_summary_text(ax, report):
    """Plot summary text box - unified layout"""
    ax.axis('off')
    
    # Build complete summary in one organized block
    summary_lines = [
        f"Drug: {report['drug']}",
        f"Expected MOA: {report['expected_moa']}",
        f"Predicted MOA: {report['predicted_moa']}",
        "",
        f"Resistance Type: {report['resistance_type']}",
        f"Resistance Mechanism: {report['resistance_mechanism']}",
    ]
    
    # Add morphology if available
    if 'morphology_features' in report and report['morphology_features']:
        morph = report['morphology_features']
        summary_lines.extend([
            "",
            "Morphology Analysis:",
            f"  Type: {morph['morphology_type'].upper()}",
            f"  Elongation: {morph['elongation']:.2f} (0=round, 1=spindle)",
            f"  Circularity: {morph['circularity']:.2f} (1=perfect circle)",
        ])
    
    # Add explanation
    summary_lines.extend([
        "",
        "Explanation:",
        f"{report['explanation']}"
    ])
    
    # Add recommendations
    if report['recommendations']:
        summary_lines.extend([
            "",
            "RECOMMENDED ALTERNATIVES:",
        ])
        for i, rec in enumerate(report['recommendations'][:3], 1):
            summary_lines.append(f"{i}. {rec['drug']} ({rec['moa']})")
            summary_lines.append(f"   Rationale: {rec['rationale']}")
    
    # Join all lines
    summary_text = "\n".join(summary_lines)
    
    # Single text box with better positioning
    ax.text(0.05, 0.98, summary_text, 
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("Cancer Resistance Detection System")
    print("=" * 60)
    
    # Load model
    model_path = './outputs/best_model.pth'
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Test image (configured in test_config.py)
    test_image = TEST_IMAGE
    test_drug = DRUG_NAME
    
    print(f"✓ Using test configuration:")
    print(f"  Image: {os.path.basename(test_image)}")
    print(f"  Drug: {test_drug} ({DRUG_CONCENTRATION})")
    print(f"  Expected MOA: {EXPECTED_MOA}")
    
    print(f"\nAnalyzing image...")
    
    # Detect resistance (using values from test_config.py)
    report = detect_resistance(
        model, 
        test_image, 
        test_drug,
        expected_moa=EXPECTED_MOA,
        concentration=DRUG_CONCENTRATION
    )
    
    # Print report
    print("\n" + "=" * 60)
    print("CANCER RESISTANCE DETECTION REPORT")
    print("=" * 60)
    print(f"Drug: {report['drug']}")
    print(f"Concentration: {report['concentration']}")
    print(f"Expected MOA: {report['expected_moa']}")
    
    print("\nRESISTANCE STATUS:")
    print("-" * 60)
    if report['resistance_score'] >= 25:
        print("⚠️  RESISTANCE DETECTED")
    else:
        print("✓ SENSITIVE")
    print(f"   Resistance Score: {report['resistance_score']:.2f}%")
    print(f"   Resistance Type: {report['resistance_type']}")
    print(f"   Mechanism: {report['resistance_mechanism']}")
    
    print("\nMOA ANALYSIS:")
    print("-" * 60)
    print(f"Predicted MOA: {report['predicted_moa']}")
    print(f"Confidence: {report['predicted_confidence']*100:.2f}%")
    print(f"Expected MOA probability: {report['expected_moa_probability']*100:.2f}%")
    
    print("\nTop 3 MOA Predictions:")
    for i, (moa, prob) in enumerate(report['top3_predictions'], 1):
        print(f"  {i}. {moa}: {prob*100:.2f}%")
    
    print("\nINTERPRETATION:")
    print("-" * 60)
    print(report['explanation'])
    
    if report['recommendations']:
        print("\nRECOMMENDED ALTERNATIVE DRUGS:")
        print("-" * 60)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec['drug']} ({rec['moa']})")
            print(f"   Score: {rec['score']:.2f}")
            print(f"   Rationale: {rec['rationale']}")
    
    print("\n" + "=" * 60)
    
    # Save visual report
    visualize_resistance_report(report)
    
    # Save JSON report
    report_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else 
                              bool(v) if isinstance(v, np.bool_) else v)
                          for k, v in report.items()}
    
    with open('resistance_report.json', 'w') as f:
        json.dump(report_serializable, f, indent=2)
    print("JSON report saved to resistance_report.json")

if __name__ == '__main__':
    main()