"""
LUMIMAP Clinical Inference System
==================================

This script provides the complete inference pipeline for resistance detection:
1. Load microscopy images (DAPI, Tubulin, Actin channels)
2. Predict resistance type
3. Generate channel-wise GradCAM visualization
4. Provide therapy recommendations

Usage:
    python lumimap_inference.py --dapi path/to/dapi.tif --tubulin path/to/tubulin.tif \
                                --actin path/to/actin.tif --drug taxol --concentration 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import json

# Import models
import sys
sys.path.append('/home/claude')
from phase1_contrastive_moa_learning import MOAEncoder, Config as Phase1Config
from phase2_generate_resistance_labels import ResistanceType
from phase3_resistance_classifier_train import ResistanceClassifier

# ============================================================================
# GRADCAM
# ============================================================================

class GradCAM:
    """
    Grad-CAM for channel-wise visualization
    Shows which regions of each channel are important for prediction
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, class_idx=None):
        """
        Generate CAM heatmap
        Returns: [H, W] heatmap
        """
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]  # [C, H, W]
        activations = self.activations.cpu().data.numpy()[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

class ChannelWiseGradCAM:
    """Generate GradCAM for each channel separately"""
    def __init__(self, model):
        self.model = model
        
        # Create GradCAM for each channel encoder
        self.dapi_gradcam = GradCAM(
            model.encoder.dapi_encoder,
            model.encoder.dapi_encoder.backbone.layer4
        )
        self.tubulin_gradcam = GradCAM(
            model.encoder.tubulin_encoder,
            model.encoder.tubulin_encoder.backbone.layer4
        )
        self.actin_gradcam = GradCAM(
            model.encoder.actin_encoder,
            model.encoder.actin_encoder.backbone.layer4
        )
    
    def generate(self, dapi, tubulin, actin, concentration, target_class):
        """
        Generate GradCAM for all channels
        
        Args:
            dapi, tubulin, actin: [1, 1, H, W] tensors
            concentration: [1, 1] tensor
            target_class: class index to visualize
        
        Returns:
            dict with 'dapi', 'tubulin', 'actin' CAMs
        """
        # Forward pass
        self.model.eval()
        logits, attention = self.model(dapi, tubulin, actin, concentration)
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAMs
        dapi_cam = self.dapi_gradcam.generate_cam()
        tubulin_cam = self.tubulin_gradcam.generate_cam()
        actin_cam = self.actin_gradcam.generate_cam()
        
        return {
            'dapi': dapi_cam,
            'tubulin': tubulin_cam,
            'actin': actin_cam,
            'attention': attention.cpu().detach().numpy()[0]
        }

# ============================================================================
# THERAPY RECOMMENDATION ENGINE
# ============================================================================

class TherapyRecommender:
    """Recommends alternative therapies based on resistance type"""
    
    # MOA to drug mapping (from BBBC021)
    MOA_TO_DRUGS = {
        'Actin disruptors': ['cytochalasin B', 'cytochalasin D', 'latrunculin B'],
        'Aurora kinase inhibitors': ['AZ258', 'AZ841', 'AZ-A'],
        'Eg5 inhibitors': ['AZ-C', 'AZ138'],
        'Microtubule destabilizers': ['vincristine', 'demecolcine', 'colchicine', 'nocodazole'],
        'Microtubule stabilizers': ['taxol', 'docetaxel', 'epothilone B'],
        'DNA damage': ['mitomycin C', 'etoposide', 'cisplatin', 'chlorambucil'],
        'DNA replication': ['mitoxantrone', 'camptothecin', 'floxuridine', 'methotrexate'],
        'Protein synthesis': ['emetine', 'anisomycin', 'cyclohexamide'],
        'Protein degradation': ['ALLN', 'MG-132', 'proteasome inhibitor I', 'lactacystin'],
        'Kinase inhibitors': ['PD-169316', 'alsterpaullone', 'bryostatin'],
        'Cholesterol-lowering': ['mevinolin/lovastatin', 'simvastatin'],
        'Epithelial': ['PP-2', 'AZ-U', 'AZ-J']
    }
    
    def recommend(self, resistance_type, current_drug, current_moa, observed_moa=None):
        """
        Generate therapy recommendations
        
        Args:
            resistance_type: ResistanceType enum
            current_drug: Current drug being tested
            current_moa: Expected MOA of current drug
            observed_moa: If cross-resistance, which MOA was observed
        
        Returns:
            dict with recommendations
        """
        recommendations = {
            'resistance_type': resistance_type.name,
            'current_drug': current_drug,
            'current_moa': current_moa,
            'recommendation': None,
            'alternatives': [],
            'combination_therapy': None,
            'rationale': None
        }
        
        if resistance_type == ResistanceType.SENSITIVE:
            recommendations['recommendation'] = f"Continue {current_drug}"
            recommendations['rationale'] = "Cell showing expected drug response"
            
        elif resistance_type == ResistanceType.PRIMARY_RESISTANCE:
            recommendations['recommendation'] = "Switch to different drug class or increase dose"
            # Suggest drugs from same MOA class (different compounds)
            same_moa_drugs = [d for d in self.MOA_TO_DRUGS.get(current_moa, []) 
                            if d != current_drug]
            recommendations['alternatives'] = same_moa_drugs[:3]
            recommendations['rationale'] = "No response to current drug. Try alternative compounds or different MOA."
            
        elif resistance_type == ResistanceType.PARTIAL_RESISTANCE:
            recommendations['recommendation'] = "Increase dose or add combination therapy"
            recommendations['rationale'] = "Weak drug response. May benefit from higher concentration."
            
        elif resistance_type == ResistanceType.CROSS_RESISTANCE:
            if observed_moa and observed_moa != current_moa:
                recommendations['recommendation'] = f"Combination therapy targeting both {current_moa} and {observed_moa}"
                # Get drugs for observed MOA
                observed_moa_drugs = self.MOA_TO_DRUGS.get(observed_moa, [])
                recommendations['alternatives'] = observed_moa_drugs[:3]
                recommendations['combination_therapy'] = f"{current_drug} + {observed_moa_drugs[0] if observed_moa_drugs else 'drug targeting ' + observed_moa}"
                recommendations['rationale'] = f"Cell activated {observed_moa} bypass pathway. Dual targeting recommended."
            else:
                recommendations['recommendation'] = "Investigate resistance mechanism"
                recommendations['rationale'] = "Cell showing unexpected phenotype."
        
        elif resistance_type == ResistanceType.UNCERTAIN:
            recommendations['recommendation'] = "Repeat test or try alternative diagnostic"
            recommendations['rationale'] = "Ambiguous resistance signal."
        
        return recommendations

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(original_images, gradcams, attention, prediction, recommendations, output_path):
    """
    Create comprehensive visualization
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    channels = ['DAPI', 'Tubulin', 'Actin']
    colors = ['Blues', 'Greens', 'Reds']
    
    # Row 1: Original images
    for i, (channel, img) in enumerate(zip(channels, original_images)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{channel} Channel', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Row 2: GradCAM overlays
    for i, (channel, img, cam, color) in enumerate(zip(channels, original_images, 
                                                        [gradcams['dapi'], gradcams['tubulin'], gradcams['actin']],
                                                        colors)):
        ax = fig.add_subplot(gs[1, i])
        
        # Resize CAM to match image
        from skimage.transform import resize
        cam_resized = resize(cam, (img.shape[0], img.shape[1]))
        
        # Overlay
        ax.imshow(img, cmap='gray', alpha=0.7)
        im = ax.imshow(cam_resized, cmap=color, alpha=0.5)
        ax.set_title(f'{channel} Importance (Attn: {attention[i]:.2f})', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Prediction and recommendations
    ax_pred = fig.add_subplot(gs[2, :2])
    ax_pred.axis('off')
    
    # Prediction info
    pred_text = f"""
    RESISTANCE PREDICTION
    {'='*50}
    
    Resistance Type: {prediction['type']}
    Confidence: {prediction['confidence']:.1%}
    
    Current Drug: {prediction['drug']}
    Expected MOA: {prediction['moa']}
    """
    
    if prediction.get('observed_moa'):
        pred_text += f"\nObserved MOA: {prediction['observed_moa']} ⚠️"
    
    ax_pred.text(0.1, 0.5, pred_text, fontsize=11, family='monospace',
                verticalalignment='center')
    
    # Recommendations
    ax_rec = fig.add_subplot(gs[2, 2:])
    ax_rec.axis('off')
    
    rec_text = f"""
    THERAPY RECOMMENDATIONS
    {'='*50}
    
    Recommendation: {recommendations['recommendation']}
    
    Rationale: {recommendations['rationale']}
    """
    
    if recommendations['alternatives']:
        rec_text += f"\n\nAlternative Drugs:\n"
        for drug in recommendations['alternatives']:
            rec_text += f"  • {drug}\n"
    
    if recommendations['combination_therapy']:
        rec_text += f"\nCombination: {recommendations['combination_therapy']}"
    
    ax_rec.text(0.1, 0.5, rec_text, fontsize=10, family='monospace',
               verticalalignment='center')
    
    plt.suptitle('LUMIMAP: Cancer Drug Resistance Detection', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")

# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class LUMIMAPInference:
    """Complete inference pipeline"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        print("Loading LUMIMAP model...")
        self.model = ResistanceClassifier(num_classes=5).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        
        # Initialize GradCAM
        self.gradcam = ChannelWiseGradCAM(self.model)
        
        # Initialize therapy recommender
        self.recommender = TherapyRecommender()
    
    def preprocess_image(self, image_path):
        """Load and preprocess single channel image"""
        img = np.array(Image.open(image_path))
        
        # Normalize
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = 255 * (img - img_min) / (img_max - img_min)
        img = img.astype(np.uint8)
        
        # Resize
        from skimage.transform import resize
        img_resized = resize(img, Phase1Config.IMG_SIZE, preserve_range=True).astype(np.uint8)
        
        # Convert to tensor
        img_normalized = (img_resized / 255.0 - 0.5) / 0.5
        img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)
        
        return img_tensor, img_resized
    
    def predict(self, dapi_path, tubulin_path, actin_path, drug, concentration, moa=None):
        """
        Complete inference pipeline
        
        Args:
            dapi_path, tubulin_path, actin_path: Paths to image files
            drug: Drug name
            concentration: Drug concentration
            moa: Expected MOA (optional)
        
        Returns:
            dict with prediction, gradcam, recommendations
        """
        print(f"\nAnalyzing cell images for {drug} at {concentration}μM...")
        
        # Load images
        dapi, dapi_img = self.preprocess_image(dapi_path)
        tubulin, tubulin_img = self.preprocess_image(tubulin_path)
        actin, actin_img = self.preprocess_image(actin_path)
        
        dapi = dapi.to(self.device)
        tubulin = tubulin.to(self.device)
        actin = actin.to(self.device)
        conc = torch.FloatTensor([[concentration]]).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, attention = self.model(dapi, tubulin, actin, conc)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        resistance_type = ResistanceType(pred_class)
        
        print(f"✓ Prediction: {resistance_type.name} (confidence: {confidence:.1%})")
        
        # Generate GradCAM
        print("Generating GradCAM visualization...")
        gradcams = self.gradcam.generate(dapi, tubulin, actin, conc, pred_class)
        
        # Get recommendations
        print("Generating therapy recommendations...")
        recommendations = self.recommender.recommend(
            resistance_type, drug, moa if moa else "Unknown"
        )
        
        # Package results
        prediction = {
            'type': resistance_type.name,
            'confidence': confidence,
            'drug': drug,
            'concentration': concentration,
            'moa': moa,
            'probabilities': {ResistanceType(i).name: probs[0, i].item() 
                            for i in range(len(ResistanceType))}
        }
        
        return {
            'prediction': prediction,
            'gradcams': gradcams,
            'recommendations': recommendations,
            'original_images': [dapi_img, tubulin_img, actin_img]
        }

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LUMIMAP Resistance Detection')
    parser.add_argument('--dapi', required=True, help='Path to DAPI channel image')
    parser.add_argument('--tubulin', required=True, help='Path to Tubulin channel image')
    parser.add_argument('--actin', required=True, help='Path to Actin channel image')
    parser.add_argument('--drug', required=True, help='Drug name')
    parser.add_argument('--concentration', type=float, required=True, help='Drug concentration (μM)')
    parser.add_argument('--moa', default=None, help='Expected MOA (optional)')
    parser.add_argument('--model', default='./outputs/phase3/phase3_best_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--output', default='./lumimap_result.png', 
                       help='Output visualization path')
    
    args = parser.parse_args()
    
    # Initialize inference system
    lumimap = LUMIMAPInference(args.model)
    
    # Run prediction
    results = lumimap.predict(
        args.dapi, args.tubulin, args.actin,
        args.drug, args.concentration, args.moa
    )
    
    # Visualize
    print("Creating visualization...")
    visualize_results(
        results['original_images'],
        results['gradcams'],
        results['gradcams']['attention'],
        results['prediction'],
        results['recommendations'],
        args.output
    )
    
    # Print summary
    print("\n" + "="*70)
    print("LUMIMAP ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResistance Type: {results['prediction']['type']}")
    print(f"Confidence: {results['prediction']['confidence']:.1%}")
    print(f"\nRecommendation: {results['recommendations']['recommendation']}")
    print(f"Rationale: {results['recommendations']['rationale']}")
    
    if results['recommendations']['alternatives']:
        print(f"\nAlternative drugs: {', '.join(results['recommendations']['alternatives'])}")
    
    print(f"\n✓ Full report saved to {args.output}")
    print("="*70)

if __name__ == '__main__':
    main()
