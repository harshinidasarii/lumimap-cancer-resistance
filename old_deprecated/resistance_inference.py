"""
LUMIMAP Resistance Detection and Therapy Recommendation
Inference script that:
1. Predicts resistance type from microscopy images
2. Generates channel-wise GradCAM explanations
3. Provides therapeutic recommendations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import argparse
import json

from resistance_type_classifier_train import ChannelAwareResistanceClassifier, Config
from therapy_recommendation_database import ResistanceType, TherapyDatabase

class GradCAM:
    """
    Channel-aware Grad-CAM for explaining resistance predictions
    Generates separate heatmaps for DAPI, Tubulin, and Actin channels
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Hook to the last convolutional layer
        target_layer = self.model.backbone[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, image, concentration, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input tensor (1, 3, H, W)
            concentration: Drug concentration tensor (1, 1)
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: GradCAM heatmap (H, W)
            predicted_class: Predicted class index
        """
        # Forward pass
        output = self.model(image, concentration)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Weight channels by gradient average
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, target_class
    
    def generate_channel_specific_cams(self, image, concentration):
        """
        Generate separate GradCAM for each channel (DAPI, Tubulin, Actin)
        
        Returns:
            Dict with 'dapi', 'tubulin', 'actin' heatmaps
        """
        cams = {}
        
        # Process each channel separately
        for idx, channel_name in enumerate(['dapi', 'tubulin', 'actin']):
            # Create single-channel image
            single_channel = torch.zeros_like(image)
            single_channel[:, idx, :, :] = image[:, idx, :, :]
            
            # Replicate to 3 channels for model
            single_channel = single_channel[:, idx:idx+1, :, :].repeat(1, 3, 1, 1)
            
            # Generate CAM
            cam, _ = self.generate_cam(single_channel, concentration)
            cams[channel_name] = cam
        
        # Also generate combined CAM
        cam_combined, predicted_class = self.generate_cam(image, concentration)
        cams['combined'] = cam_combined
        cams['predicted_class'] = predicted_class
        
        return cams

class ResistancePredictor:
    """Main class for resistance prediction and therapy recommendation"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = ChannelAwareResistanceClassifier(
            backbone=Config.BACKBONE,
            num_classes=Config.NUM_RESISTANCE_TYPES,
            use_attention=Config.USE_CHANNEL_ATTENTION,
            use_concentration=Config.USE_CONCENTRATION_ENCODING
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize GradCAM
        self.gradcam = GradCAM(self.model)
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def load_image(self, dapi_path, tubulin_path, actin_path):
        """Load and preprocess 3-channel image"""
        
        def load_and_normalize(path):
            img = np.array(Image.open(path))
            img = img.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = 255 * (img - img_min) / (img_max - img_min)
            return img.astype(np.uint8)
        
        dapi = load_and_normalize(dapi_path)
        tubulin = load_and_normalize(tubulin_path)
        actin = load_and_normalize(actin_path)
        
        # Stack channels
        image = np.stack([dapi, tubulin, actin], axis=-1)
        
        # Resize
        image = cv2.resize(image, Config.IMG_SIZE)
        
        # Normalize (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.float()
    
    def predict(self, dapi_path, tubulin_path, actin_path, 
                compound, concentration):
        """
        Predict resistance type and generate recommendations
        
        Args:
            dapi_path: Path to DAPI channel image
            tubulin_path: Path to Tubulin channel image
            actin_path: Path to Actin channel image
            compound: Drug name
            concentration: Drug concentration
        
        Returns:
            Dictionary with prediction results
        """
        # Load image
        image = self.load_image(dapi_path, tubulin_path, actin_path)
        image = image.to(self.device)
        
        conc_tensor = torch.tensor([[concentration]], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image, conc_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[predicted_class].item()
        
        # Get resistance type
        resistance_type = ResistanceType(predicted_class)
        
        # Generate GradCAM
        cams = self.gradcam.generate_channel_specific_cams(image, conc_tensor)
        
        # Get therapy recommendation
        recommendation = TherapyDatabase.get_recommendation(
            resistance_type, compound, self._get_moa(compound)
        )
        
        # Generate report
        report = TherapyDatabase.generate_report(
            resistance_type, compound, 
            self._get_moa(compound), confidence
        )
        
        return {
            'resistance_type': resistance_type.name,
            'confidence': confidence,
            'probabilities': {ResistanceType(i).name: probabilities[i].item() 
                            for i in range(len(probabilities))},
            'gradcam': cams,
            'recommendation': recommendation,
            'report': report,
            'compound': compound,
            'concentration': concentration
        }
    
    def _get_moa(self, compound):
        """Get MOA for a compound (simplified - should query database)"""
        moa_map = {
            'taxol': 'Microtubule stabilizers',
            'docetaxel': 'Microtubule stabilizers',
            'cytochalasin B': 'Actin disruptors',
            'nocodazole': 'Microtubule destabilizers',
            'AZ-A': 'Aurora kinase inhibitors',
        }
        return moa_map.get(compound, 'Unknown')
    
    def visualize_results(self, results, save_path=None):
        """
        Create comprehensive visualization of results
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Load original images for display
        cams = results['gradcam']
        
        # Row 1: Channel-specific GradCAMs
        channel_names = ['DAPI (Nucleus)', 'Tubulin', 'Actin', 'Combined']
        channel_keys = ['dapi', 'tubulin', 'actin', 'combined']
        
        for i, (name, key) in enumerate(zip(channel_names, channel_keys)):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(cams[key], cmap='jet')
            ax.set_title(f'{name}\nGradCAM', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 2: Resistance type probabilities
        ax = fig.add_subplot(gs[1, :2])
        probs = results['probabilities']
        names = list(probs.keys())
        values = list(probs.values())
        
        colors = ['green' if name == results['resistance_type'] else 'gray' 
                 for name in names]
        
        bars = ax.barh(names, values, color=colors)
        ax.set_xlabel('Probability', fontweight='bold')
        ax.set_title('Resistance Type Classification', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
        
        # Row 2-3: Therapy recommendations
        ax = fig.add_subplot(gs[1:, 2:])
        ax.axis('off')
        
        rec = results['recommendation']
        
        text = f"""
RESISTANCE ANALYSIS
{'='*50}

Detected Type: {results['resistance_type']}
Confidence: {results['confidence']:.1%}

Drug: {results['compound']}
Concentration: {results['concentration']} µM

{'='*50}
THERAPEUTIC STRATEGY
{'='*50}

{rec.get('strategy', 'N/A')}

FIRST-LINE RECOMMENDATIONS:
"""
        for drug in rec.get('first_line', [])[:3]:
            text += f"• {drug}\n"
        
        text += f"""
COMBINATION THERAPY:
{rec.get('combination', 'N/A')}

RATIONALE:
{rec.get('rationale', 'N/A')}

KEY BIOMARKERS:
"""
        for marker in rec.get('biomarkers', [])[:3]:
            text += f"• {marker}\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'LUMIMAP Resistance Analysis: {results["compound"]}',
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, results, output_path):
        """Save results to JSON and text report"""
        
        # Save JSON (exclude GradCAM for size)
        json_results = {k: v for k, v in results.items() if k != 'gradcam'}
        
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save text report
        report_path = Path(output_path).with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write(results['report'])
        
        print(f"Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  Report: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='LUMIMAP: Predict resistance type and recommend therapy'
    )
    parser.add_argument('--model', required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--dapi', required=True, 
                       help='Path to DAPI channel image')
    parser.add_argument('--tubulin', required=True, 
                       help='Path to Tubulin channel image')
    parser.add_argument('--actin', required=True, 
                       help='Path to Actin channel image')
    parser.add_argument('--compound', required=True, 
                       help='Drug/compound name')
    parser.add_argument('--concentration', type=float, required=True, 
                       help='Drug concentration')
    parser.add_argument('--output', default='./resistance_analysis', 
                       help='Output path prefix')
    parser.add_argument('--device', default='cuda', 
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ResistancePredictor(args.model, args.device)
    
    # Run prediction
    print("\nAnalyzing resistance...")
    results = predictor.predict(
        args.dapi, args.tubulin, args.actin,
        args.compound, args.concentration
    )
    
    # Print summary
    print("\n" + "="*70)
    print(f"RESISTANCE TYPE: {results['resistance_type']}")
    print(f"CONFIDENCE: {results['confidence']:.1%}")
    print("="*70)
    
    # Visualize
    viz_path = args.output + '_visualization.png'
    predictor.visualize_results(results, viz_path)
    
    # Save results
    predictor.save_results(results, args.output)
    
    # Print report
    print(results['report'])

if __name__ == '__main__':
    main()
