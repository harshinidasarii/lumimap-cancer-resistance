"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Cancer Resistance Detection
Shows which parts of cells the AI looks at to make predictions

This makes your AI explainable - perfect for ISEF presentations!
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

class GradCAM:
    """
    Grad-CAM: Visual Explanations for Deep Networks
    
    Shows heatmap of which parts of image influenced the prediction
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Your trained CNN model
            target_layer: Which layer to visualize (usually last conv layer)
                         For ResNet50: 'layer4'
                         For EfficientNet: 'features'
        """
        self.model = model
        self.model.eval()
        
        # Hook into the target layer
        self.target_layer = self._get_target_layer(target_layer)
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _get_target_layer(self, layer_name):
        """Get the specific layer from model"""
        if hasattr(self.model, 'backbone'):
            # If using our MOAClassifier structure
            if layer_name == 'layer4':
                return self.model.backbone.layer4
            elif layer_name == 'features':
                return self.model.backbone.features
        else:
            # Direct model access
            return getattr(self.model, layer_name)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, image_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for an image
        
        Args:
            image_tensor: Input image (1, 3, H, W)
            target_class: Which class to explain (None = predicted class)
        
        Returns:
            heatmap: 2D array of importance scores (H, W)
        """
        # Forward pass
        output = self.model(image_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize(self, image_tensor, original_image, target_class=None, 
                  alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Create visualization overlay
        
        Args:
            image_tensor: Preprocessed tensor for model
            original_image: Original image for visualization (H, W, 3)
            target_class: Which class to explain
            alpha: Transparency of heatmap overlay (0-1)
            colormap: OpenCV colormap
        
        Returns:
            overlay: RGB image with heatmap overlay
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(image_tensor, target_class)
        
        # Resize heatmap to match original image size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Normalize original image to [0, 255]
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(
            original_image,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return overlay, heatmap_resized


class ResistanceExplainer:
    """
    Generate explanations for resistance detection decisions
    Shows what the AI is looking at
    """
    
    def __init__(self, model):
        self.model = model
        self.gradcam = GradCAM(model, target_layer='layer4')  # For ResNet50
    # Import the EXACT same list used in training
    from moa_classifier_train import Config

    moa_classes = Config.MOA_CLASSES
    
    def explain_prediction(self, image_tensor, original_image, 
                          drug_info=None, save_path=None):
        """
        Create comprehensive explanation figure
        
        Args:
            image_tensor: Preprocessed tensor (1, 3, H, W)
            original_image: Original image array (H, W, 3)
            drug_info: Dict with 'name' and 'expected_moa'
            save_path: Where to save figure
        
        Returns:
            fig: Matplotlib figure
        """
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)[0]
        
        predicted_idx = probs.argmax().item()
        predicted_moa = self.moa_classes[predicted_idx]
        confidence = probs[predicted_idx].item()
        
        # Generate Grad-CAM for predicted class
        overlay_predicted, heatmap_predicted = self.gradcam.visualize(
            image_tensor, 
            original_image,
            target_class=predicted_idx
        )
        
        # If drug info provided, also show expected MOA heatmap
        if drug_info and 'expected_moa' in drug_info:
            expected_moa = drug_info['expected_moa']
            if expected_moa in self.moa_classes:
                expected_idx = self.moa_classes.index(expected_moa)
                expected_prob = probs[expected_idx].item()
                
                overlay_expected, heatmap_expected = self.gradcam.visualize(
                    image_tensor,
                    original_image,
                    target_class=expected_idx
                )
            else:
                overlay_expected = None
                expected_prob = 0.0
        else:
            overlay_expected = None
            expected_prob = None
        
        # Create figure
        if overlay_expected is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Predicted MOA
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original Image', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(heatmap_predicted, cmap='jet')
            axes[0, 1].set_title(f'Predicted: {predicted_moa}\nConfidence: {confidence:.2%}', 
                               fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(overlay_predicted)
            axes[0, 2].set_title('Grad-CAM Overlay', fontsize=12)
            axes[0, 2].axis('off')
            
            # Row 2: Expected MOA
            axes[1, 0].imshow(original_image)
            axes[1, 0].set_title('Original Image', fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(heatmap_expected, cmap='jet')
            axes[1, 1].set_title(f'Expected: {expected_moa}\nProbability: {expected_prob:.2%}',
                               fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(overlay_expected)
            axes[1, 2].set_title('Grad-CAM Overlay', fontsize=12)
            axes[1, 2].axis('off')
            
            # Add overall title
            resistance_score = 1.0 - expected_prob
            if resistance_score > 0.5:
                status = "⚠️ RESISTANT"
                color = 'red'
            else:
                status = "✓ SENSITIVE"
                color = 'green'
            
            fig.suptitle(
                f"{status} (Resistance Score: {resistance_score:.2%})\n"
                f"Drug: {drug_info.get('name', 'Unknown')}",
                fontsize=16,
                fontweight='bold',
                color=color
            )
        
        else:
            # Simple version - just predicted class
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image', fontsize=14)
            axes[0].axis('off')
            
            axes[1].imshow(heatmap_predicted, cmap='jet')
            axes[1].set_title(f'Activation Heatmap', fontsize=14)
            axes[1].axis('off')
            
            axes[2].imshow(overlay_predicted)
            axes[2].set_title(f'Predicted: {predicted_moa}\n{confidence:.2%} confidence',
                            fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            fig.suptitle('Grad-CAM Visualization', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Explanation saved to {save_path}")
        
        return fig
    
    def create_comparison_figure(self, examples, save_path='gradcam_comparison.png'):
        """
        Create side-by-side comparison of multiple examples
        Perfect for ISEF poster!
        
        Args:
            examples: List of dicts with 'image_tensor', 'original_image', 
                     'drug_info', 'title'
        """
        n_examples = len(examples)
        fig, axes = plt.subplots(2, n_examples, figsize=(5*n_examples, 10))
        
        if n_examples == 1:
            axes = axes.reshape(2, 1)
        
        for i, example in enumerate(examples):
            # Generate Grad-CAM
            image_tensor = example['image_tensor']
            original_image = example['original_image']
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probs = F.softmax(output, dim=1)[0]
            
            predicted_idx = probs.argmax().item()
            predicted_moa = self.moa_classes[predicted_idx]
            
            overlay, heatmap = self.gradcam.visualize(
                image_tensor,
                original_image,
                target_class=predicted_idx
            )
            
            # Plot
            axes[0, i].imshow(original_image)
            axes[0, i].set_title(example.get('title', f'Example {i+1}'), 
                               fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'{predicted_moa}\n{probs[predicted_idx].item():.1%}',
                               fontsize=11)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison figure saved to {save_path}")
        
        return fig


# ============================================================================
# EASY-TO-USE DEMO SCRIPT
# ============================================================================

def demo_gradcam():
    """
    Simple demo showing how to use Grad-CAM
    Run this after training your model!
    """
    from moa_classifier_train import MOAClassifier, get_val_transforms
    import torch
    
    print("Loading trained model...")

    # Load model
    from moa_classifier_train import Config

    model = MOAClassifier(
        backbone='resnet50', 
        num_classes=Config.NUM_CLASSES,  # Uses same number as training
        pretrained=False
    )
    checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded")
    
    # Create explainer
    explainer = ResistanceExplainer(model)
    
    # Load test image
    image_path = 'data/Week1/Week1_22161/Week1_150607_B02_s3_w1FE9E7681-E7DA-4BE8-B72E-66489E8726BE.tif'
    print(f"Loading image: {image_path}")
    
    # Load and preprocess
    original_image = np.array(Image.open(image_path))
    
    # Normalize for display
    if original_image.ndim == 2:
        original_image = np.stack([original_image]*3, axis=-1)
    
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    original_image = (original_image * 255).astype(np.uint8)
    
    # Preprocess for model
    transform = get_val_transforms()
    image_tensor = transform(image=original_image)['image'].unsqueeze(0)
    
    print("Generating Grad-CAM explanation...")
    
    # Generate explanation
    drug_info = {
        'name': 'cytochalasin B',
        'expected_moa': 'Actin disruptors'
    }
    
    fig = explainer.explain_prediction(
        image_tensor,
        original_image,
        drug_info=drug_info,
        save_path='gradcam_explanation.png'
    )
    
    plt.show()
    
    print("\n✓ Done! Check gradcam_explanation.png")


if __name__ == '__main__':
    demo_gradcam()
