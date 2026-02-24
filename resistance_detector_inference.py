"""
Cancer Resistance Detection - Inference Script
Phase 2: Detecting resistance using MOA prediction mismatch

This script uses the trained MOA classifier to detect drug resistance by comparing
predicted MOA vs expected MOA for a given drug treatment.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import json

# Import from training script
from moa_classifier_train import MOAClassifier, Config, get_val_transforms

# ============================================================================
# RESISTANCE DETECTOR
# ============================================================================

class ResistanceDetector:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize resistance detector with trained MOA classifier
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        self.transform = get_val_transforms()
        
        # MOA information
        self.moa_classes = Config.MOA_CLASSES
        self.moa_to_idx = {moa: idx for idx, moa in enumerate(self.moa_classes)}
        
        # Drug-MOA mapping (from BBBC021 + literature)
        self.drug_moa_mapping = self._load_drug_database()
    
    def _load_model(self, model_path):
        """Load trained model"""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = MOAClassifier(
            backbone=Config.BACKBONE,
            num_classes=Config.NUM_CLASSES,
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Model loaded. Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
        return model
    
    def _load_drug_database(self):
        """
        Load drug-MOA mapping
        In production, this would come from a comprehensive database
        """
        # Example mapping based on BBBC021 compounds
        return {
            'cytochalasin B': 'Actin disruptors',
            'latrunculin B': 'Actin disruptors',
            'barasertib': 'Aurora kinase inhibitors',
            'paclitaxel': 'Microtubule stabilizers',
            'taxol': 'Microtubule stabilizers',
            'nocodazole': 'Microtubule destabilizers',
            'colchicine': 'Microtubule destabilizers',
            'monastrol': 'Eg5 inhibitors',
            'doxorubicin': 'DNA damage',
            'etoposide': 'DNA damage',
            'cisplatin': 'DNA damage',
            'camptothecin': 'DNA damage',
            'hydroxyurea': 'DNA replication',
            'atorvastatin': 'Cholesterol-lowering',
            'bortezomib': 'Protein degradation',
            'MG132': 'Protein degradation',
            'emetine': 'Protein synthesis',
            'cycloheximide': 'Protein synthesis',
            # Add more as needed
        }
    
    def load_image(self, image_path):
        """
        Load and preprocess microscopy image
        
        Args:
            image_path: Path to image file (or tuple of 3 paths for DAPI, Tubulin, Actin)
        
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image_path, (tuple, list)):
            # Load 3 separate channel files
            channels = []
            for path in image_path:
                img = np.array(Image.open(path))
                channels.append(img)
            image = np.stack(channels, axis=-1)
        else:
            # Single 3-channel image
            image = np.array(Image.open(image_path))
            if image.ndim == 2:
                # Grayscale - replicate to 3 channels
                image = np.stack([image] * 3, axis=-1)
        
        # Normalize
        image = self._normalize_image(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def _normalize_image(self, image):
        """Normalize image to 0-255 range"""
        image = image.astype(np.float32)
        for c in range(3):
            channel = image[:, :, c]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                image[:, :, c] = 255 * (channel - channel_min) / (channel_max - channel_min)
        return image.astype(np.uint8)
    
    def predict_moa(self, image_tensor):
        """
        Predict MOA from image
        
        Returns:
            dict with predicted_moa, confidence, and full probability distribution
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]  # Get probabilities
        
        # Get top prediction
        top_prob, top_idx = probs.max(0)
        predicted_moa = self.moa_classes[top_idx.item()]
        confidence = top_prob.item()
        
        # Get top-3 predictions
        top3_probs, top3_indices = probs.topk(3)
        top3 = [
            {'moa': self.moa_classes[idx.item()], 'probability': prob.item()}
            for prob, idx in zip(top3_probs, top3_indices)
        ]
        
        return {
            'predicted_moa': predicted_moa,
            'confidence': confidence,
            'top3': top3,
            'all_probabilities': probs.cpu().numpy()
        }
    
    def detect_resistance(self, image_path, drug_info):
        """
        Main resistance detection function
        
        Args:
            image_path: Path to microscopy image(s)
            drug_info: Dict with keys:
                - 'name': Drug name
                - 'expected_moa': Expected MOA (optional, will look up if not provided)
                - 'concentration': Drug concentration in nM (optional)
        
        Returns:
            Comprehensive resistance analysis
        """
        # Load image
        image_tensor = self.load_image(image_path)
        
        # Get MOA prediction
        prediction = self.predict_moa(image_tensor)
        predicted_moa = prediction['predicted_moa']
        
        # Determine expected MOA
        if 'expected_moa' in drug_info:
            expected_moa = drug_info['expected_moa']
        else:
            drug_name = drug_info['name'].lower()
            expected_moa = self.drug_moa_mapping.get(drug_name, 'Unknown')
        
        # Calculate resistance score
        if expected_moa in self.moa_to_idx:
            expected_moa_idx = self.moa_to_idx[expected_moa]
            expected_moa_prob = prediction['all_probabilities'][expected_moa_idx]
        else:
            expected_moa_prob = 0.0
        
        # Resistance score = 1 - probability of expected MOA
        resistance_score = 1.0 - expected_moa_prob
        
        # Classify resistance type
        resistance_type, explanation = self._classify_resistance_type(
            predicted_moa, expected_moa, resistance_score
        )
        
        # Determine if resistant (threshold = 0.5)
        is_resistant = resistance_score > 0.5
        
        result = {
            'is_resistant': is_resistant,
            'resistance_score': float(resistance_score),
            'resistance_type': resistance_type,
            'predicted_moa': predicted_moa,
            'expected_moa': expected_moa,
            'prediction_confidence': prediction['confidence'],
            'expected_moa_probability': float(expected_moa_prob),
            'top3_predictions': prediction['top3'],
            'explanation': explanation,
            'drug_info': drug_info
        }
        
        return result
    
    def _classify_resistance_type(self, predicted_moa, expected_moa, resistance_score):
        """Classify the type of resistance"""
        
        if resistance_score < 0.3:
            return 'sensitive', 'Cells show normal drug response'
        
        elif resistance_score < 0.5:
            return 'partial_resistance', 'Cells show weakened but detectable drug response'
        
        elif predicted_moa == expected_moa:
            # High resistance score but predicted correct MOA?
            # This means low confidence in prediction
            return 'uncertain', 'Unclear morphology - may need higher dose or longer treatment'
        
        else:
            # Different MOA predicted
            if 'DMSO' in predicted_moa or resistance_score > 0.8:
                return 'complete_resistance', \
                       'Cells appear untreated - likely efflux pump or target mutation'
            else:
                return 'cross_resistance', \
                       f'Cells activated alternative {predicted_moa} pathway'
    
    def generate_heatmap(self, image_path, drug_info, save_path=None):
        """
        Generate Grad-CAM heatmap showing which regions influence prediction
        
        This helps explain WHY the AI thinks cells are resistant
        """
        from gradcam_explainer import ResistanceExplainer
        from PIL import Image
        import numpy as np
        
        # Load image
        image_tensor = self.load_image(image_path)
        
        # Load original for visualization
        if isinstance(image_path, (tuple, list)):
            original = np.array(Image.open(image_path[0]))
        else:
            original = np.array(Image.open(image_path))
        
        if original.ndim == 2:
            original = np.stack([original]*3, axis=-1)
        
        # Normalize for display
        original = self._normalize_image(original)
        
        # Create explainer
        explainer = ResistanceExplainer(self.moa_predictor)
        
        # Generate explanation
        fig = explainer.explain_prediction(
            image_tensor,
            original,
            drug_info=drug_info,
            save_path=save_path or 'gradcam_heatmap.png'
        )
        
        return fig

# ============================================================================
# DRUG RECOMMENDER
# ============================================================================

class DrugRecommender:
    def __init__(self):
        # MOA similarity matrix (can be learned from data or predefined)
        self.moa_similarity = self._build_moa_similarity_matrix()
        
        # Drug database with alternative options
        self.drug_database = self._load_drug_database()
    
    def _build_moa_similarity_matrix(self):
        """
        Build similarity matrix between MOAs
        High similarity = drugs likely to show cross-resistance
        """
        moas = Config.MOA_CLASSES
        n = len(moas)
        similarity = np.eye(n)  # Start with identity
        
        # Define similarities (0-1 scale)
        # Example: microtubule drugs are similar to each other
        similar_pairs = [
            ('Microtubule stabilizers', 'Microtubule destabilizers', 0.6),
            ('DNA damage', 'DNA replication', 0.5),
            ('Protein degradation', 'Protein synthesis', 0.4),
            # Add more based on biological knowledge
        ]
        
        moa_to_idx = {moa: i for i, moa in enumerate(moas)}
        
        for moa1, moa2, sim in similar_pairs:
            if moa1 in moa_to_idx and moa2 in moa_to_idx:
                i, j = moa_to_idx[moa1], moa_to_idx[moa2]
                similarity[i, j] = sim
                similarity[j, i] = sim
        
        return similarity
    
    def _load_drug_database(self):
        """Load comprehensive drug database"""
        # In production, this would be a real database
        drugs = [
            {'name': 'cisplatin', 'moa': 'DNA damage', 'mdr1_substrate': False},
            {'name': 'paclitaxel', 'moa': 'Microtubule stabilizers', 'mdr1_substrate': True},
            {'name': 'doxorubicin', 'moa': 'DNA damage', 'mdr1_substrate': True},
            {'name': 'gemcitabine', 'moa': 'DNA replication', 'mdr1_substrate': False},
            {'name': 'bortezomib', 'moa': 'Protein degradation', 'mdr1_substrate': False},
            {'name': 'barasertib', 'moa': 'Aurora kinase inhibitors', 'mdr1_substrate': False},
            {'name': 'monastrol', 'moa': 'Eg5 inhibitors', 'mdr1_substrate': False},
            # Add more drugs...
        ]
        return drugs
    
    def recommend_alternatives(self, resistance_result):
        """
        Recommend alternative drugs based on resistance analysis
        
        Args:
            resistance_result: Output from ResistanceDetector.detect_resistance()
        
        Returns:
            List of recommended drugs with rationale
        """
        if not resistance_result['is_resistant']:
            return []  # No alternatives needed
        
        failed_moa = resistance_result['expected_moa']
        resistance_type = resistance_result['resistance_type']
        
        recommendations = []
        
        if resistance_type == 'complete_resistance':
            # Try drugs with completely different MOAs
            for drug in self.drug_database:
                if drug['moa'] != failed_moa:
                    # Calculate MOA distance
                    moa_idx_failed = Config.MOA_CLASSES.index(failed_moa)
                    moa_idx_candidate = Config.MOA_CLASSES.index(drug['moa'])
                    similarity = self.moa_similarity[moa_idx_failed, moa_idx_candidate]
                    score = 1.0 - similarity  # Less similar = better
                    
                    recommendations.append({
                        'drug': drug['name'],
                        'moa': drug['moa'],
                        'score': score,
                        'rationale': f"Different mechanism from {failed_moa}",
                        'evidence': 'High' if not drug['mdr1_substrate'] else 'Moderate'
                    })
        
        elif resistance_type == 'cross_resistance':
            # Try drugs from unrelated pathways
            predicted_moa = resistance_result['predicted_moa']
            for drug in self.drug_database:
                if drug['moa'] not in [failed_moa, predicted_moa]:
                    recommendations.append({
                        'drug': drug['name'],
                        'moa': drug['moa'],
                        'score': 0.7,
                        'rationale': f"Avoids both {failed_moa} and {predicted_moa} pathways"
                    })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:5]  # Top 5

# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ResistanceReport:
    @staticmethod
    def generate_text_report(resistance_result, recommendations):
        """Generate human-readable text report"""
        report = []
        report.append("="*60)
        report.append("CANCER RESISTANCE DETECTION REPORT")
        report.append("="*60)
        report.append("")
        
        # Drug information
        drug_info = resistance_result['drug_info']
        report.append(f"Drug: {drug_info['name']}")
        if 'concentration' in drug_info:
            report.append(f"Concentration: {drug_info['concentration']} nM")
        report.append(f"Expected MOA: {resistance_result['expected_moa']}")
        report.append("")
        
        # Resistance status
        report.append("RESISTANCE STATUS:")
        report.append("-" * 60)
        if resistance_result['is_resistant']:
            report.append(f"⚠️  RESISTANCE DETECTED")
            report.append(f"   Resistance Score: {resistance_result['resistance_score']:.2%}")
            report.append(f"   Resistance Type: {resistance_result['resistance_type']}")
        else:
            report.append(f"✓  SENSITIVE")
            report.append(f"   Cells respond normally to drug")
        report.append("")
        
        # MOA analysis
        report.append("MOA ANALYSIS:")
        report.append("-" * 60)
        report.append(f"Predicted MOA: {resistance_result['predicted_moa']}")
        report.append(f"Confidence: {resistance_result['prediction_confidence']:.2%}")
        report.append(f"Expected MOA probability: {resistance_result['expected_moa_probability']:.2%}")
        report.append("")
        
        # Top 3 predictions
        report.append("Top 3 MOA Predictions:")
        for i, pred in enumerate(resistance_result['top3_predictions'], 1):
            report.append(f"  {i}. {pred['moa']}: {pred['probability']:.2%}")
        report.append("")
        
        # Explanation
        report.append("INTERPRETATION:")
        report.append("-" * 60)
        report.append(resistance_result['explanation'])
        report.append("")
        
        # Recommendations
        if recommendations:
            report.append("RECOMMENDED ALTERNATIVE DRUGS:")
            report.append("-" * 60)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec['drug']} ({rec['moa']})")
                report.append(f"   Score: {rec['score']:.2f}")
                report.append(f"   Rationale: {rec['rationale']}")
                report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    @staticmethod
    def generate_visual_report(resistance_result, recommendations, output_path='report.png'):
        """Generate visual report with plots"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Resistance gauge
        ax1 = fig.add_subplot(gs[0, 0])
        score = resistance_result['resistance_score']
        colors = ['green', 'yellow', 'orange', 'red']
        color = colors[int(score * 3)]
        ax1.barh(0, score, color=color, height=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_title(f"Resistance Score: {score:.2%}", fontsize=14, fontweight='bold')
        ax1.set_xlabel('Score')
        ax1.axvline(0.5, color='black', linestyle='--', alpha=0.5)
        ax1.text(0.5, -0.3, 'Threshold', ha='center', fontsize=10)
        
        # 2. MOA probability distribution
        ax2 = fig.add_subplot(gs[0, 1])
        moa_probs = resistance_result['top3_predictions']
        moas = [p['moa'] for p in moa_probs]
        probs = [p['probability'] for p in moa_probs]
        bars = ax2.barh(moas, probs)
        # Highlight expected MOA
        expected_idx = next((i for i, m in enumerate(moas) 
                           if m == resistance_result['expected_moa']), None)
        if expected_idx is not None:
            bars[expected_idx].set_color('red')
        ax2.set_xlabel('Probability')
        ax2.set_title('Top 3 MOA Predictions', fontsize=14, fontweight='bold')
        
        # 3. Resistance type
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        resistance_info = f"""
        Drug: {resistance_result['drug_info']['name']}
        Expected MOA: {resistance_result['expected_moa']}
        Predicted MOA: {resistance_result['predicted_moa']}
        
        Resistance Type: {resistance_result['resistance_type'].replace('_', ' ').title()}
        
        Explanation:
        {resistance_result['explanation']}
        """
        ax3.text(0.1, 0.5, resistance_info, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. Recommendations
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        if recommendations:
            rec_text = "RECOMMENDED ALTERNATIVES:\n\n"
            for i, rec in enumerate(recommendations[:3], 1):
                rec_text += f"{i}. {rec['drug']} ({rec['moa']})\n"
                rec_text += f"   Rationale: {rec['rationale']}\n\n"
        else:
            rec_text = "Cells are sensitive - no alternative drugs needed"
        
        ax4.text(0.1, 0.5, rec_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visual report saved to {output_path}")
        return output_path

# ============================================================================
# MAIN INFERENCE SCRIPT
# ============================================================================

def main():
    # Initialize detector
    detector = ResistanceDetector(
        model_path='./outputs/best_model.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize recommender
    recommender = DrugRecommender()
    
    # Example usage
    print("\n" + "="*60)
    print("Cancer Resistance Detection System")
    print("="*60 + "\n")
    
    # Test image (replace with your actual image path)
    image_path = 'data/Week1/Week1_22161/Week1_150607_B02_s3_w1FE9E7681-E7DA-4BE8-B72E-66489E8726BE.tif'
    
    # Use an actual image from your data folder
    
    # Or for 3 separate channel files:
    # image_path = (
    #     'Week1_150607_B04_s2_w1_DAPI.tif',
    #     'Week1_150607_B04_s2_w2_Tubulin.tif',
    #     'Week1_150607_B04_s2_w4_Actin.tif'
    # )
    
    # Drug information
    drug_info = {
        'name': 'cytochalasin B',
        'expected_moa': 'Actin disruptors',
        'concentration': 10  # nM
    }
    
    # Detect resistance
    print("Analyzing image...")
    result = detector.detect_resistance(image_path, drug_info)
    
    # Get recommendations if resistant
    recommendations = recommender.recommend_alternatives(result)
    
    # Generate reports
    print("\n" + ResistanceReport.generate_text_report(result, recommendations))
    
    # Visual report
    ResistanceReport.generate_visual_report(result, recommendations, 'resistance_report.png')
    
    # # Save JSON
    # with open('resistance_result.json', 'w') as f:
    #     # Convert numpy types to Python types for JSON serialization
    #     result_json = {k: float(v) if isinstance(v, np.floating) else v 
    #                   for k, v in result.items()}
    #     json.dump({'result': result_json, 'recommendations': recommendations}, f, indent=2)
    # print("\nResults saved to resistance_result.json")

if __name__ == '__main__':
    main()
