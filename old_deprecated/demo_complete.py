"""
LUMIMAP Complete Demo - Uses Phase 2 Results
=============================================

Shows:
- Cell images (DAPI, Tubulin, Actin)
- Similarity scores to MOA/DMSO
- Resistance classification
- Channel attention
- Treatment recommendation

Usage: python demo_complete.py --idx 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    LABELS_CSV = './output/phase2_strategic/resistance_labels.csv'
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128

class ChannelEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone = mobilenet.features
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, output_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projector(features)
        return F.normalize(embeddings, dim=1)

class ConcentrationEncoder(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, concentration):
        log_conc = torch.log1p(concentration)
        return self.encoder(log_conc)

class ChannelAttention(nn.Module):
    def __init__(self, num_channels=3, embed_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_channels),
            nn.Softmax(dim=1)
        )
    
    def forward(self, channel_embeddings):
        batch_size = channel_embeddings.shape[0]
        flat = channel_embeddings.view(batch_size, -1)
        weights = self.attention(flat)
        weights = weights.unsqueeze(2)
        weighted = channel_embeddings * weights
        return weighted, weights.squeeze(2)

class MOAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dapi_encoder = ChannelEncoder(Config.EMBEDDING_DIM)
        self.tubulin_encoder = ChannelEncoder(Config.EMBEDDING_DIM)
        self.actin_encoder = ChannelEncoder(Config.EMBEDDING_DIM)
        self.concentration_encoder = ConcentrationEncoder(16)
        self.channel_attention = ChannelAttention(3, Config.EMBEDDING_DIM)
        
        self.fusion = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM * 3 + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, Config.EMBEDDING_DIM)
        )
    
    def forward(self, dapi, tubulin, actin, concentration):
        dapi_emb = self.dapi_encoder(dapi)
        tubulin_emb = self.tubulin_encoder(tubulin)
        actin_emb = self.actin_encoder(actin)
        
        channel_embs = torch.stack([dapi_emb, tubulin_emb, actin_emb], dim=1)
        weighted_channels, attention_weights = self.channel_attention(channel_embs)
        weighted_flat = weighted_channels.view(weighted_channels.shape[0], -1)
        
        conc_emb = self.concentration_encoder(concentration)
        combined = torch.cat([weighted_flat, conc_emb], dim=1)
        final_embedding = self.fusion(combined)
        
        return F.normalize(final_embedding, dim=1), attention_weights

def get_treatment_recommendation(resistance_type):
    """Get clinical recommendation based on resistance type"""
    recommendations = {
        'SENSITIVE': {
            'action': 'Continue Current Treatment',
            'details': 'Cells are responding well to the drug. Maintain current dosage.',
            'color': 'green'
        },
        'PARTIAL_RESISTANCE': {
            'action': 'Adjust Treatment Strategy',
            'details': 'Partial response detected. Consider: 1) Increase dosage, 2) Add adjuvant therapy, 3) Monitor closely',
            'color': 'orange'
        },
        'CROSS_RESISTANCE': {
            'action': 'Switch Drug Class',
            'details': 'Cells showing response to different drug mechanism. Recommend switching to alternative MOA.',
            'color': 'red'
        },
        'PRIMARY_RESISTANCE': {
            'action': 'Change Treatment Immediately',
            'details': 'No response to current drug. Switch to different drug class or combination therapy.',
            'color': 'darkred'
        },
        'UNCERTAIN': {
            'action': 'Additional Testing Required',
            'details': 'Unclear phenotype. Recommend: 1) Repeat assay, 2) Test alternative markers',
            'color': 'gray'
        }
    }
    return recommendations.get(resistance_type, recommendations['UNCERTAIN'])

def visualize_complete_analysis(images, attention, label_row, recommendation, save_path=None):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(20, 14))  # Larger figure
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)  # More spacing
    
    # Title
    fig.suptitle('LUMIMAP: AI-Powered Drug Resistance Detection', 
                fontsize=22, fontweight='bold', y=0.98)
    
    # Row 1: Individual channel images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(images['dapi'], cmap='Blues')
    ax1.set_title(f'DAPI (Nucleus)\nAttention: {attention[0]:.3f}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(images['tubulin'], cmap='Greens')
    ax2.set_title(f'Tubulin (Microtubules)\nAttention: {attention[1]:.3f}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(images['actin'], cmap='Reds')
    ax3.set_title(f'Actin (Cytoskeleton)\nAttention: {attention[2]:.3f}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Composite
    ax4 = fig.add_subplot(gs[0, 3])
    composite = np.stack([
        images['actin'] / images['actin'].max(),
        images['tubulin'] / images['tubulin'].max(),
        images['dapi'] / images['dapi'].max()
    ], axis=-1)
    ax4.imshow(composite)
    ax4.set_title('RGB Composite', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Similarity Analysis
    ax5 = fig.add_subplot(gs[1, :2])
    sims = ['DMSO\n(Baseline)', f"{label_row['moa']}\n(Expected)", 'Cross-MOA\n(Other Drug)']
    sim_vals = [label_row['dmso_similarity'], label_row['moa_similarity'], 
                label_row.get('cross_moa_similarity', 0.0)]
    colors = ['#808080', '#4472C4', '#ED7D31']
    
    bars = ax5.bar(sims, sim_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.axhline(y=0.80, color='green', linestyle='--', linewidth=2, label='High Threshold (0.80)', alpha=0.7)
    ax5.axhline(y=0.65, color='orange', linestyle='--', linewidth=2, label='Medium Threshold (0.65)', alpha=0.7)
    ax5.set_ylabel('Similarity Score', fontsize=13, fontweight='bold')
    ax5.set_title('Phenotype Similarity Analysis', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 1.0])
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Add value labels on bars
    for bar, val in zip(bars, sim_vals):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Row 2: Channel Attention
    ax6 = fig.add_subplot(gs[1, 2:])
    channels = ['DAPI\n(Nucleus)', 'Tubulin\n(Microtubules)', 'Actin\n(Cytoskeleton)']
    attn_colors = ['#5B9BD5', '#70AD47', '#C55A11']
    bars2 = ax6.bar(channels, attention, color=attn_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Attention Weight', fontsize=13, fontweight='bold')
    ax6.set_title('Channel Attention Mechanism', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 1.0])
    ax6.grid(axis='y', alpha=0.3, linestyle=':')
    
    for bar, val in zip(bars2, attention):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Row 3: Sample Information
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.axis('off')
    info_text = f"""
Sample Information:

Compound: {label_row['compound']}
Concentration: {label_row['concentration']:.2e} M
Mechanism of Action: {label_row['moa']}

Analysis Method:
• Phase 1: Contrastive MOA learning
• Phase 2: Similarity-based classification
• Strategic sampling: 60+ compounds
    """
    ax7.text(0.05, 0.95, info_text.strip(), 
            transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Row 3: Classification Result
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('off')
    
    resistance_type = label_row['resistance_type']
    rec = recommendation
    
    # Big classification label
    ax8.text(0.5, 0.85, resistance_type,
            ha='center', va='top', fontsize=24, fontweight='bold',
            color=rec['color'], transform=ax8.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=rec['color'], linewidth=3))
    
    # Recommendation
    ax8.text(0.5, 0.55, rec['action'],
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax8.transAxes)
    
    ax8.text(0.5, 0.35, rec['details'],
            ha='center', va='top', fontsize=10,
            transform=ax8.transAxes, wrap=True)
    
    # Row 4: Interpretation
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')
    
    interpretation = f"""
INTERPRETATION:

The cell shows {label_row['moa_similarity']:.1%} similarity to the expected {label_row['moa']} drug response.
DMSO (baseline) similarity is {label_row['dmso_similarity']:.1%}.

Channel attention analysis reveals the model focuses on:
• {'DAPI (nucleus)' if attention[0] == max(attention) else 'Tubulin (microtubules)' if attention[1] == max(attention) else 'Actin (cytoskeleton)'} 
  with {max(attention):.1%} attention weight

Clinical Significance: {rec['details']}
    """
    
    ax9.text(0.5, 0.5, interpretation.strip(),
            ha='center', va='center', fontsize=11,
            transform=ax9.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Visualization saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='LUMIMAP Complete Demo')
    parser.add_argument('--idx', type=int, required=True, help='Sample index')
    parser.add_argument('--save', action='store_true', help='Save visualization (default: True)')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save visualization')
    parser.set_defaults(save=True)
    args = parser.parse_args()
    
    # Create output directory
    import os
    output_dir = './output/demo_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("🔬 LUMIMAP: AI-Powered Drug Resistance Detection")
    print("="*70)
    
    # Load model
    print("\n📂 Loading Phase 1 model...")
    model = MOAEncoder()
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    print("   ✓ Model loaded")
    
    # Load data
    print("\n📂 Loading data...")
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(Config.IMAGE_CSV)
    
    # Get sample
    if args.idx not in labels_df['idx'].values:
        print(f"❌ Index {args.idx} not found!")
        # Show available indices
        print(f"\nAvailable indices: {labels_df['idx'].min()} to {labels_df['idx'].max()}")
        print(f"Try: python demo_complete.py --idx {labels_df['idx'].iloc[0]}")
        return
    
    label_row = labels_df[labels_df['idx'] == args.idx].iloc[0]
    image_row = image_df.iloc[args.idx]
    
    print(f"\n🔍 Analyzing sample {args.idx}...")
    print(f"   Compound: {label_row['compound']}")
    print(f"   MOA: {label_row['moa']}")
    print(f"   Concentration: {label_row['concentration']:.2e}")
    print(f"   True resistance: {label_row['resistance_type']}")
    
    # Load images
    dapi_path = Path(Config.DATA_DIR) / image_row['Image_PathName_DAPI'] / image_row['Image_FileName_DAPI']
    tubulin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Tubulin'] / image_row['Image_FileName_Tubulin']
    actin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Actin'] / image_row['Image_FileName_Actin']
    
    def load_normalize(path):
        img = np.array(Image.open(path)).astype(np.float32)
        c_min, c_max = img.min(), img.max()
        if c_max > c_min:
            img = 255 * (img - c_min) / (c_max - c_min)
        return img.astype(np.uint8)
    
    dapi = load_normalize(dapi_path)
    tubulin = load_normalize(tubulin_path)
    actin = load_normalize(actin_path)
    
    # Get attention weights
    transform = A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    dapi_t = transform(image=np.stack([dapi]*3, axis=-1))['image'][0:1].unsqueeze(0)
    tubulin_t = transform(image=np.stack([tubulin]*3, axis=-1))['image'][0:1].unsqueeze(0)
    actin_t = transform(image=np.stack([actin]*3, axis=-1))['image'][0:1].unsqueeze(0)
    conc_t = torch.tensor([[label_row['concentration']]], dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        dapi_t = dapi_t.to(Config.DEVICE)
        tubulin_t = tubulin_t.to(Config.DEVICE)
        actin_t = actin_t.to(Config.DEVICE)
        conc_t = conc_t.to(Config.DEVICE)
        
        _, attention = model(dapi_t, tubulin_t, actin_t, conc_t)
        attention = attention.cpu().numpy()[0]
    
    print(f"\n✅ Classification: {label_row['resistance_type']}")
    print(f"\n📊 Similarities:")
    print(f"   DMSO (baseline): {label_row['dmso_similarity']:.3f}")
    print(f"   MOA (expected):  {label_row['moa_similarity']:.3f}")
    print(f"\n🔍 Attention Weights:")
    print(f"   DAPI:    {attention[0]:.3f}")
    print(f"   Tubulin: {attention[1]:.3f}")
    print(f"   Actin:   {attention[2]:.3f}")
    
    # Get recommendation
    recommendation = get_treatment_recommendation(label_row['resistance_type'])
    print(f"\n💊 Recommendation: {recommendation['action']}")
    print(f"   {recommendation['details']}")
    
    # Prepare save path
    save_path = None
    if args.save:
        save_path = os.path.join(output_dir, f'sample_{args.idx}_analysis.png')
    
    # Visualize
    images = {'dapi': dapi, 'tubulin': tubulin, 'actin': actin}
    visualize_complete_analysis(images, attention, label_row, recommendation, save_path)

if __name__ == '__main__':
    main()