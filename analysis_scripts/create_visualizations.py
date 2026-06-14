"""
Batch Visualization Creator
============================

Creates publication-quality figures for science fair poster

Generates:
- 5 SENSITIVE examples
- 5 RESISTANT examples (PARTIAL/CROSS/PRIMARY combined)
- Saved to ./output/demo_visualizations/
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
import os

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    LABELS_CSV = './output/phase2_strategic/resistance_labels.csv'
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    OUTPUT_DIR = './output/demo_visualizations'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128

# [Model classes - same as demo_complete.py]
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

def create_visualization(images, attention, label_row, save_path):
    """Create compact visualization for poster"""
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: Images
    axes[0, 0].imshow(images['dapi'], cmap='Blues')
    axes[0, 0].set_title(f'DAPI\nAttn: {attention[0]:.2f}', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(images['tubulin'], cmap='Greens')
    axes[0, 1].set_title(f'Tubulin\nAttn: {attention[1]:.2f}', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(images['actin'], cmap='Reds')
    axes[0, 2].set_title(f'Actin\nAttn: {attention[2]:.2f}', fontsize=10, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Composite and analysis
    composite = np.stack([
        images['actin'] / images['actin'].max(),
        images['tubulin'] / images['tubulin'].max(),
        images['dapi'] / images['dapi'].max()
    ], axis=-1)
    axes[1, 0].imshow(composite)
    axes[1, 0].set_title('Composite', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Similarity bars
    sims = ['DMSO', 'MOA', 'Cross']
    vals = [label_row['dmso_similarity'], label_row['moa_similarity'], 
            label_row.get('cross_moa_similarity', 0.0)]
    colors = ['gray', 'blue', 'orange']
    axes[1, 1].bar(sims, vals, color=colors, alpha=0.6, edgecolor='black')
    axes[1, 1].axhline(0.80, color='green', linestyle='--', linewidth=1)
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_title('Similarities', fontsize=10, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Classification
    axes[1, 2].axis('off')
    resistance_type = label_row['resistance_type']
    color = 'green' if resistance_type == 'SENSITIVE' else 'red'
    
    axes[1, 2].text(0.5, 0.8, resistance_type,
                   ha='center', fontsize=14, fontweight='bold',
                   color=color, transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
    
    axes[1, 2].text(0.5, 0.5, f"MOA Sim: {label_row['moa_similarity']:.3f}",
                   ha='center', fontsize=9, transform=axes[1, 2].transAxes)
    
    axes[1, 2].text(0.5, 0.3, f"{label_row['compound']}",
                   ha='center', fontsize=8, transform=axes[1, 2].transAxes)
    
    plt.suptitle(f'Sample {label_row["idx"]}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("🎨 Creating visualizations for poster...")
    
    # Load model
    print("\n📂 Loading model...")
    model = MOAEncoder()
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    
    # Load data
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(Config.IMAGE_CSV)
    
    # Filter to Week1-6
    week_pattern = '|'.join([f'Week{i}/' for i in range(1, 7)])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    # Sample examples
    print("\n📊 Sampling examples...")
    sensitive = labels_df[labels_df['resistance_type'] == 'SENSITIVE'].sample(min(5, len(labels_df[labels_df['resistance_type'] == 'SENSITIVE'])), random_state=42)
    resistant = labels_df[labels_df['resistance_type'] != 'SENSITIVE'].sample(min(5, len(labels_df[labels_df['resistance_type'] != 'SENSITIVE'])), random_state=42)
    
    transform = A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    model.eval()
    
    # Process sensitive
    print(f"\n🟢 Creating SENSITIVE examples...")
    for i, (idx, row) in enumerate(sensitive.iterrows()):
        try:
            image_row = image_df.iloc[row['idx']]
            
            dapi_path = Path(Config.DATA_DIR) / image_row['Image_PathName_DAPI'] / image_row['Image_FileName_DAPI']
            tubulin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Tubulin'] / image_row['Image_FileName_Tubulin']
            actin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Actin'] / image_row['Image_FileName_Actin']
            
            def load_norm(path):
                img = np.array(Image.open(path)).astype(np.float32)
                c_min, c_max = img.min(), img.max()
                if c_max > c_min:
                    img = 255 * (img - c_min) / (c_max - c_min)
                return img.astype(np.uint8)
            
            dapi = load_norm(dapi_path)
            tubulin = load_norm(tubulin_path)
            actin = load_norm(actin_path)
            
            # Get attention
            dapi_t = transform(image=np.stack([dapi]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
            tubulin_t = transform(image=np.stack([tubulin]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
            actin_t = transform(image=np.stack([actin]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
            conc_t = torch.tensor([[row['concentration']]], dtype=torch.float32).to(Config.DEVICE)
            
            with torch.no_grad():
                _, attention = model(dapi_t, tubulin_t, actin_t, conc_t)
                attention = attention.cpu().numpy()[0]
            
            images = {'dapi': dapi, 'tubulin': tubulin, 'actin': actin}
            save_path = os.path.join(Config.OUTPUT_DIR, f'sensitive_{i+1}.png')
            create_visualization(images, attention, row, save_path)
            
            print(f"   ✓ saved: sensitive_{i+1}.png")
        except Exception as e:
            print(f"   ✗ Error on {row['idx']}: {e}")
    
    # Process resistant
    print(f"\n🔴 Creating RESISTANT examples...")
    for i, (idx, row) in enumerate(resistant.iterrows()):
        try:
            image_row = image_df.iloc[row['idx']]
            
            dapi_path = Path(Config.DATA_DIR) / image_row['Image_PathName_DAPI'] / image_row['Image_FileName_DAPI']
            tubulin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Tubulin'] / image_row['Image_FileName_Tubulin']
            actin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Actin'] / image_row['Image_FileName_Actin']
            
            def load_norm(path):
                img = np.array(Image.open(path)).astype(np.float32)
                c_min, c_max = img.min(), img.max()
                if c_max > c_min:
                    img = 255 * (img - c_min) / (c_max - c_min)
                return img.astype(np.uint8)
            
            dapi = load_norm(dapi_path)
            tubulin = load_norm(tubulin_path)
            actin = load_norm(actin_path)
            
            # Get attention
            dapi_t = transform(image=np.stack([dapi]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
            tubulin_t = transform(image=np.stack([tubulin]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
            actin_t = transform(image=np.stack([actin]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
            conc_t = torch.tensor([[row['concentration']]], dtype=torch.float32).to(Config.DEVICE)
            
            with torch.no_grad():
                _, attention = model(dapi_t, tubulin_t, actin_t, conc_t)
                attention = attention.cpu().numpy()[0]
            
            images = {'dapi': dapi, 'tubulin': tubulin, 'actin': actin}
            save_path = os.path.join(Config.OUTPUT_DIR, f'resistant_{i+1}.png')
            create_visualization(images, attention, row, save_path)
            
            print(f"   ✓ saved: resistant_{i+1}.png ({row['resistance_type']})")
        except Exception as e:
            print(f"   ✗ Error on {row['idx']}: {e}")
    
    print(f"\n✅ Visualizations complete!")
    print(f"   Saved to: {Config.OUTPUT_DIR}/")
    print(f"\nFiles created:")
    print(f"   • 5 sensitive examples: sensitive_1.png to sensitive_5.png")
    print(f"   • 5 resistant examples: resistant_1.png to resistant_5.png")

if __name__ == '__main__':
    main()
