"""
Demo Script - Single Image Prediction
======================================

Usage:
    python demo_predict.py --idx 100
    
Shows prediction for a single cell image with visualization
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
    MODEL_PATH = './output/phase3_binary/phase3_binary_best.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128

# [Model classes - copy from gradcam]
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

class ResistanceClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = MOAEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, dapi, tubulin, actin, concentration):
        embeddings, attention = self.encoder(dapi, tubulin, actin, concentration)
        logits = self.classifier(embeddings)
        return logits, attention, embeddings

def predict_single(model, dapi_path, tubulin_path, actin_path, concentration):
    """Predict resistance for single sample"""
    
    # Load and normalize
    def load_and_normalize(path):
        img = np.array(Image.open(path))
        img = img.astype(np.float32)
        c_min, c_max = img.min(), img.max()
        if c_max > c_min:
            img = 255 * (img - c_min) / (c_max - c_min)
        return img.astype(np.uint8)
    
    dapi = load_and_normalize(dapi_path)
    tubulin = load_and_normalize(tubulin_path)
    actin = load_and_normalize(actin_path)
    
    # Transform
    transform = A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    dapi_t = transform(image=np.stack([dapi]*3, axis=-1))['image'][0:1].unsqueeze(0)
    tubulin_t = transform(image=np.stack([tubulin]*3, axis=-1))['image'][0:1].unsqueeze(0)
    actin_t = transform(image=np.stack([actin]*3, axis=-1))['image'][0:1].unsqueeze(0)
    conc_t = torch.tensor([[concentration]], dtype=torch.float32)
    
    # Predict
    model.eval()
    with torch.no_grad():
        dapi_t = dapi_t.to(Config.DEVICE)
        tubulin_t = tubulin_t.to(Config.DEVICE)
        actin_t = actin_t.to(Config.DEVICE)
        conc_t = conc_t.to(Config.DEVICE)
        
        logits, attention, embeddings = model(dapi_t, tubulin_t, actin_t, conc_t)
        probs = F.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted].item()
        
        attention = attention.cpu().numpy()[0]
        sensitive_prob = probs[0, 0].item()
        resistant_prob = probs[0, 1].item()
    
    return {
        'predicted_class': predicted,
        'predicted_label': 'RESISTANT' if predicted == 1 else 'SENSITIVE',
        'confidence': confidence,
        'sensitive_prob': sensitive_prob,
        'resistant_prob': resistant_prob,
        'attention_weights': {
            'dapi': attention[0],
            'tubulin': attention[1],
            'actin': attention[2]
        },
        'images': {
            'dapi': dapi,
            'tubulin': tubulin,
            'actin': actin
        }
    }

def visualize_prediction(result, compound, concentration, true_label=None):
    """Visualize prediction with images and attention"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Images
    axes[0, 0].imshow(result['images']['dapi'], cmap='Blues')
    axes[0, 0].set_title(f"DAPI\\nAttention: {result['attention_weights']['dapi']:.3f}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(result['images']['tubulin'], cmap='Greens')
    axes[0, 1].set_title(f"Tubulin\\nAttention: {result['attention_weights']['tubulin']:.3f}")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result['images']['actin'], cmap='Reds')
    axes[0, 2].set_title(f"Actin\\nAttention: {result['attention_weights']['actin']:.3f}")
    axes[0, 2].axis('off')
    
    # Composite RGB
    composite = np.stack([
        result['images']['actin'] / result['images']['actin'].max(),
        result['images']['tubulin'] / result['images']['tubulin'].max(),
        result['images']['dapi'] / result['images']['dapi'].max()
    ], axis=-1)
    axes[1, 0].imshow(composite)
    axes[1, 0].set_title('RGB Composite')
    axes[1, 0].axis('off')
    
    # Attention bars
    channels = ['DAPI', 'Tubulin', 'Actin']
    colors = ['blue', 'green', 'red']
    attention_vals = [result['attention_weights']['dapi'], 
                     result['attention_weights']['tubulin'],
                     result['attention_weights']['actin']]
    axes[1, 1].bar(channels, attention_vals, color=colors, alpha=0.6)
    axes[1, 1].set_ylabel('Attention Weight')
    axes[1, 1].set_title('Channel Attention')
    axes[1, 1].set_ylim([0, 1])
    
    # Prediction
    axes[1, 2].text(0.1, 0.9, f'Compound: {compound}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.75, f'Concentration: {concentration:.2e}', fontsize=12, transform=axes[1, 2].transAxes)
    if true_label:
        axes[1, 2].text(0.1, 0.6, f'True: {true_label}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.45, f"Predicted: {result['predicted_label']}", fontsize=14, 
                   fontweight='bold', transform=axes[1, 2].transAxes,
                   color='red' if result['predicted_label'] == 'RESISTANT' else 'green')
    axes[1, 2].text(0.1, 0.3, f"Confidence: {result['confidence']:.1%}", fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.15, f"Sensitive: {result['sensitive_prob']:.1%}", fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.05, f"Resistant: {result['resistant_prob']:.1%}", fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict resistance for single image')
    parser.add_argument('--idx', type=int, required=True, help='Image index to predict')
    args = parser.parse_args()
    
    print("🔬 LUMIMAP Resistance Prediction Demo")
    print("="*50)
    
    # Load model
    print("\n📂 Loading model...")
    model = ResistanceClassifier(num_classes=2)
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    print(f"   ✓ Model loaded")
    
    # Load data
    print("\n📂 Loading data...")
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(Config.IMAGE_CSV)
    
    # Get sample
    if args.idx not in labels_df['idx'].values:
        print(f"❌ Index {args.idx} not found in labels!")
        return
    
    label_row = labels_df[labels_df['idx'] == args.idx].iloc[0]
    image_row = image_df.iloc[args.idx]
    
    print(f"\n🔍 Sample {args.idx}:")
    print(f"   Compound: {label_row['compound']}")
    print(f"   Concentration: {label_row['concentration']:.2e}")
    print(f"   MOA: {label_row['moa']}")
    print(f"   True label: {label_row['resistance_type']}")
    
    # Get paths
    dapi_path = Path(Config.DATA_DIR) / image_row['Image_PathName_DAPI'] / image_row['Image_FileName_DAPI']
    tubulin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Tubulin'] / image_row['Image_FileName_Tubulin']
    actin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Actin'] / image_row['Image_FileName_Actin']
    
    # Predict
    print("\n🤖 Predicting...")
    result = predict_single(model, dapi_path, tubulin_path, actin_path, label_row['concentration'])
    
    print(f"\n✅ Prediction: {result['predicted_label']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Sensitive probability: {result['sensitive_prob']:.1%}")
    print(f"   Resistant probability: {result['resistant_prob']:.1%}")
    print(f"\n   Channel Attention:")
    print(f"      DAPI: {result['attention_weights']['dapi']:.3f}")
    print(f"      Tubulin: {result['attention_weights']['tubulin']:.3f}")
    print(f"      Actin: {result['attention_weights']['actin']:.3f}")
    
    # Visualize
    true_binary = 'RESISTANT' if label_row['resistance_type'] != 'SENSITIVE' else 'SENSITIVE'
    visualize_prediction(result, label_row['compound'], label_row['concentration'], true_binary)

if __name__ == '__main__':
    main()
