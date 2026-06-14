"""
GradCAM Visualization for Resistance Detection
===============================================

Shows which parts of cell images the model focuses on
when making resistance predictions
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

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    LABELS_CSV = './output/phase2_strategic/resistance_labels.csv'
    
    # Use binary model
    MODEL_PATH = './output/phase3_binary/phase3_binary_best.pth'
    OUTPUT_DIR = './output/gradcam_visualizations'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128

# [Model classes - same as training]
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

def normalize_channel(channel):
    channel = channel.astype(np.float32)
    c_min, c_max = channel.min(), channel.max()
    if c_max > c_min:
        channel = 255 * (channel - c_min) / (c_max - c_min)
    return channel.astype(np.uint8)

def visualize_sample(model, dapi_path, tubulin_path, actin_path, concentration, 
                     true_label, compound, save_path):
    """Visualize single sample with attention weights"""
    
    # Load images
    dapi = np.array(Image.open(dapi_path))
    tubulin = np.array(Image.open(tubulin_path))
    actin = np.array(Image.open(actin_path))
    
    # Normalize
    dapi_norm = normalize_channel(dapi)
    tubulin_norm = normalize_channel(tubulin)
    actin_norm = normalize_channel(actin)
    
    # Transform
    transform = A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    dapi_t = transform(image=np.stack([dapi_norm]*3, axis=-1))['image'][0:1].unsqueeze(0)
    tubulin_t = transform(image=np.stack([tubulin_norm]*3, axis=-1))['image'][0:1].unsqueeze(0)
    actin_t = transform(image=np.stack([actin_norm]*3, axis=-1))['image'][0:1].unsqueeze(0)
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
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(dapi, cmap='Blues')
    axes[0, 0].set_title(f'DAPI (Nucleus)\nAttention: {attention[0]:.3f}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tubulin, cmap='Greens')
    axes[0, 1].set_title(f'Tubulin (Microtubules)\nAttention: {attention[1]:.3f}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(actin, cmap='Reds')
    axes[0, 2].set_title(f'Actin (Cytoskeleton)\nAttention: {attention[2]:.3f}')
    axes[0, 2].axis('off')
    
    # Composite
    composite = np.stack([
        actin / actin.max(),
        tubulin / tubulin.max(),
        dapi / dapi.max()
    ], axis=-1)
    axes[1, 0].imshow(composite)
    axes[1, 0].set_title('RGB Composite')
    axes[1, 0].axis('off')
    
    # Attention bar chart
    channels = ['DAPI', 'Tubulin', 'Actin']
    colors = ['blue', 'green', 'red']
    axes[1, 1].bar(channels, attention, color=colors, alpha=0.6)
    axes[1, 1].set_ylabel('Attention Weight')
    axes[1, 1].set_title('Channel Attention')
    axes[1, 1].set_ylim([0, 1])
    
    # Prediction info
    pred_label = 'RESISTANT' if predicted == 1 else 'SENSITIVE'
    axes[1, 2].text(0.1, 0.9, f'Compound: {compound}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f'Concentration: {concentration:.2e}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'True: {true_label}', fontsize=12, transform=axes[1, 2].transAxes, 
                   color='green' if true_label == pred_label else 'red')
    axes[1, 2].text(0.1, 0.3, f'Predicted: {pred_label}', fontsize=14, fontweight='bold', 
                   transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.1, f'Confidence: {confidence:.1%}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'predicted': pred_label,
        'confidence': confidence,
        'attention': attention.tolist(),
        'correct': (true_label == pred_label)
    }

def main():
    import os
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("📂 Loading model...")
    model = ResistanceClassifier(num_classes=2)
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    print(f"   ✓ Loaded from {Config.MODEL_PATH}")
    
    print("\n📂 Loading data...")
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(Config.IMAGE_CSV)
    
    # Filter to Week1-6
    week_pattern = '|'.join([f'Week{i}/' for i in range(1, 7)])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    # Merge
    data = labels_df.merge(image_df, left_on='idx', right_index=True, how='inner')
    
    # Binary labels
    data['binary_label'] = data['resistance_type'].apply(
        lambda x: 'RESISTANT' if x != 'SENSITIVE' else 'SENSITIVE'
    )
    
    print(f"\n📊 Generating visualizations...")
    print(f"   Total samples: {len(data)}")
    
    # Sample examples from each class
    sensitive_samples = data[data['binary_label'] == 'SENSITIVE'].sample(min(5, len(data[data['binary_label'] == 'SENSITIVE'])), random_state=42)
    resistant_samples = data[data['binary_label'] == 'RESISTANT'].sample(min(5, len(data[data['binary_label'] == 'RESISTANT'])), random_state=42)
    
    results = []
    
    print("\n🔬 Processing SENSITIVE samples...")
    for idx, row in sensitive_samples.iterrows():
        dapi_path = Path(Config.DATA_DIR) / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
        tubulin_path = Path(Config.DATA_DIR) / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
        actin_path = Path(Config.DATA_DIR) / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
        
        save_path = os.path.join(Config.OUTPUT_DIR, f'sensitive_{idx}.png')
        
        result = visualize_sample(
            model, dapi_path, tubulin_path, actin_path,
            row['concentration'], row['binary_label'], row['compound'], save_path
        )
        results.append({**result, 'true_label': row['binary_label'], 'idx': idx})
        print(f"   ✓ {idx}: {result['predicted']} ({result['confidence']:.1%})")
    
    print("\n🔬 Processing RESISTANT samples...")
    for idx, row in resistant_samples.iterrows():
        dapi_path = Path(Config.DATA_DIR) / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
        tubulin_path = Path(Config.DATA_DIR) / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
        actin_path = Path(Config.DATA_DIR) / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
        
        save_path = os.path.join(Config.OUTPUT_DIR, f'resistant_{idx}.png')
        
        result = visualize_sample(
            model, dapi_path, tubulin_path, actin_path,
            row['concentration'], row['binary_label'], row['compound'], save_path
        )
        results.append({**result, 'true_label': row['binary_label'], 'idx': idx})
        print(f"   ✓ {idx}: {result['predicted']} ({result['confidence']:.1%})")
    
    # Summary
    correct = sum(1 for r in results if r['correct'])
    print(f"\n✅ Visualizations complete!")
    print(f"   Accuracy on samples: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"   Saved to: {Config.OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
