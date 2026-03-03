"""
Batch Prediction Script
=======================

Run predictions on multiple samples and generate report

Usage:
    python batch_predict.py --n 20  # Predict on 20 random samples
    python batch_predict.py --all   # Predict on all samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse
import os

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    LABELS_CSV = './output/phase2_strategic/resistance_labels.csv'
    MODEL_PATH = './output/phase3_binary/phase3_binary_best.pth'
    OUTPUT_DIR = './output/batch_predictions'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128

# [Model classes - same as before]
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
        return logits, attention

def batch_predict(model, data_df, image_df):
    """Predict on batch of samples"""
    
    results = []
    
    transform = A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    model.eval()
    
    for idx, label_row in tqdm(data_df.iterrows(), total=len(data_df), desc='Predicting'):
        image_row = image_df.iloc[label_row['idx']]
        
        # Load images
        try:
            dapi_path = Path(Config.DATA_DIR) / image_row['Image_PathName_DAPI'] / image_row['Image_FileName_DAPI']
            tubulin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Tubulin'] / image_row['Image_FileName_Tubulin']
            actin_path = Path(Config.DATA_DIR) / image_row['Image_PathName_Actin'] / image_row['Image_FileName_Actin']
            
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
            dapi_t = transform(image=np.stack([dapi]*3, axis=-1))['image'][0:1].unsqueeze(0)
            tubulin_t = transform(image=np.stack([tubulin]*3, axis=-1))['image'][0:1].unsqueeze(0)
            actin_t = transform(image=np.stack([actin]*3, axis=-1))['image'][0:1].unsqueeze(0)
            conc_t = torch.tensor([[label_row['concentration']]], dtype=torch.float32)
            
            # Predict
            with torch.no_grad():
                dapi_t = dapi_t.to(Config.DEVICE)
                tubulin_t = tubulin_t.to(Config.DEVICE)
                actin_t = actin_t.to(Config.DEVICE)
                conc_t = conc_t.to(Config.DEVICE)
                
                logits, attention = model(dapi_t, tubulin_t, actin_t, conc_t)
                probs = F.softmax(logits, dim=1)
                predicted = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted].item()
                
                attention = attention.cpu().numpy()[0]
            
            true_binary = 'RESISTANT' if label_row['resistance_type'] != 'SENSITIVE' else 'SENSITIVE'
            pred_label = 'RESISTANT' if predicted == 1 else 'SENSITIVE'
            
            results.append({
                'idx': label_row['idx'],
                'compound': label_row['compound'],
                'concentration': label_row['concentration'],
                'moa': label_row['moa'],
                'true_type': label_row['resistance_type'],
                'true_binary': true_binary,
                'predicted': pred_label,
                'confidence': confidence,
                'sensitive_prob': probs[0, 0].item(),
                'resistant_prob': probs[0, 1].item(),
                'attention_dapi': attention[0],
                'attention_tubulin': attention[1],
                'attention_actin': attention[2],
                'correct': (true_binary == pred_label)
            })
            
        except Exception as e:
            print(f"   ⚠️  Error on idx {label_row['idx']}: {e}")
            continue
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Batch prediction')
    parser.add_argument('--n', type=int, help='Number of random samples')
    parser.add_argument('--all', action='store_true', help='Predict on all samples')
    args = parser.parse_args()
    
    print("🚀 LUMIMAP Batch Prediction")
    print("="*50)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
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
    
    # Filter to Week1-6
    week_pattern = '|'.join([f'Week{i}/' for i in range(1, 7)])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    # Sample
    if args.all:
        data_to_predict = labels_df
        print(f"   Predicting on all {len(data_to_predict)} samples")
    elif args.n:
        data_to_predict = labels_df.sample(min(args.n, len(labels_df)), random_state=42)
        print(f"   Predicting on {len(data_to_predict)} random samples")
    else:
        data_to_predict = labels_df.sample(20, random_state=42)
        print(f"   Predicting on 20 random samples (default)")
    
    # Predict
    results_df = batch_predict(model, data_to_predict, image_df)
    
    # Save
    output_file = os.path.join(Config.OUTPUT_DIR, 'predictions.csv')
    results_df.to_csv(output_file, index=False)
    
    # Statistics
    print("\n" + "="*50)
    print("📊 RESULTS")
    print("="*50)
    print(f"Total samples: {len(results_df)}")
    print(f"Correct: {results_df['correct'].sum()} ({100*results_df['correct'].mean():.1f}%)")
    print()
    print("By true label:")
    for label in results_df['true_binary'].unique():
        subset = results_df[results_df['true_binary'] == label]
        acc = 100 * subset['correct'].mean()
        print(f"  {label}: {subset['correct'].sum()}/{len(subset)} ({acc:.1f}%)")
    print()
    print("Predicted distribution:")
    print(results_df['predicted'].value_counts())
    print()
    print("Average attention weights:")
    print(f"  DAPI: {results_df['attention_dapi'].mean():.3f}")
    print(f"  Tubulin: {results_df['attention_tubulin'].mean():.3f}")
    print(f"  Actin: {results_df['attention_actin'].mean():.3f}")
    print()
    print(f"✅ Results saved to: {output_file}")

if __name__ == '__main__':
    main()
