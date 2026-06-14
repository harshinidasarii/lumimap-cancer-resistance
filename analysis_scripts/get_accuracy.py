"""
Simple Accuracy Calculator
===========================

Just prints the MOA classification accuracy number.
No fancy visualizations - just the number!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    MOA_CSV = './data/BBBC021_v1_moa.csv'
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128
    BATCH_SIZE = 64

class ChannelEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone = mobilenet.features
        self.projector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, output_dim))
    def forward(self, x):
        return F.normalize(self.projector(self.backbone(x)), dim=1)

class ConcentrationEncoder(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, output_dim))
    def forward(self, concentration):
        return self.encoder(torch.log1p(concentration))

class ChannelAttention(nn.Module):
    def __init__(self, num_channels=3, embed_dim=128):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(embed_dim * num_channels, 64), nn.ReLU(), 
                                      nn.Linear(64, num_channels), nn.Softmax(dim=1))
    def forward(self, channel_embeddings):
        batch_size = channel_embeddings.shape[0]
        weights = self.attention(channel_embeddings.view(batch_size, -1)).unsqueeze(2)
        return channel_embeddings * weights, weights.squeeze(2)

class MOAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dapi_encoder = ChannelEncoder(Config.EMBEDDING_DIM)
        self.tubulin_encoder = ChannelEncoder(Config.EMBEDDING_DIM)
        self.actin_encoder = ChannelEncoder(Config.EMBEDDING_DIM)
        self.concentration_encoder = ConcentrationEncoder(16)
        self.channel_attention = ChannelAttention(3, Config.EMBEDDING_DIM)
        self.fusion = nn.Sequential(nn.Linear(Config.EMBEDDING_DIM * 3 + 16, 256), nn.ReLU(),
                                    nn.Dropout(0.2), nn.Linear(256, Config.EMBEDDING_DIM))
    
    def forward(self, dapi, tubulin, actin, concentration):
        dapi_emb = self.dapi_encoder(dapi)
        tubulin_emb = self.tubulin_encoder(tubulin)
        actin_emb = self.actin_encoder(actin)
        channel_embs = torch.stack([dapi_emb, tubulin_emb, actin_emb], dim=1)
        weighted_channels, attention_weights = self.channel_attention(channel_embs)
        weighted_flat = weighted_channels.view(weighted_channels.shape[0], -1)
        conc_emb = self.concentration_encoder(concentration)
        combined = torch.cat([weighted_flat, conc_emb], dim=1)
        return F.normalize(self.fusion(combined), dim=1), attention_weights

def main():
    print("Calculating MOA classification accuracy...")
    
    # Load model
    model = MOAEncoder()
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    # Load data
    image_df = pd.read_csv(Config.IMAGE_CSV)
    moa_df = pd.read_csv(Config.MOA_CSV)
    
    week_pattern = '|'.join([f'Week{i}/' for i in range(1, 7)])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    data = image_df.merge(moa_df, 
                         left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
                         right_on=['compound', 'concentration'], how='inner')
    
    # Create dataset
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, data_df, root_dir):
            self.data = data_df
            self.root_dir = Path(root_dir)
            self.transform = A.Compose([A.Resize(*Config.IMG_SIZE), 
                                       A.Normalize(mean=[0.5], std=[0.5]), ToTensorV2()])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            
            def load_norm(path):
                img = np.array(Image.open(path)).astype(np.float32)
                c_min, c_max = img.min(), img.max()
                if c_max > c_min:
                    img = 255 * (img - c_min) / (c_max - c_min)
                return img.astype(np.uint8)
            
            dapi_path = self.root_dir / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
            tubulin_path = self.root_dir / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
            actin_path = self.root_dir / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
            
            dapi = self.transform(image=np.stack([load_norm(dapi_path)]*3, axis=-1))['image'][0:1]
            tubulin = self.transform(image=np.stack([load_norm(tubulin_path)]*3, axis=-1))['image'][0:1]
            actin = self.transform(image=np.stack([load_norm(actin_path)]*3, axis=-1))['image'][0:1]
            
            return {
                'dapi': dapi, 'tubulin': tubulin, 'actin': actin,
                'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
                'moa': row['moa']
            }
    
    dataset = SimpleDataset(data, Config.DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Extract embeddings
    all_embeddings = []
    all_moas = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing'):
            dapi = batch['dapi'].to(Config.DEVICE)
            tubulin = batch['tubulin'].to(Config.DEVICE)
            actin = batch['actin'].to(Config.DEVICE)
            conc = batch['concentration'].to(Config.DEVICE)
            
            embeddings, _ = model(dapi, tubulin, actin, conc)
            all_embeddings.append(embeddings.cpu())
            all_moas.extend(batch['moa'])
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Compute centroids
    unique_moas = sorted(list(set(all_moas)))
    moa_centroids = {}
    
    for moa in unique_moas:
        moa_mask = [m == moa for m in all_moas]
        moa_embeddings = all_embeddings[moa_mask]
        moa_centroids[moa] = moa_embeddings.mean(dim=0)
    
    # Predict
    predictions = []
    for embedding in all_embeddings:
        similarities = {}
        for moa, centroid in moa_centroids.items():
            sim = F.cosine_similarity(embedding.unsqueeze(0), centroid.unsqueeze(0)).item()
            similarities[moa] = sim
        predictions.append(max(similarities, key=similarities.get))
    
    # Calculate accuracy
    accuracy = accuracy_score(all_moas, predictions)
    
    # Print results
    print("\n" + "="*50)
    print("LUMIMAP MOA CLASSIFICATION ACCURACY")
    print("="*50)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"\nSamples tested: {len(all_moas)}")
    print(f"MOA classes: {len(unique_moas)}")
    print("\n" + "="*50)

if __name__ == '__main__':
    main()
