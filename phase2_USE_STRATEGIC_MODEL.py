"""
Phase 2 - Using STRATEGIC Model
================================

UPDATED to use: ./output/phase1_strategic/phase1_strategic_best.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
from enum import Enum
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    MOA_CSV = './data/BBBC021_v1_moa.csv'
    OUTPUT_DIR = './output/phase2_strategic'
    
    # USE STRATEGIC MODEL HERE!
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    
    # Relaxed thresholds
    DMSO_HIGH_SIMILARITY = 0.85
    MOA_HIGH_SIMILARITY = 0.80
    MOA_MEDIUM_SIMILARITY = 0.65
    CROSS_MOA_THRESHOLD = 0.70
    
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128
    
    # Use all 6 weeks for inference (we have diverse model now!)
    WEEKS_TO_USE = ['Week1', 'Week2', 'Week3', 'Week4', 'Week5', 'Week6']

print("🚀 PHASE 2 - Using STRATEGIC MODEL")
print(f"   Model: {Config.PHASE1_MODEL}")
print(f"   Weeks: {Config.WEEKS_TO_USE}")

class ResistanceType(Enum):
    SENSITIVE = 0
    PRIMARY_RESISTANCE = 1
    PARTIAL_RESISTANCE = 2
    CROSS_RESISTANCE = 3
    UNCERTAIN = 4

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

class InferenceDataset(Dataset):
    def __init__(self, image_df, moa_df, root_dir):
        self.root_dir = Path(root_dir)
        self.data = image_df.merge(
            moa_df,
            left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
            right_on=['compound', 'concentration'],
            how='inner'
        )
        self.transform = A.Compose([
            A.Resize(*Config.IMG_SIZE),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
        print(f"📊 Dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dapi_path = self.root_dir / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
        tubulin_path = self.root_dir / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
        actin_path = self.root_dir / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
        
        dapi = np.array(Image.open(dapi_path))
        tubulin = np.array(Image.open(tubulin_path))
        actin = np.array(Image.open(actin_path))
        
        dapi = self._normalize_channel(dapi)
        tubulin = self._normalize_channel(tubulin)
        actin = self._normalize_channel(actin)
        
        dapi = self._apply_transform(dapi)
        tubulin = self._apply_transform(tubulin)
        actin = self._apply_transform(actin)
        
        return {
            'dapi': dapi, 'tubulin': tubulin, 'actin': actin,
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'compound': row['compound'], 'moa': row['moa'], 'idx': idx
        }
    
    def _normalize_channel(self, channel):
        channel = channel.astype(np.float32)
        c_min, c_max = channel.min(), channel.max()
        if c_max > c_min:
            channel = 255 * (channel - c_min) / (c_max - c_min)
        return channel.astype(np.uint8)
    
    def _apply_transform(self, channel):
        channel_3ch = np.stack([channel, channel, channel], axis=-1)
        augmented = self.transform(image=channel_3ch)
        return augmented['image'][0:1]

class ResistanceLabelGenerator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.moa_references = {}
        self.dmso_reference = None
        self.all_embeddings = []
        self.all_metadata = []
    
    def extract_embeddings(self):
        print("\n📊 Extracting embeddings...")
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Processing'):
                dapi = batch['dapi'].to(self.device)
                tubulin = batch['tubulin'].to(self.device)
                actin = batch['actin'].to(self.device)
                conc = batch['concentration'].to(self.device)
                embeddings, attention = self.model(dapi, tubulin, actin, conc)
                self.all_embeddings.append(embeddings.cpu())
                self.all_metadata.extend([
                    {
                        'compound': batch['compound'][i],
                        'concentration': batch['concentration'][i].item(),
                        'moa': batch['moa'][i],
                        'attention': attention[i].cpu().numpy(),
                        'idx': batch['idx'][i].item()
                    }
                    for i in range(len(batch['compound']))
                ])
        self.all_embeddings = torch.cat(self.all_embeddings, dim=0)
        print(f"   ✓ Extracted {len(self.all_embeddings)} embeddings")
    
    def compute_moa_references(self):
        print("\n📊 Computing MOA references...")
        moa_groups = defaultdict(list)
        for i, meta in enumerate(self.all_metadata):
            moa_groups[meta['moa']].append(i)
        for moa, indices in moa_groups.items():
            moa_embeddings = self.all_embeddings[indices]
            reference = moa_embeddings.mean(dim=0)
            self.moa_references[moa] = reference
            if moa == 'DMSO':
                self.dmso_reference = reference
                print(f"   ✓ DMSO baseline: {len(indices)} samples")
        print(f"   ✓ {len(self.moa_references)} MOA references")
    
    def classify_resistance(self):
        print("\n📊 Classifying...")
        labels = []
        for i, meta in enumerate(tqdm(self.all_metadata, desc='Classifying')):
            embedding = self.all_embeddings[i]
            moa = meta['moa']
            
            if moa == 'DMSO':
                labels.append({**meta, 'resistance_type': ResistanceType.SENSITIVE.name,
                             'resistance_code': ResistanceType.SENSITIVE.value,
                             'dmso_similarity': 1.0, 'moa_similarity': 1.0,
                             'cross_moa': None, 'cross_moa_similarity': 0.0})
                continue
            
            dmso_sim = F.cosine_similarity(embedding.unsqueeze(0), self.dmso_reference.unsqueeze(0)).item()
            moa_sim = F.cosine_similarity(embedding.unsqueeze(0), self.moa_references[moa].unsqueeze(0)).item()
            
            cross_moa_sims = {}
            for other_moa, ref in self.moa_references.items():
                if other_moa != moa and other_moa != 'DMSO':
                    sim = F.cosine_similarity(embedding.unsqueeze(0), ref.unsqueeze(0)).item()
                    cross_moa_sims[other_moa] = sim
            
            if cross_moa_sims:
                max_cross_moa = max(cross_moa_sims, key=cross_moa_sims.get)
                max_cross_sim = cross_moa_sims[max_cross_moa]
            else:
                max_cross_moa = None
                max_cross_sim = 0.0
            
            resistance_type = self._classify_sample(dmso_sim, moa_sim, max_cross_sim)
            labels.append({
                **meta, 'resistance_type': resistance_type.name,
                'resistance_code': resistance_type.value,
                'dmso_similarity': dmso_sim, 'moa_similarity': moa_sim,
                'cross_moa': max_cross_moa if resistance_type == ResistanceType.CROSS_RESISTANCE else None,
                'cross_moa_similarity': max_cross_sim
            })
        return pd.DataFrame(labels)
    
    def _classify_sample(self, dmso_sim, moa_sim, cross_moa_sim):
        if dmso_sim >= Config.DMSO_HIGH_SIMILARITY and moa_sim < Config.MOA_MEDIUM_SIMILARITY:
            return ResistanceType.PRIMARY_RESISTANCE
        if moa_sim >= Config.MOA_HIGH_SIMILARITY:
            return ResistanceType.SENSITIVE
        if moa_sim < Config.MOA_MEDIUM_SIMILARITY and cross_moa_sim >= Config.CROSS_MOA_THRESHOLD:
            return ResistanceType.CROSS_RESISTANCE
        if moa_sim >= Config.MOA_MEDIUM_SIMILARITY:
            return ResistanceType.PARTIAL_RESISTANCE
        return ResistanceType.UNCERTAIN

def main():
    print("\n" + "="*70)
    print("LUMIMAP PHASE 2 - Using STRATEGIC MODEL")
    print("="*70)
    
    print("\n📂 Loading metadata...")
    image_df = pd.read_csv(Config.IMAGE_CSV)
    moa_df = pd.read_csv(Config.MOA_CSV)
    
    print(f"🔍 Filtering to: {Config.WEEKS_TO_USE}")
    week_pattern = '|'.join([f'{w}/' for w in Config.WEEKS_TO_USE])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    print(f"   ✓ {len(image_df)} images")
    
    dataset = InferenceDataset(image_df, moa_df, Config.DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                          num_workers=Config.NUM_WORKERS, pin_memory=False)
    
    print(f"\n🧠 Loading STRATEGIC model...")
    model = MOAEncoder()
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    print(f"   ✓ Loaded from {Config.PHASE1_MODEL}")
    
    generator = ResistanceLabelGenerator(model, dataloader, Config.DEVICE)
    generator.extract_embeddings()
    generator.compute_moa_references()
    labels_df = generator.classify_resistance()
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(Config.OUTPUT_DIR, 'resistance_labels.csv')
    labels_df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("RESISTANCE CLASSIFICATION SUMMARY")
    print("="*70)
    print(labels_df['resistance_type'].value_counts())
    print("\nAverage Similarities:")
    print(labels_df.groupby('resistance_type')[['dmso_similarity', 'moa_similarity']].mean())
    
    n_types = len(labels_df['resistance_type'].unique())
    print(f"\n📊 Resistance types found: {n_types}")
    
    if n_types >= 3:
        print("✅ EXCELLENT! Multiple resistance types detected")
        print("   Ready for Phase 3!")
    else:
        print(f"⚠️  Only {n_types} types found")
    
    print(f"\n✅ PHASE 2 COMPLETE!")
    print(f"   Labels: {output_file}")
    print("="*70)

if __name__ == '__main__':
    main()
