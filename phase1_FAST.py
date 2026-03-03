"""
LUMIMAP Phase 1 - FAST TRAINING VERSION
========================================

Optimizations for speed:
- Smaller model (MobileNetV2 instead of ResNet18)
- Smaller images (128x128 instead of 256x256)
- Only Week1 data (instead of Week1-3)
- Fewer epochs (15 instead of 30)
- Mixed precision training (FP16)
- Data subsampling (50% of available data)

Expected time: 30-60 minutes on GPU, 2-4 hours on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ============================================================================
# FAST CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    MOA_CSV = './data/BBBC021_v1_moa.csv'
    OUTPUT_DIR = './output/phase1_fast'
    
    # FAST SETTINGS
    BACKBONE = 'mobilenet_v2'  # Lighter and faster
    IMG_SIZE = (128, 128)  # Smaller = faster
    EMBEDDING_DIM = 128    # Smaller embeddings
    
    BATCH_SIZE = 64        # Larger batches = fewer iterations
    NUM_EPOCHS = 15        # Fewer epochs
    LEARNING_RATE = 2e-4   # Higher LR for faster convergence
    
    # Only use Week1 for speed
    WEEKS_TO_USE = ['Week1']
    
    # Subsample dataset (use 50% of data)
    SUBSAMPLE_FRACTION = 0.5
    
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision (FP16)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    
    CONCENTRATION_EMBED_DIM = 16  # Smaller
    NUM_CHANNELS = 3

print(f"🚀 FAST TRAINING MODE")
print(f"   Using: {Config.WEEKS_TO_USE}")
print(f"   Image size: {Config.IMG_SIZE}")
print(f"   Epochs: {Config.NUM_EPOCHS}")
print(f"   Subsample: {Config.SUBSAMPLE_FRACTION*100}%")

# ============================================================================
# DATASET (Same as before)
# ============================================================================

class ContrastiveDataset(Dataset):
    def __init__(self, image_df, moa_df, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.data = image_df.merge(
            moa_df,
            left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
            right_on=['compound', 'concentration'],
            how='inner'
        )
        
        self.moa_to_samples = {}
        for moa in self.data['moa'].unique():
            self.moa_to_samples[moa] = self.data[self.data['moa'] == moa].index.tolist()
        
        self.compound_to_samples = {}
        for compound in self.data['compound'].unique():
            self.compound_to_samples[compound] = self.data[self.data['compound'] == compound].index.tolist()
        
        self.moa_list = list(self.moa_to_samples.keys())
        
        print(f"Dataset: {len(self.data)} samples, {len(self.moa_list)} MOAs")
    
    def __len__(self):
        return len(self.data)
    
    def load_triplet_channels(self, idx):
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
        
        compound = row['compound']
        concentration = row['concentration']
        moa = row['moa']
        
        return dapi, tubulin, actin, compound, concentration, moa
    
    def _normalize_channel(self, channel):
        channel = channel.astype(np.float32)
        c_min, c_max = channel.min(), channel.max()
        if c_max > c_min:
            channel = 255 * (channel - c_min) / (c_max - c_min)
        return channel.astype(np.uint8)
    
    def __getitem__(self, idx):
        anchor_dapi, anchor_tubulin, anchor_actin, anchor_compound, anchor_conc, anchor_moa = \
            self.load_triplet_channels(idx)
        
        moa_samples = self.moa_to_samples[anchor_moa].copy()
        same_moa_diff_compound = [i for i in moa_samples 
                                  if self.data.iloc[i]['compound'] != anchor_compound]
        if same_moa_diff_compound:
            pos_idx = np.random.choice(same_moa_diff_compound)
        else:
            pos_idx = np.random.choice(moa_samples)
        
        pos_dapi, pos_tubulin, pos_actin, pos_compound, pos_conc, pos_moa = \
            self.load_triplet_channels(pos_idx)
        
        negative_moas = [m for m in self.moa_list if m != anchor_moa]
        neg_moa = np.random.choice(negative_moas)
        neg_idx = np.random.choice(self.moa_to_samples[neg_moa])
        
        neg_dapi, neg_tubulin, neg_actin, neg_compound, neg_conc, neg_moa = \
            self.load_triplet_channels(neg_idx)
        
        if self.transform:
            anchor_dapi = self.apply_transform(anchor_dapi)
            anchor_tubulin = self.apply_transform(anchor_tubulin)
            anchor_actin = self.apply_transform(anchor_actin)
            
            pos_dapi = self.apply_transform(pos_dapi)
            pos_tubulin = self.apply_transform(pos_tubulin)
            pos_actin = self.apply_transform(pos_actin)
            
            neg_dapi = self.apply_transform(neg_dapi)
            neg_tubulin = self.apply_transform(neg_tubulin)
            neg_actin = self.apply_transform(neg_actin)
        
        return {
            'anchor': {
                'dapi': anchor_dapi, 'tubulin': anchor_tubulin, 'actin': anchor_actin,
                'concentration': torch.tensor([anchor_conc], dtype=torch.float32),
                'moa': anchor_moa
            },
            'positive': {
                'dapi': pos_dapi, 'tubulin': pos_tubulin, 'actin': pos_actin,
                'concentration': torch.tensor([pos_conc], dtype=torch.float32),
                'moa': pos_moa
            },
            'negative': {
                'dapi': neg_dapi, 'tubulin': neg_tubulin, 'actin': neg_actin,
                'concentration': torch.tensor([neg_conc], dtype=torch.float32),
                'moa': neg_moa
            }
        }
    
    def apply_transform(self, channel):
        channel_3ch = np.stack([channel, channel, channel], axis=-1)
        if self.transform:
            augmented = self.transform(image=channel_3ch)
            channel_tensor = augmented['image'][0:1]
        return channel_tensor

# ============================================================================
# LIGHTER TRANSFORMS (Less augmentation = faster)
# ============================================================================

def get_transforms():
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

# ============================================================================
# LIGHTER MODEL (MobileNet instead of ResNet)
# ============================================================================

class ChannelEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        
        # MobileNetV2 - much faster than ResNet
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Modify first conv for 1 channel
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Remove classifier
        self.backbone = mobilenet.features
        
        # Smaller projection
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
        
        self.concentration_encoder = ConcentrationEncoder(Config.CONCENTRATION_EMBED_DIM)
        self.channel_attention = ChannelAttention(Config.NUM_CHANNELS, Config.EMBEDDING_DIM)
        
        self.fusion = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM * Config.NUM_CHANNELS + Config.CONCENTRATION_EMBED_DIM, 256),
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

# ============================================================================
# LOSS
# ============================================================================

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# ============================================================================
# TRAINER WITH MIXED PRECISION
# ============================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = Config.DEVICE
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if Config.USE_AMP else None
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            anchor_dapi = batch['anchor']['dapi'].to(self.device)
            anchor_tubulin = batch['anchor']['tubulin'].to(self.device)
            anchor_actin = batch['anchor']['actin'].to(self.device)
            anchor_conc = batch['anchor']['concentration'].to(self.device)
            
            pos_dapi = batch['positive']['dapi'].to(self.device)
            pos_tubulin = batch['positive']['tubulin'].to(self.device)
            pos_actin = batch['positive']['actin'].to(self.device)
            pos_conc = batch['positive']['concentration'].to(self.device)
            
            neg_dapi = batch['negative']['dapi'].to(self.device)
            neg_tubulin = batch['negative']['tubulin'].to(self.device)
            neg_actin = batch['negative']['actin'].to(self.device)
            neg_conc = batch['negative']['concentration'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward
            if Config.USE_AMP:
                with torch.cuda.amp.autocast():
                    anchor_emb, _ = self.model(anchor_dapi, anchor_tubulin, anchor_actin, anchor_conc)
                    pos_emb, _ = self.model(pos_dapi, pos_tubulin, pos_actin, pos_conc)
                    neg_emb, _ = self.model(neg_dapi, neg_tubulin, neg_actin, neg_conc)
                    loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                anchor_emb, _ = self.model(anchor_dapi, anchor_tubulin, anchor_actin, anchor_conc)
                pos_emb, _ = self.model(pos_dapi, pos_tubulin, pos_actin, pos_conc)
                neg_emb, _ = self.model(neg_dapi, neg_tubulin, neg_actin, neg_conc)
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return running_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                anchor_dapi = batch['anchor']['dapi'].to(self.device)
                anchor_tubulin = batch['anchor']['tubulin'].to(self.device)
                anchor_actin = batch['anchor']['actin'].to(self.device)
                anchor_conc = batch['anchor']['concentration'].to(self.device)
                
                pos_dapi = batch['positive']['dapi'].to(self.device)
                pos_tubulin = batch['positive']['tubulin'].to(self.device)
                pos_actin = batch['positive']['actin'].to(self.device)
                pos_conc = batch['positive']['concentration'].to(self.device)
                
                neg_dapi = batch['negative']['dapi'].to(self.device)
                neg_tubulin = batch['negative']['tubulin'].to(self.device)
                neg_actin = batch['negative']['actin'].to(self.device)
                neg_conc = batch['negative']['concentration'].to(self.device)
                
                anchor_emb, _ = self.model(anchor_dapi, anchor_tubulin, anchor_actin, anchor_conc)
                pos_emb, _ = self.model(pos_dapi, pos_tubulin, pos_actin, pos_conc)
                neg_emb, _ = self.model(neg_dapi, neg_tubulin, neg_actin, neg_conc)
                
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                running_loss += loss.item()
        
        return running_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        print(f"\n🚀 Training on {self.device}")
        print(f"   Mixed precision: {Config.USE_AMP}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('phase1_fast_best.pth')
                print(f"✓ Best model! Loss: {val_loss:.4f}")
        
        self.plot_losses()
        print(f"\n✅ Phase 1 FAST Complete!")
        print(f"   Best loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss
        }, os.path.join(Config.OUTPUT_DIR, filename))
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train', marker='o')
        plt.plot(self.val_losses, label='Val', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Phase 1 FAST Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'losses.png'))
        print(f"📊 Plot saved: {Config.OUTPUT_DIR}/losses.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*70)
    print("LUMIMAP PHASE 1 - FAST TRAINING")
    print("="*70)
    
    # Load data
    print("\n📂 Loading metadata...")
    image_df = pd.read_csv(Config.IMAGE_CSV)
    moa_df = pd.read_csv(Config.MOA_CSV)
    print(f"   Total: {len(image_df)} images")
    
    # Filter to Week1 only
    print(f"\n🔍 Filtering to: {Config.WEEKS_TO_USE}")
    week_pattern = '|'.join([f'{w}/' for w in Config.WEEKS_TO_USE])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    print(f"   After filter: {len(image_df)} images")
    
    # Create dataset
    dataset = ContrastiveDataset(image_df, moa_df, Config.DATA_DIR, transform=get_transforms())
    
    # Subsample for speed
    if Config.SUBSAMPLE_FRACTION < 1.0:
        n_samples = int(len(dataset) * Config.SUBSAMPLE_FRACTION)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset = Subset(dataset, indices)
        print(f"\n⚡ Subsampled to {len(dataset)} samples ({Config.SUBSAMPLE_FRACTION*100}%)")
    
    # Split
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    
    # Model
    print(f"\n🧠 Initializing {Config.BACKBONE} model...")
    model = MOAEncoder().to(Config.DEVICE)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    criterion = TripletLoss(margin=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    trainer.train(Config.NUM_EPOCHS)
    
    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print(f"   Model: {Config.OUTPUT_DIR}/phase1_fast_best.pth")
    print(f"   Time saved: ~70% faster than full training!")
    print("="*70)

if __name__ == '__main__':
    main()
