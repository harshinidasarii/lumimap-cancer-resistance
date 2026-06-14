"""
LUMIMAP Phase 3 FAST - Resistance Classifier Training
======================================================

Optimized for 10-core processor
Expected time: 60-90 minutes
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    LABELS_CSV = './output/phase2_fast/resistance_labels.csv'
    OUTPUT_DIR = './output/phase3_fast'
    
    # Phase 1 model for transfer learning
    PHASE1_MODEL = './output/phase1_fast/phase1_fast_best.pth'
    
    # Model settings (lightweight)
    BACKBONE = 'mobilenet_v2'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128
    NUM_CLASSES = 5  # 5 resistance types
    
    # Training (optimized)
    BATCH_SIZE = 128
    NUM_EPOCHS = 20      # Fewer epochs
    LEARNING_RATE = 2e-4
    NUM_WORKERS = 8
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEEKS_TO_USE = ['Week1']

print(f"🚀 PHASE 3 FAST - Resistance Classifier")
print(f"   Epochs: {Config.NUM_EPOCHS}")
print(f"   Batch size: {Config.BATCH_SIZE}")
print(f"   Workers: {Config.NUM_WORKERS}")

# ============================================================================
# DATASET
# ============================================================================

class ResistanceDataset(Dataset):
    def __init__(self, labels_df, image_df, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Merge labels with image paths
        image_sample = image_df.groupby(
            ['Image_Metadata_Compound', 'Image_Metadata_Concentration']
        ).first().reset_index()
        
        self.data = labels_df.merge(
            image_sample,
            left_on=['compound', 'concentration'],
            right_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
            how='inner'
        )
        
        print(f"📊 Dataset: {len(self.data)} samples")
        print(f"   Resistance types: {self.data['resistance_type'].value_counts().to_dict()}")
    
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
            'dapi': dapi,
            'tubulin': tubulin,
            'actin': actin,
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'label': torch.tensor(row['resistance_code'], dtype=torch.long),
            'moa': row['moa'],
            'compound': row['compound']
        }
    
    def _normalize_channel(self, channel):
        channel = channel.astype(np.float32)
        c_min, c_max = channel.min(), channel.max()
        if c_max > c_min:
            channel = 255 * (channel - c_min) / (c_max - c_min)
        return channel.astype(np.uint8)
    
    def _apply_transform(self, channel):
        channel_3ch = np.stack([channel, channel, channel], axis=-1)
        if self.transform:
            augmented = self.transform(image=channel_3ch)
            return augmented['image'][0:1]

# ============================================================================
# MODEL (from Phase 1)
# ============================================================================

class ChannelEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
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
    """Resistance classifier built on Phase 1 encoder"""
    def __init__(self, pretrained_encoder=None):
        super().__init__()
        
        if pretrained_encoder:
            self.encoder = pretrained_encoder
            # Freeze encoder initially
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder = MOAEncoder()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, Config.NUM_CLASSES)
        )
    
    def forward(self, dapi, tubulin, actin, concentration):
        features, attention = self.encoder(dapi, tubulin, actin, concentration)
        logits = self.classifier(features)
        return logits, attention
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transforms():
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = Config.DEVICE
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            dapi = batch['dapi'].to(self.device)
            tubulin = batch['tubulin'].to(self.device)
            actin = batch['actin'].to(self.device)
            conc = batch['concentration'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(dapi, tubulin, actin, conc)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                dapi = batch['dapi'].to(self.device)
                tubulin = batch['tubulin'].to(self.device)
                actin = batch['actin'].to(self.device)
                conc = batch['concentration'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(dapi, tubulin, actin, conc)
                loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs, unfreeze_at=10):
        print(f"\n🚀 Training on {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Unfreeze encoder
            if epoch == unfreeze_at:
                print("🔓 Unfreezing encoder")
                self.model.unfreeze_encoder()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('phase3_best_model.pth')
                print(f"✓ Best model! Acc: {val_acc:.4f}")
        
        self.plot_training()
        self.generate_report(val_preds, val_labels)
        print(f"\n✅ Training Complete! Best: {self.best_val_acc:.4f}")
    
    def save_checkpoint(self, filename):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc
        }, os.path.join(Config.OUTPUT_DIR, filename))
    
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='Train', marker='o')
        ax1.plot(self.val_losses, label='Val', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Curves')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train', marker='o')
        ax2.plot(self.val_accs, label='Val', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training.png'))
        print(f"📊 Plot saved")
    
    def generate_report(self, preds, labels):
        resistance_types = ['SENSITIVE', 'PRIMARY_RESISTANCE', 'PARTIAL_RESISTANCE', 'CROSS_RESISTANCE', 'UNCERTAIN']
        
        # Filter to only classes present
        unique = sorted(set(labels + preds))
        names = [resistance_types[i] for i in unique]
        
        report = classification_report(labels, preds, target_names=names, digits=3)
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(report)
        
        with open(os.path.join(Config.OUTPUT_DIR, 'report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=names, yticklabels=names, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'))
        print(f"📊 Confusion matrix saved")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("LUMIMAP PHASE 3 FAST - Resistance Classifier Training")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(Config.IMAGE_CSV)
    
    # Filter
    week_pattern = '|'.join([f'{w}/' for w in Config.WEEKS_TO_USE])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    # Remove UNCERTAIN
    print(f"   Total: {len(labels_df)}")
    labels_df = labels_df[labels_df['resistance_type'] != 'UNCERTAIN']
    print(f"   After removing UNCERTAIN: {len(labels_df)}")
    
    # Create dataset
    dataset = ResistanceDataset(labels_df, image_df, Config.DATA_DIR)
    
    # Split
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=dataset.data['resistance_code']
    )
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    
    # Load Phase 1 encoder
    print(f"\n🧠 Loading Phase 1 encoder...")
    pretrained_encoder = MOAEncoder()
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    pretrained_encoder.load_state_dict(checkpoint['model_state_dict'])
    print("   ✓ Loaded")
    
    # Create classifier
    model = ResistanceClassifier(pretrained_encoder).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    trainer.train(Config.NUM_EPOCHS, unfreeze_at=10)
    
    print("\n" + "="*70)
    print("✅ PHASE 3 COMPLETE!")
    print(f"   Model: {Config.OUTPUT_DIR}/phase3_best_model.pth")
    print(f"   Accuracy: {trainer.best_val_acc:.1%}")
    print("="*70)

if __name__ == '__main__':
    main()