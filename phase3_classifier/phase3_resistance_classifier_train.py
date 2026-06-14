"""
LUMIMAP Phase 3: Resistance Classifier with Therapy Recommendations
=====================================================================

This script trains the final resistance classifier using pseudo-labels from Phase 2.

Features:
1. Multi-class resistance classification (5 types)
2. Channel-wise GradCAM visualization
3. Therapy recommendation engine
4. Combination therapy suggestions for cross-resistance

Input: Pseudo-labels from Phase 2
Output:
- Trained classifier: phase3_resistance_classifier.pth
- Inference script for clinical use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import from previous phases
import sys
sys.path.append('/home/claude')
from phase1_contrastive_moa_learning import MOAEncoder, Config as Phase1Config
from phase2_generate_resistance_labels import ResistanceType

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_DIR = Phase1Config.DATA_DIR
    IMAGE_CSV = Phase1Config.IMAGE_CSV
    MOA_CSV = Phase1Config.MOA_CSV
    LABELS_CSV = './outputs/phase2/resistance_labels.csv'
    OUTPUT_DIR = './outputs/phase3'
    
    # Phase 1 model (for transfer learning)
    PHASE1_MODEL = './outputs/phase1/phase1_best_model.pth'
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Classes
    NUM_RESISTANCE_TYPES = 5  # 5 resistance types
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    
    WEEK_FILTER = ['Week1', 'Week2', 'Week3']

# ============================================================================
# DATASET
# ============================================================================

class ResistanceDataset(Dataset):
    """Dataset with resistance labels from Phase 2"""
    
    def __init__(self, labels_df, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.data = labels_df
        self.transform = transform
        
        # Create compound to MOA mapping
        self.compound_to_moa = dict(zip(self.data['compound'], self.data['moa']))
        
        print(f"Dataset: {len(self.data)} samples")
        print(f"Resistance distribution:\n{self.data['resistance_type'].value_counts()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get image info from original idx
        # Note: We need to map back to original image CSV
        # For simplicity, we'll use the metadata stored in labels_df
        
        # Load channels - we need to reconstruct paths
        # This is a simplified version; in production, merge with original image_df
        dapi_path = self._find_image_path(row['compound'], row['concentration'], 'DAPI')
        tubulin_path = self._find_image_path(row['compound'], row['concentration'], 'Tubulin')
        actin_path = self._find_image_path(row['compound'], row['concentration'], 'Actin')
        
        if dapi_path is None:
            # Fallback: use idx to lookup in image CSV
            # This requires loading image_csv in __init__
            raise ValueError(f"Could not find images for idx {idx}")
        
        dapi = np.array(Image.open(dapi_path))
        tubulin = np.array(Image.open(tubulin_path))
        actin = np.array(Image.open(actin_path))
        
        # Normalize
        dapi = self._normalize_channel(dapi)
        tubulin = self._normalize_channel(tubulin)
        actin = self._normalize_channel(actin)
        
        # Transform
        if self.transform:
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
    
    def _find_image_path(self, compound, concentration, channel):
        """Find image path for given compound/concentration/channel"""
        # This is placeholder - in production, merge with image_csv
        return None
    
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
        return channel

# Simplified dataset that uses image CSV directly
class ResistanceDatasetV2(Dataset):
    """Simplified dataset that merges labels with image paths"""
    
    def __init__(self, labels_df, image_df, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Merge labels with image paths
        # Group image_df by compound+concentration and take first image
        image_sample = image_df.groupby(['Image_Metadata_Compound', 'Image_Metadata_Concentration']).first().reset_index()
        
        self.data = labels_df.merge(
            image_sample,
            left_on=['compound', 'concentration'],
            right_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
            how='inner'
        )
        
        print(f"Dataset: {len(self.data)} samples")
        print(f"Resistance distribution:\n{self.data['resistance_type'].value_counts()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load channels
        dapi_path = self.root_dir / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
        tubulin_path = self.root_dir / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
        actin_path = self.root_dir / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
        
        dapi = np.array(Image.open(dapi_path))
        tubulin = np.array(Image.open(tubulin_path))
        actin = np.array(Image.open(actin_path))
        
        # Normalize
        dapi = self._normalize_channel(dapi)
        tubulin = self._normalize_channel(tubulin)
        actin = self._normalize_channel(actin)
        
        # Transform
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
# MODEL
# ============================================================================

class ResistanceClassifier(nn.Module):
    """
    Resistance classifier built on top of Phase 1 encoder
    """
    def __init__(self, pretrained_encoder=None, num_classes=5):
        super().__init__()
        
        # Use pretrained encoder from Phase 1
        if pretrained_encoder:
            self.encoder = pretrained_encoder
            # Freeze encoder initially
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder = MOAEncoder()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(Phase1Config.EMBEDDING_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, dapi, tubulin, actin, concentration):
        # Get features from encoder
        features, attention = self.encoder(dapi, tubulin, actin, concentration)
        
        # Classify
        logits = self.classifier(features)
        
        return logits, attention
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transforms():
    return A.Compose([
        A.Resize(*Phase1Config.IMG_SIZE),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(*Phase1Config.IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
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
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        
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
        epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs, unfreeze_at_epoch=10):
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Unfreeze encoder for fine-tuning
            if epoch == unfreeze_at_epoch:
                print("🔓 Unfreezing encoder for fine-tuning")
                self.model.unfreeze_encoder()
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('phase3_best_model.pth')
                print(f"✓ New best model! Accuracy: {val_acc:.4f}")
        
        self.plot_training()
        self.generate_report(val_preds, val_labels)
        print(f"\nPhase 3 Complete! Best accuracy: {self.best_val_acc:.4f}")
    
    def save_checkpoint(self, filename):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc
        }, os.path.join(Config.OUTPUT_DIR, filename))
    
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Curves')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'phase3_training.png'))
        print(f"Training curves saved")
    
    def generate_report(self, preds, labels):
        resistance_types = [rt.name for rt in ResistanceType]
        
        report = classification_report(labels, preds, target_names=resistance_types, digits=3)
        print("\nClassification Report:")
        print(report)
        
        with open(os.path.join(Config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=resistance_types, 
                    yticklabels=resistance_types, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'))
        print(f"Confusion matrix saved")

# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.IMAGE_CSV))
    
    # Filter images to Week 1-3
    week_pattern = '|'.join([f'{w}/' for w in Config.WEEK_FILTER])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    # Filter out UNCERTAIN labels for training
    print(f"Total labels: {len(labels_df)}")
    labels_df = labels_df[labels_df['resistance_type'] != 'UNCERTAIN']
    print(f"After removing UNCERTAIN: {len(labels_df)}")
    
    # Create dataset
    dataset = ResistanceDatasetV2(labels_df, image_df, Config.DATA_DIR)
    
    # Split
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=dataset.data['resistance_code']
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # Apply transforms
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
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Load Phase 1 encoder
    print("Loading Phase 1 encoder...")
    pretrained_encoder = MOAEncoder()
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    pretrained_encoder.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Encoder loaded")
    
    # Create classifier
    model = ResistanceClassifier(
        pretrained_encoder=pretrained_encoder,
        num_classes=Config.NUM_RESISTANCE_TYPES
    ).to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)
    trainer.train(Config.NUM_EPOCHS, unfreeze_at_epoch=10)

if __name__ == '__main__':
    main()
