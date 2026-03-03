"""
LUMIMAP: Multi-Type Resistance Classifier
Trains a model to classify 7 types of cancer drug resistance based on
morphological features from microscopy images (DAPI, Tubulin, Actin channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from pathlib import Path
from typing import Tuple, Dict, List

from therapy_recommendation_database import ResistanceType, TherapyDatabase

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_DIR = './data'
    IMAGE_CSV = 'BBBC021_v1_image.csv'
    MOA_CSV = 'BBBC021_v1_moa.csv'
    OUTPUT_DIR = './outputs_resistance'
    
    # Model parameters
    BACKBONE = 'resnet50'
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Resistance types (8 classes: 7 resistance + 1 sensitive)
    NUM_RESISTANCE_TYPES = 8
    
    # Training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 15
    
    # Feature extraction
    USE_CHANNEL_ATTENTION = True
    USE_CONCENTRATION_ENCODING = True

# ============================================================================
# CHANNEL-AWARE DATASET
# ============================================================================

class ResistanceDataset(Dataset):
    """
    Dataset that loads separate channels and generates pseudo-labels
    for resistance types based on morphological features
    """
    
    def __init__(self, image_df, moa_df, root_dir, transform=None, 
                 generate_labels=True):
        """
        Args:
            image_df: DataFrame with image file paths
            moa_df: DataFrame with compound MOA labels
            root_dir: Root directory containing images
            transform: Albumentations transforms
            generate_labels: If True, generate pseudo-labels for resistance
        """
        self.root_dir = Path(root_dir)
        
        # Merge image metadata with MOA labels
        self.data = image_df.merge(
            moa_df, 
            left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
            right_on=['compound', 'concentration'],
            how='inner'
        )
        
        self.transform = transform
        self.generate_labels = generate_labels
        
        # For now, we'll use a simple heuristic to generate pseudo-labels
        # In production, this would be replaced with the contrastive learning
        # approach we discussed
        if self.generate_labels:
            self._generate_pseudo_labels()
        
        print(f"Dataset loaded: {len(self.data)} images")
        if 'resistance_type' in self.data.columns:
            print(f"Resistance type distribution:\n{self.data['resistance_type'].value_counts()}")
    
    def _generate_pseudo_labels(self):
        """
        Generate pseudo-labels for resistance types
        
        This is a PLACEHOLDER. In production, this would be replaced with:
        1. Contrastive learning to learn MOA phenotypes
        2. Comparison to DMSO baseline
        3. Feature-based classification
        
        For now, we'll use a simple rule-based approach:
        - DMSO → SENSITIVE
        - Others → Random assignment based on MOA (for demonstration)
        """
        
        def assign_resistance_type(row):
            # DMSO is always sensitive
            if row['moa'] == 'DMSO':
                return ResistanceType.SENSITIVE.value
            
            # Simple heuristic based on MOA
            # This is TEMPORARY - replace with actual feature-based classification
            moa_to_resistance = {
                'Actin disruptors': ResistanceType.EMT_LIKE.value,
                'Aurora kinase inhibitors': ResistanceType.TARGET_THERAPY.value,
                'Microtubule stabilizers': ResistanceType.DRUG_EFFLUX.value,
                'Microtubule destabilizers': ResistanceType.DRUG_EFFLUX.value,
                'DNA damage': ResistanceType.APOPTOSIS_RESISTANCE.value,
                'DNA replication': ResistanceType.APOPTOSIS_RESISTANCE.value,
                'Protein synthesis': ResistanceType.METABOLIC_REWIRING.value,
                'Protein degradation': ResistanceType.METABOLIC_REWIRING.value,
                'Kinase inhibitors': ResistanceType.TARGET_THERAPY.value,
                'Eg5 inhibitors': ResistanceType.TARGET_THERAPY.value,
                'Epithelial': ResistanceType.ENDOCRINE_HORMONE.value,
                'Cholesterol-lowering': ResistanceType.METABOLIC_REWIRING.value,
            }
            
            return moa_to_resistance.get(row['moa'], ResistanceType.SENSITIVE.value)
        
        self.data['resistance_type'] = self.data.apply(assign_resistance_type, axis=1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load 3 channels SEPARATELY (not stacked as RGB)
        dapi_path = self.root_dir / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
        tubulin_path = self.root_dir / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
        actin_path = self.root_dir / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
        
        # Read images (grayscale)
        dapi = np.array(Image.open(dapi_path))
        tubulin = np.array(Image.open(tubulin_path))
        actin = np.array(Image.open(actin_path))
        
        # Normalize each channel
        dapi = self._normalize_channel(dapi)
        tubulin = self._normalize_channel(tubulin)
        actin = self._normalize_channel(actin)
        
        # Stack for transform (but we'll process separately later)
        image = np.stack([dapi, tubulin, actin], axis=-1)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get compound and concentration info
        compound = row['Image_Metadata_Compound']
        concentration = float(row['Image_Metadata_Concentration'])
        
        # Get resistance label
        if 'resistance_type' in row:
            label = int(row['resistance_type'])
        else:
            label = ResistanceType.SENSITIVE.value
        
        return {
            'image': image,
            'compound': compound,
            'concentration': torch.tensor([concentration], dtype=torch.float32),
            'moa': row['moa'],
            'label': label
        }
    
    def _normalize_channel(self, channel):
        """Normalize single channel to 0-255 range"""
        channel = channel.astype(np.float32)
        channel_min = channel.min()
        channel_max = channel.max()
        if channel_max > channel_min:
            channel = 255 * (channel - channel_min) / (channel_max - channel_min)
        return channel.astype(np.uint8)

# ============================================================================
# CHANNEL-AWARE MODEL ARCHITECTURE
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Attention module to weight the importance of each channel
    based on the drug's MOA
    """
    def __init__(self, num_channels=3, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAwareResistanceClassifier(nn.Module):
    """
    Multi-channel resistance classifier with:
    1. Separate encoders for each channel (DAPI, Tubulin, Actin)
    2. Channel attention mechanism
    3. Concentration encoding
    4. Multi-class resistance type classification
    """
    
    def __init__(self, backbone='resnet50', num_classes=8, 
                 use_attention=True, use_concentration=True):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_concentration = use_concentration
        
        # Shared backbone (pretrained on ImageNet)
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer from base model
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Channel attention
        if use_attention:
            self.channel_attention = ChannelAttention(num_channels=3)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Concentration encoder
        if use_concentration:
            self.conc_encoder = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 128),
                nn.ReLU()
            )
            fusion_dim = self.feature_dim + 128
        else:
            fusion_dim = self.feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Feature extraction hook for GradCAM
        self.feature_maps = None
        self.backbone.register_forward_hook(self._save_features)
    
    def _save_features(self, module, input, output):
        """Hook to save feature maps for GradCAM"""
        self.feature_maps = output
    
    def forward(self, x, concentration=None):
        """
        Args:
            x: Input image tensor (B, 3, H, W) - channels are DAPI, Tubulin, Actin
            concentration: Drug concentration (B, 1)
        """
        # Apply channel attention
        if self.use_attention:
            x = self.channel_attention(x)
        
        # Extract features
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Encode concentration
        if self.use_concentration and concentration is not None:
            conc_features = self.conc_encoder(concentration)
            features = torch.cat([features, conc_features], dim=1)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_channel_features(self, x):
        """
        Extract features from each channel separately
        Useful for interpretation and GradCAM
        """
        with torch.no_grad():
            # Process each channel
            dapi = x[:, 0:1, :, :].repeat(1, 3, 1, 1)
            tubulin = x[:, 1:2, :, :].repeat(1, 3, 1, 1)
            actin = x[:, 2:3, :, :].repeat(1, 3, 1, 1)
            
            dapi_features = self.backbone(dapi)
            tubulin_features = self.backbone(tubulin)
            actin_features = self.backbone(actin)
            
            return {
                'dapi': dapi_features,
                'tubulin': tubulin_features,
                'actin': actin_features
            }

# ============================================================================
# TRAINING
# ============================================================================

def get_train_transforms():
    """Augmentations for training"""
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, 
                          rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                   contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    """Transforms for validation"""
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, 
                 optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = Config.DEVICE
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            concentrations = batch['concentration'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images, concentrations)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_compounds = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                concentrations = batch['concentration'].to(self.device)
                
                outputs = self.model(images, concentrations)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_compounds.extend(batch['compound'])
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc, all_preds, all_labels, all_compounds
    
    def train(self, num_epochs):
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels, val_compounds = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint('best_resistance_model.pth')
                print(f"✓ New best model saved! Accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        print(f"\nBest validation accuracy: {self.best_val_acc:.4f}")
        
        # Generate reports
        self.plot_training_curves()
        self.generate_report(val_preds, val_labels, val_compounds)
    
    def save_checkpoint(self, filename):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, os.path.join(Config.OUTPUT_DIR, filename))
    
    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_curves.png'))
        print(f"Training curves saved")
    
    def generate_report(self, preds, labels, compounds):
        # Classification report
        resistance_names = [r.name for r in ResistanceType]
        report = classification_report(labels, preds, 
                                      target_names=resistance_names, 
                                      digits=3)
        print("\nClassification Report:")
        print(report)
        
        with open(os.path.join(Config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=resistance_names,
                   yticklabels=resistance_names,
                   cmap='Blues')
        plt.title('Resistance Type Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'))
        print("Confusion matrix saved")

# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading BBBC021 metadata...")
    image_df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.IMAGE_CSV))
    moa_df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.MOA_CSV))
    
    # Filter to weeks 1-3
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(
        'Week1/Week1_|Week2/Week2_|Week3/Week3_')]
    print(f"Using {len(image_df)} images from Weeks 1-3")
    
    # Create dataset
    full_dataset = ResistanceDataset(image_df, moa_df, Config.DATA_DIR)
    
    # Split
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[full_dataset.data.iloc[i]['resistance_type'] 
                 for i in range(len(full_dataset))]
    )
    
    # Create loaders
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=True, num_workers=Config.NUM_WORKERS,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                           shuffle=False, num_workers=Config.NUM_WORKERS,
                           pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Initialize model
    print(f"\nInitializing channel-aware resistance classifier...")
    model = ChannelAwareResistanceClassifier(
        backbone=Config.BACKBONE,
        num_classes=Config.NUM_RESISTANCE_TYPES,
        use_attention=Config.USE_CHANNEL_ATTENTION,
        use_concentration=Config.USE_CONCENTRATION_ENCODING
    ).to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=Config.LEARNING_RATE,
                                 weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, 
                     criterion, optimizer, scheduler)
    trainer.train(Config.NUM_EPOCHS)
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best accuracy: {trainer.best_val_acc:.4f}")
    print(f"Model saved to: {Config.OUTPUT_DIR}/best_resistance_model.pth")
    print("="*70)

if __name__ == '__main__':
    main()
