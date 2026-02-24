"""
Cancer Resistance Detection System - Starter Code
Phase 1: MOA Classification using BBBC021

This script trains a CNN to classify Mechanism of Action (MOA) from microscopy images.
This is the foundation for resistance detection.
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths (modify these for your setup)
    DATA_DIR = './data'  # ← CHANGE TO THIS!
    IMAGE_CSV = 'BBBC021_v1_image.csv'
    MOA_CSV = 'BBBC021_v1_moa.csv'
    OUTPUT_DIR = './outputs' 
    
    # Model parameters
    BACKBONE = 'resnet50'  # or 'efficientnet_b0', 'vit_b_16'
    IMG_SIZE = (512, 512)  # Resize images to this size
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # MOA Classes (from BBBC021)
    MOA_CLASSES = [
        'Actin disruptors',
        'Aurora kinase inhibitors',
        'Cholesterol-lowering',
        'DMSO',
        'DNA damage',
        'DNA replication',
        'Eg5 inhibitors',
        'Epithelial',
        'Kinase inhibitors',
        'Microtubule destabilizers',
        'Microtubule stabilizers',
        'Protein degradation',
        'Protein synthesis'
    ]
    
    NUM_CLASSES = len(MOA_CLASSES)  # Now 13 instead of 12
    NUM_CLASSES = len(MOA_CLASSES)
    
    # Training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 10

# ============================================================================
# DATASET
# ============================================================================

class BBBC021Dataset(Dataset):
    """
    Dataset for BBBC021 microscopy images
    Loads 3-channel images: DAPI (nucleus), Tubulin, Actin
    """
    
    def __init__(self, image_df, moa_df, root_dir, transform=None):
        """
        Args:
            image_df: DataFrame with image file paths
            moa_df: DataFrame with compound MOA labels
            root_dir: Root directory containing images
            transform: Albumentations transforms
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
        
        # Create MOA to index mapping
        self.moa_to_idx = {moa: idx for idx, moa in enumerate(Config.MOA_CLASSES)}
        
        print(f"Dataset loaded: {len(self.data)} images")
        print(f"MOA distribution:\n{self.data['moa'].value_counts()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
    
    # Load 3 channels
    # Fix path: CSV has "Week1/Week1_22123" but folder is just "Week1_22123"
        dapi_path = self.root_dir / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
        tubulin_path = self.root_dir / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
        actin_path = self.root_dir / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
        
        # Read images (grayscale)
        dapi = np.array(Image.open(dapi_path))
        tubulin = np.array(Image.open(tubulin_path))
        actin = np.array(Image.open(actin_path))
        
        # Stack channels to create RGB-like image
        # (H, W, 3) where channels are DAPI, Tubulin, Actin
        image = np.stack([dapi, tubulin, actin], axis=-1)
        
        # Normalize to 0-255 range (if needed)
        image = self._normalize_image(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get label
        moa = row['moa']
        label = self.moa_to_idx[moa]
        
        return image, label
    
    def _normalize_image(self, image):
        """Normalize image to 0-255 range"""
        image = image.astype(np.float32)
        # Per-channel normalization
        for c in range(3):
            channel = image[:, :, c]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                image[:, :, c] = 255 * (channel - channel_min) / (channel_max - channel_min)
        return image.astype(np.uint8)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_train_transforms():
    """Augmentations for training"""
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=0.1, 
            rotate_limit=15, 
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms():
    """Transforms for validation/test"""
    return A.Compose([
        A.Resize(*Config.IMG_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# ============================================================================
# MODEL
# ============================================================================

class MOAClassifier(nn.Module):
    """
    CNN model for MOA classification
    Uses pre-trained backbone (ResNet, EfficientNet, etc.)
    """
    
    def __init__(self, backbone='resnet50', num_classes=12, pretrained=True):
        super().__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        elif backbone == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features for visualization/analysis"""
        with torch.no_grad():
            features = self.backbone(x)
        return features

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
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Metrics tracking
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
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
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
            val_loss, val_acc, val_preds, val_labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"✓ New best model saved! Accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Generate final classification report
        self.generate_report(val_preds, val_labels)
    
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
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_curves.png'))
        print(f"Training curves saved to {Config.OUTPUT_DIR}/training_curves.png")
    
    def generate_report(self, preds, labels):
        # Get only the classes that actually exist in the data
        # labels and preds are already lists, not tensors
        unique_classes = sorted(set(labels + preds))
        actual_moa_names = [Config.MOA_CLASSES[i] for i in unique_classes]

        report = classification_report(
            labels, preds,
            target_names=actual_moa_names,  # Only the 5 that exist
            digits=3
        )
        print("\nClassification Report:")
        print(report)
        
        # Save to file
        with open(os.path.join(Config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            xticklabels=Config.MOA_CLASSES,
            yticklabels=Config.MOA_CLASSES,
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to {Config.OUTPUT_DIR}/confusion_matrix.png")

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load metadata
# Load metadata
    print("Loading BBBC021 metadata...")
    image_df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.IMAGE_CSV))
    moa_df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.MOA_CSV))

    # FILTER TO ONLY WEEK 1 (since we only downloaded Week 1)
    print("Filtering to Week 1 data only...")
    # Use all Week1, Week2, and Week3 data
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains('Week1/Week1_|Week2/Week2_|Week3/Week3_')]
    print(f"After filtering: {len(image_df)} images from Week 1")

    # Create dataset
    full_dataset = BBBC021Dataset(
        image_df, 
        moa_df, 
        Config.DATA_DIR,
        transform=None
    )
    
    # Split dataset (80% train, 20% val)
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[full_dataset.data.iloc[i]['moa'] for i in range(len(full_dataset))]
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print(f"\nInitializing {Config.BACKBONE} model...")
    model = MOAClassifier(
        backbone=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=True
    )
    model = model.to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Create trainer and start training
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler
    )
    
    trainer.train(Config.NUM_EPOCHS)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Model saved to: {Config.OUTPUT_DIR}/best_model.pth")
    print("="*50)

if __name__ == '__main__':
    main()
