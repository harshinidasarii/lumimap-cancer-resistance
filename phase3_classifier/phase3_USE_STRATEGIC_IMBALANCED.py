"""
Phase 3 - Resistance Classifier (Handles Class Imbalance)
==========================================================

Uses strategic model and handles imbalanced labels:
- SENSITIVE: 2457 samples (95.6%)
- PARTIAL_RESISTANCE: 64 samples (2.5%)
- CROSS_RESISTANCE: 46 samples (1.8%)
- PRIMARY_RESISTANCE: 1 sample (REMOVED - can't train on 1 sample)

Strategy:
1. Remove PRIMARY_RESISTANCE (only 1 sample)
2. Train on 3 classes: SENSITIVE, PARTIAL, CROSS
3. Use class weights to handle imbalance
4. Use stratified sampling for train/val split
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
import matplotlib.pyplot as plt
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    LABELS_CSV = './output/phase2_strategic/resistance_labels.csv'
    OUTPUT_DIR = './output/phase3_strategic'
    
    # USE STRATEGIC MODEL
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    
    # Training
    BATCH_SIZE = 32  # Smaller for imbalanced data
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 3 CLASSES (removed PRIMARY_RESISTANCE)
    NUM_CLASSES = 3
    CLASS_NAMES = ['SENSITIVE', 'PARTIAL_RESISTANCE', 'CROSS_RESISTANCE']
    
    NUM_WORKERS = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128

print("="*70)
print("🚀 PHASE 3 - Strategic Model + Class Imbalance Handling")
print("="*70)
print(f"   Classes: {Config.NUM_CLASSES} (removed PRIMARY_RESISTANCE)")
print(f"   Model: {Config.PHASE1_MODEL}")
print(f"   Strategy: Class weights + stratified sampling")
print("="*70)

# [Model classes from Phase 1]
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
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = MOAEncoder()
        
        # Freeze encoder initially
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, dapi, tubulin, actin, concentration):
        embeddings, attention = self.encoder(dapi, tubulin, actin, concentration)
        logits = self.classifier(embeddings)
        return logits, attention
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

class ResistanceDataset(Dataset):
    def __init__(self, labels_df, image_df, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Merge labels with image paths
        self.data = labels_df.merge(
            image_df,
            left_on='idx',
            right_index=True,
            how='inner'
        )
        
        # Map resistance types to class indices
        # Remove PRIMARY_RESISTANCE (only 1 sample)
        self.class_map = {
            'SENSITIVE': 0,
            'PARTIAL_RESISTANCE': 1,
            'CROSS_RESISTANCE': 2
        }
        
        # Filter out PRIMARY_RESISTANCE and UNCERTAIN
        self.data = self.data[self.data['resistance_type'].isin(self.class_map.keys())]
        
        print(f"📊 Dataset after filtering:")
        print(self.data['resistance_type'].value_counts())
        print(f"   Total: {len(self.data)} samples")
    
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
        
        if self.transform:
            dapi = self._apply_transform(dapi)
            tubulin = self._apply_transform(tubulin)
            actin = self._apply_transform(actin)
        
        label = self.class_map[row['resistance_type']]
        
        return {
            'dapi': dapi,
            'tubulin': tubulin,
            'actin': actin,
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
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

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(*Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(*Config.IMG_SIZE),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])

def compute_class_weights(labels):
    """Compute class weights for imbalanced data"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, class_weights):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_weights = class_weights
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
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
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs, unfreeze_at_epoch=10):
        print(f"\n🚀 Training on {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Unfreeze encoder halfway through
            if epoch == unfreeze_at_epoch:
                print("🔓 Unfreezing encoder!")
                self.model.unfreeze_encoder()
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=Config.LEARNING_RATE / 10,  # Lower LR for fine-tuning
                    weight_decay=Config.WEIGHT_DECAY
                )
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('phase3_best_model.pth', val_preds, val_labels)
                print(f"✓ Best model! Acc: {val_acc:.2f}%")
        
        self.plot_training()
        print(f"\n✅ Training Complete!")
        print(f"   Best accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, filename, val_preds, val_labels):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc
        }, os.path.join(Config.OUTPUT_DIR, filename))
        
        # Save classification report
        report = classification_report(
            val_labels, val_preds,
            target_names=Config.CLASS_NAMES,
            digits=3
        )
        with open(os.path.join(Config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Save confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=Config.CLASS_NAMES,
                   yticklabels=Config.CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
        plt.close()
    
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='Train', marker='o')
        ax1.plot(self.val_losses, label='Val', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train', marker='o')
        ax2.plot(self.val_accs, label='Val', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_curves.png'), dpi=150)
        print(f"📊 Plots saved")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n📂 Loading data...")
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(Config.IMAGE_CSV)
    
    # CRITICAL: Filter to same weeks as Phase 2!
    print(f"🔍 Filtering images to Week1-6 (matching Phase 2)...")
    week_pattern = '|'.join([f'Week{i}/' for i in range(1, 7)])
    image_df_filtered = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    print(f"   Images before: {len(image_df)}")
    print(f"   Images after: {len(image_df_filtered)}")
    
    # Create dataset
    full_dataset = ResistanceDataset(labels_df, image_df_filtered, Config.DATA_DIR, transform=None)
    
    # Get labels for stratified split
    all_labels = [full_dataset.class_map[row['resistance_type']] 
                  for _, row in full_dataset.data.iterrows()]
    
    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_idx)} samples")
    print(f"   Val: {len(val_idx)} samples")
    
    # Compute class weights
    train_labels = [all_labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels)
    print(f"\n⚖️  Class weights: {class_weights}")
    class_weights = class_weights.to(Config.DEVICE)
    
    # Create datasets with transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Apply transforms
    full_dataset.transform = get_transforms(train=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    full_dataset.transform = get_transforms(train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    print(f"\n🧠 Building model...")
    model = ResistanceClassifier(num_classes=Config.NUM_CLASSES)
    
    # Load Phase 1 weights
    print(f"   Loading Phase 1 weights from {Config.PHASE1_MODEL}")
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.encoder.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(Config.DEVICE)
    print(f"   ✓ Model on {Config.DEVICE}")
    
    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),  # Only train classifier initially
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, Config.DEVICE, class_weights)
    trainer.train(Config.NUM_EPOCHS, unfreeze_at_epoch=10)
    
    print("\n" + "="*70)
    print("✅ PHASE 3 COMPLETE!")
    print(f"   Model: {Config.OUTPUT_DIR}/phase3_best_model.pth")
    print(f"   Best accuracy: {trainer.best_val_acc:.2f}%")
    print("="*70)

if __name__ == '__main__':
    main()