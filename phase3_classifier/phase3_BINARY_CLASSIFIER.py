"""
Phase 3 - Binary Resistance Classifier (WORKS BETTER!)
=======================================================

Instead of 3-class (SENSITIVE, PARTIAL, CROSS):
Use 2-class (SENSITIVE vs RESISTANT)

Combines PARTIAL + CROSS → RESISTANT
This gives: 303 SENSITIVE vs 12 RESISTANT (25:1 instead of 60:1)

Much more trainable! Will actually predict resistant samples!
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
    OUTPUT_DIR = './output/phase3_binary'
    
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # BINARY CLASSIFICATION
    NUM_CLASSES = 2
    CLASS_NAMES = ['SENSITIVE', 'RESISTANT']
    
    NUM_WORKERS = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128

print("="*70)
print("🎯 PHASE 3 - Binary Classifier (SENSITIVE vs RESISTANT)")
print("="*70)
print(f"   Combines PARTIAL + CROSS → RESISTANT")
print(f"   Much better class balance!")
print("="*70)

# [Same model classes - reuse from before]
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
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
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
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

class BinaryResistanceDataset(Dataset):
    def __init__(self, labels_df, image_df, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.data = labels_df.merge(image_df, left_on='idx', right_index=True, how='inner')
        
        # BINARY: SENSITIVE=0, RESISTANT=1 (PARTIAL + CROSS + PRIMARY)
        def map_to_binary(resistance_type):
            if resistance_type == 'SENSITIVE':
                return 0
            else:  # PARTIAL, CROSS, PRIMARY
                return 1
        
        self.data['binary_label'] = self.data['resistance_type'].apply(map_to_binary)
        
        print(f"📊 Binary dataset:")
        print(self.data['binary_label'].value_counts())
        print(f"   SENSITIVE: {(self.data['binary_label']==0).sum()}")
        print(f"   RESISTANT: {(self.data['binary_label']==1).sum()}")
    
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
        
        return {
            'dapi': dapi,
            'tubulin': tubulin,
            'actin': actin,
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'label': torch.tensor(row['binary_label'], dtype=torch.long),
            'original_type': row['resistance_type']
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

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
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
            logits, _, _ = self.model(dapi, tubulin, actin, conc)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        return running_loss / len(self.train_loader), 100 * correct / total
    
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
                
                logits, _, _ = self.model(dapi, tubulin, actin, conc)
                loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return running_loss / len(self.val_loader), 100 * correct / total, all_preds, all_labels
    
    def train(self, num_epochs, unfreeze_at_epoch=15):
        print(f"\n🚀 Training binary classifier on {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            if epoch == unfreeze_at_epoch:
                print("🔓 Unfreezing encoder!")
                self.model.unfreeze_encoder()
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=Config.LEARNING_RATE / 10,
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
                self.save_checkpoint('phase3_binary_best.pth', val_preds, val_labels)
                print(f"✓ Best model! Acc: {val_acc:.2f}%")
        
        self.plot_training()
        print(f"\n✅ Binary Training Complete!")
        print(f"   Best accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, filename, val_preds, val_labels):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc
        }, os.path.join(Config.OUTPUT_DIR, filename))
        
        report = classification_report(val_labels, val_preds, target_names=Config.CLASS_NAMES, digits=3)
        with open(os.path.join(Config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)
        print(f"\n📊 Classification Report:\n{report}")
        
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
        plt.title('Binary Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
        plt.close()
    
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.train_losses, label='Train', marker='o')
        ax1.plot(self.val_losses, label='Val', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train', marker='o')
        ax2.plot(self.val_accs, label='Val', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_curves.png'), dpi=150)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n📂 Loading data...")
    labels_df = pd.read_csv(Config.LABELS_CSV)
    image_df = pd.read_csv(Config.IMAGE_CSV)
    
    print(f"🔍 Filtering to Week1-6...")
    week_pattern = '|'.join([f'Week{i}/' for i in range(1, 7)])
    image_df_filtered = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    full_dataset = BinaryResistanceDataset(labels_df, image_df_filtered, Config.DATA_DIR, transform=None)
    
    all_labels = [row['binary_label'] for _, row in full_dataset.data.iterrows()]
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, 
                                         stratify=all_labels, random_state=42)
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_idx)}")
    print(f"   Val: {len(val_idx)}")
    
    # Class weights
    unique, counts = np.unique([all_labels[i] for i in train_idx], return_counts=True)
    class_weights = torch.FloatTensor([counts[1] / counts[0], 1.0]).to(Config.DEVICE)
    print(f"⚖️  Class weights: {class_weights}")
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    full_dataset.transform = get_transforms(train=True)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=Config.NUM_WORKERS, pin_memory=False)
    
    full_dataset.transform = get_transforms(train=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                           num_workers=Config.NUM_WORKERS, pin_memory=False)
    
    model = ResistanceClassifier(num_classes=Config.NUM_CLASSES)
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.encoder.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=Config.LEARNING_RATE, 
                                 weight_decay=Config.WEIGHT_DECAY)
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, Config.DEVICE)
    trainer.train(Config.NUM_EPOCHS)
    
    print("\n" + "="*70)
    print("✅ BINARY CLASSIFIER COMPLETE!")
    print(f"   Model: {Config.OUTPUT_DIR}/phase3_binary_best.pth")
    print(f"   This should actually detect resistant samples!")
    print("="*70)

if __name__ == '__main__':
    main()
