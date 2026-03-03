"""
LUMIMAP Phase 2: Multi-Type Resistance Label Generation
========================================================

This script generates resistance labels using the trained MOA encoder from Phase 1.

Resistance Types Identified:
1. SENSITIVE: Shows expected MOA phenotype
2. PRIMARY_RESISTANCE: Looks like DMSO (no drug effect)
3. PARTIAL_RESISTANCE: Weak MOA phenotype
4. CROSS_RESISTANCE: Shows DIFFERENT MOA (bypass pathway activated)

Algorithm:
- For each drug-treated cell:
  1. Extract features using Phase 1 encoder
  2. Compare similarity to DMSO reference (healthy baseline)
  3. Compare similarity to same-MOA reference (expected phenotype)
  4. Compare similarity to all other MOAs (cross-resistance detection)
  5. Classify resistance type based on similarities

Output:
- CSV file with resistance labels: resistance_labels.csv
- Visualization of resistance distribution
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
from enum import Enum
from collections import defaultdict

# Import Phase 1 model
import sys
sys.path.append('/home/claude')
from phase1_contrastive_moa_learning import MOAEncoder, Config as Phase1Config

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Inherit from Phase 1
    DATA_DIR = Phase1Config.DATA_DIR
    IMAGE_CSV = Phase1Config.IMAGE_CSV
    MOA_CSV = Phase1Config.MOA_CSV
    OUTPUT_DIR = './outputs/phase2'
    
    # Phase 1 model
    PHASE1_MODEL = './outputs/phase1/phase1_best_model.pth'
    
    # Similarity thresholds for classification
    DMSO_HIGH_SIMILARITY = 0.70  # High similarity to DMSO → resistance
    MOA_HIGH_SIMILARITY = 0.70   # High similarity to MOA → sensitive
    MOA_MEDIUM_SIMILARITY = 0.50  # Medium similarity → partial resistance
    CROSS_MOA_THRESHOLD = 0.60   # High similarity to DIFFERENT MOA
    
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    WEEK_FILTER = ['Week1', 'Week2', 'Week3']

# ============================================================================
# RESISTANCE TYPES
# ============================================================================

class ResistanceType(Enum):
    SENSITIVE = 0
    PRIMARY_RESISTANCE = 1
    PARTIAL_RESISTANCE = 2
    CROSS_RESISTANCE = 3
    UNCERTAIN = 4

# ============================================================================
# DATASET
# ============================================================================

class InferenceDataset(Dataset):
    """Dataset for generating labels - no augmentation"""
    
    def __init__(self, image_df, moa_df, root_dir):
        self.root_dir = Path(root_dir)
        
        self.data = image_df.merge(
            moa_df,
            left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
            right_on=['compound', 'concentration'],
            how='inner'
        )
        
        # Also keep track of samples WITHOUT MOA labels (for unlabeled compounds)
        self.all_data = image_df.copy()
        
        self.transform = A.Compose([
            A.Resize(*Phase1Config.IMG_SIZE),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
        
        print(f"Inference dataset: {len(self.data)} labeled samples")
        print(f"Total images: {len(self.all_data)}")
    
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
        
        # Apply transforms
        dapi = self._apply_transform(dapi)
        tubulin = self._apply_transform(tubulin)
        actin = self._apply_transform(actin)
        
        return {
            'dapi': dapi,
            'tubulin': tubulin,
            'actin': actin,
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'compound': row['compound'],
            'moa': row['moa'],
            'idx': idx
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

# ============================================================================
# LABEL GENERATOR
# ============================================================================

class ResistanceLabelGenerator:
    """Generate resistance labels using trained MOA encoder"""
    
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        # Store MOA reference embeddings
        self.moa_references = {}
        self.dmso_reference = None
        
        # Store all embeddings
        self.all_embeddings = []
        self.all_metadata = []
    
    def extract_embeddings(self):
        """Extract embeddings for all samples"""
        print("Extracting embeddings...")
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                dapi = batch['dapi'].to(self.device)
                tubulin = batch['tubulin'].to(self.device)
                actin = batch['actin'].to(self.device)
                conc = batch['concentration'].to(self.device)
                
                # Get embeddings
                embeddings, attention = self.model(dapi, tubulin, actin, conc)
                
                # Store
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
        print(f"Extracted {len(self.all_embeddings)} embeddings")
    
    def compute_moa_references(self):
        """Compute reference embedding for each MOA"""
        print("Computing MOA references...")
        
        # Group by MOA
        moa_groups = defaultdict(list)
        for i, meta in enumerate(self.all_metadata):
            moa_groups[meta['moa']].append(i)
        
        # Compute mean embedding for each MOA
        for moa, indices in moa_groups.items():
            moa_embeddings = self.all_embeddings[indices]
            reference = moa_embeddings.mean(dim=0)
            self.moa_references[moa] = reference
            
            if moa == 'DMSO':
                self.dmso_reference = reference
                print(f"DMSO reference: {len(indices)} samples")
        
        print(f"Computed references for {len(self.moa_references)} MOAs")
    
    def classify_resistance(self):
        """Classify resistance type for each sample"""
        print("Classifying resistance...")
        
        labels = []
        
        for i, meta in enumerate(tqdm(self.all_metadata)):
            embedding = self.all_embeddings[i]
            moa = meta['moa']
            
            # Skip DMSO (it's the baseline)
            if moa == 'DMSO':
                labels.append({
                    **meta,
                    'resistance_type': ResistanceType.SENSITIVE.name,
                    'resistance_code': ResistanceType.SENSITIVE.value,
                    'dmso_similarity': 1.0,
                    'moa_similarity': 1.0,
                    'cross_moa': None,
                    'cross_moa_similarity': 0.0
                })
                continue
            
            # Compute similarities
            dmso_sim = F.cosine_similarity(
                embedding.unsqueeze(0),
                self.dmso_reference.unsqueeze(0)
            ).item()
            
            moa_sim = F.cosine_similarity(
                embedding.unsqueeze(0),
                self.moa_references[moa].unsqueeze(0)
            ).item()
            
            # Check cross-MOA similarity
            cross_moa_sims = {}
            for other_moa, ref in self.moa_references.items():
                if other_moa != moa and other_moa != 'DMSO':
                    sim = F.cosine_similarity(
                        embedding.unsqueeze(0),
                        ref.unsqueeze(0)
                    ).item()
                    cross_moa_sims[other_moa] = sim
            
            # Find highest cross-MOA similarity
            if cross_moa_sims:
                max_cross_moa = max(cross_moa_sims, key=cross_moa_sims.get)
                max_cross_sim = cross_moa_sims[max_cross_moa]
            else:
                max_cross_moa = None
                max_cross_sim = 0.0
            
            # Classify
            resistance_type = self._classify_sample(
                dmso_sim, moa_sim, max_cross_sim
            )
            
            labels.append({
                **meta,
                'resistance_type': resistance_type.name,
                'resistance_code': resistance_type.value,
                'dmso_similarity': dmso_sim,
                'moa_similarity': moa_sim,
                'cross_moa': max_cross_moa if resistance_type == ResistanceType.CROSS_RESISTANCE else None,
                'cross_moa_similarity': max_cross_sim
            })
        
        return pd.DataFrame(labels)
    
    def _classify_sample(self, dmso_sim, moa_sim, cross_moa_sim):
        """
        Classification logic:
        1. High DMSO similarity + Low MOA similarity → PRIMARY_RESISTANCE
        2. Low DMSO similarity + High MOA similarity → SENSITIVE
        3. Medium MOA similarity → PARTIAL_RESISTANCE
        4. Low MOA similarity + High cross-MOA similarity → CROSS_RESISTANCE
        5. Otherwise → UNCERTAIN
        """
        
        # Primary resistance: looks like DMSO
        if dmso_sim >= Config.DMSO_HIGH_SIMILARITY and moa_sim < Config.MOA_MEDIUM_SIMILARITY:
            return ResistanceType.PRIMARY_RESISTANCE
        
        # Sensitive: shows expected MOA
        if moa_sim >= Config.MOA_HIGH_SIMILARITY:
            return ResistanceType.SENSITIVE
        
        # Cross-resistance: shows different MOA
        if (moa_sim < Config.MOA_MEDIUM_SIMILARITY and 
            cross_moa_sim >= Config.CROSS_MOA_THRESHOLD):
            return ResistanceType.CROSS_RESISTANCE
        
        # Partial resistance: weak MOA signal
        if moa_sim >= Config.MOA_MEDIUM_SIMILARITY:
            return ResistanceType.PARTIAL_RESISTANCE
        
        # Uncertain
        return ResistanceType.UNCERTAIN

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(labels_df, output_dir):
    """Create visualizations of resistance distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Resistance type distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    resistance_counts = labels_df['resistance_type'].value_counts()
    plt.bar(resistance_counts.index, resistance_counts.values)
    plt.xlabel('Resistance Type')
    plt.ylabel('Count')
    plt.title('Distribution of Resistance Types')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    # Resistance by MOA
    moa_resistance = pd.crosstab(
        labels_df['moa'],
        labels_df['resistance_type'],
        normalize='index'
    )
    moa_resistance.plot(kind='bar', stacked=True)
    plt.xlabel('MOA')
    plt.ylabel('Proportion')
    plt.title('Resistance Types by MOA')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Resistance Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resistance_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir}/resistance_distribution.png")
    
    # Similarity distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for rtype in labels_df['resistance_type'].unique():
        subset = labels_df[labels_df['resistance_type'] == rtype]
        plt.hist(subset['dmso_similarity'], alpha=0.5, label=rtype, bins=20)
    plt.xlabel('DMSO Similarity')
    plt.ylabel('Count')
    plt.title('DMSO Similarity by Resistance Type')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for rtype in labels_df['resistance_type'].unique():
        subset = labels_df[labels_df['resistance_type'] == rtype]
        plt.hist(subset['moa_similarity'], alpha=0.5, label=rtype, bins=20)
    plt.xlabel('MOA Similarity')
    plt.ylabel('Count')
    plt.title('MOA Similarity by Resistance Type')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    for rtype in labels_df['resistance_type'].unique():
        subset = labels_df[labels_df['resistance_type'] == rtype]
        plt.hist(subset['cross_moa_similarity'], alpha=0.5, label=rtype, bins=20)
    plt.xlabel('Cross-MOA Similarity')
    plt.ylabel('Count')
    plt.title('Cross-MOA Similarity by Resistance Type')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), dpi=300)
    print(f"Saved similarity distributions to {output_dir}/similarity_distributions.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading metadata...")
    image_df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.IMAGE_CSV))
    moa_df = pd.read_csv(os.path.join(Config.DATA_DIR, Config.MOA_CSV))
    
    # Filter to Week 1-3
    week_pattern = '|'.join([f'{w}/' for w in Config.WEEK_FILTER])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    print(f"Filtered to {len(image_df)} images")
    
    # Create dataset
    dataset = InferenceDataset(image_df, moa_df, Config.DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Load Phase 1 model
    print("Loading Phase 1 model...")
    model = MOAEncoder().to(Config.DEVICE)
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")
    
    # Generate labels
    generator = ResistanceLabelGenerator(model, dataloader, Config.DEVICE)
    generator.extract_embeddings()
    generator.compute_moa_references()
    labels_df = generator.classify_resistance()
    
    # Save labels
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(Config.OUTPUT_DIR, 'resistance_labels.csv')
    labels_df.to_csv(output_file, index=False)
    print(f"\n✓ Labels saved to {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("RESISTANCE CLASSIFICATION SUMMARY")
    print("="*60)
    print(labels_df['resistance_type'].value_counts())
    print()
    print(labels_df.groupby('resistance_type')[['dmso_similarity', 'moa_similarity', 'cross_moa_similarity']].mean())
    
    # Visualize
    visualize_results(labels_df, Config.OUTPUT_DIR)
    
    print("\n✓ Phase 2 Complete!")

if __name__ == '__main__':
    main()
