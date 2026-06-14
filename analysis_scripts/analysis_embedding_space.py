"""
Embedding Space Analysis
========================

Generates:
1. t-SNE visualization of embeddings colored by MOA
2. PCA visualization with variance explained
3. Morphological deviation framework - biological clustering analysis
4. Inter-MOA similarity heatmap

Saves all to: ./output/analysis_results/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from tqdm import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    MOA_CSV = './data/BBBC021_v1_moa.csv'
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    OUTPUT_DIR = './output/analysis_results'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128
    BATCH_SIZE = 64
    NUM_WORKERS = 0  # Fix for Python 3.14 multiprocessing issue

# [Model classes - abbreviated for space]
class ChannelEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone = mobilenet.features
        self.projector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, output_dim))
    
    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(self.projector(features), dim=1)

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
        flat = channel_embeddings.view(batch_size, -1)
        weights = self.attention(flat).unsqueeze(2)
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

def extract_embeddings(model, dataloader, device):
    """Extract all embeddings with labels"""
    model.eval()
    
    all_embeddings = []
    all_moas = []
    all_compounds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            dapi = batch['dapi'].to(device)
            tubulin = batch['tubulin'].to(device)
            actin = batch['actin'].to(device)
            conc = batch['concentration'].to(device)
            
            embeddings, _ = model(dapi, tubulin, actin, conc)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_moas.extend(batch['moa'])
            all_compounds.extend(batch['compound'])
    
    all_embeddings = np.vstack(all_embeddings)
    
    return all_embeddings, all_moas, all_compounds

def plot_tsne(embeddings, moas, save_path):
    """t-SNE visualization colored by MOA"""
    
    print("   Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)  # Changed n_iter to max_iter
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create color map
    unique_moas = sorted(list(set(moas)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_moas)))
    moa_to_color = {moa: colors[i] for i, moa in enumerate(unique_moas)}
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for moa in unique_moas:
        mask = [m == moa for m in moas]
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[moa_to_color[moa]], label=moa, alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    
    ax.set_title('t-SNE Embedding Space (Colored by MOA)', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ t-SNE plot saved: {save_path}")

def plot_pca(embeddings, moas, save_path):
    """PCA visualization with variance explained"""
    
    print("   Computing PCA...")
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    
    unique_moas = sorted(list(set(moas)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_moas)))
    moa_to_color = {moa: colors[i] for i, moa in enumerate(unique_moas)}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # PC1 vs PC2
    for moa in unique_moas:
        mask = [m == moa for m in moas]
        ax1.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                   c=[moa_to_color[moa]], label=moa, alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    
    ax1.set_title(f'PCA Embedding Space (PC1 vs PC2)\nVariance: {pca.explained_variance_ratio_[:2].sum():.1%}',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Variance explained
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum_variance)+1), cumsum_variance, 'o-', linewidth=2, markersize=4)
    ax2.axhline(y=0.95, color='red', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.set_title('PCA Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ PCA plot saved: {save_path}")

def plot_morphological_deviation(embeddings, moas, save_path):
    """Morphological deviation framework - biological clustering analysis"""
    
    print("   Computing MOA centroids and similarities...")
    
    # Compute MOA centroids
    unique_moas = sorted(list(set(moas)))
    moa_centroids = {}
    
    for moa in unique_moas:
        mask = [m == moa for m in moas]
        moa_centroids[moa] = embeddings[mask].mean(axis=0)
    
    # Compute inter-MOA similarity matrix
    n_moas = len(unique_moas)
    similarity_matrix = np.zeros((n_moas, n_moas))
    
    for i, moa1 in enumerate(unique_moas):
        for j, moa2 in enumerate(unique_moas):
            # Cosine similarity between centroids
            sim = np.dot(moa_centroids[moa1], moa_centroids[moa2])
            similarity_matrix[i, j] = sim
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Heatmap
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=unique_moas, yticklabels=unique_moas,
                ax=ax1, vmin=0, vmax=1, cbar_kws={'label': 'Cosine Similarity'})
    ax1.set_title('Inter-MOA Similarity Matrix\n(Biological Clustering Analysis)', 
                 fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Hierarchical clustering dendrogram
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    
    dendrogram(linkage_matrix, labels=unique_moas, ax=ax2, leaf_font_size=10)
    ax2.set_title('MOA Hierarchical Clustering\n(Similar mechanisms cluster together)', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Distance (1 - Similarity)', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Morphological deviation plot saved: {save_path}")
    
    # Find most similar MOA pairs
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
    print("\n   📊 Most similar MOA pairs (biological clustering):")
    for i in range(n_moas):
        for j in range(i+1, n_moas):
            if similarity_matrix[i, j] > 0.85:  # High similarity threshold
                print(f"      {unique_moas[i]} <-> {unique_moas[j]}: {similarity_matrix[i, j]:.3f}")
    
    return similarity_matrix, unique_moas

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("🔬 EMBEDDING SPACE ANALYSIS")
    print("="*70)
    
    # Load model
    print("\n📂 Loading Phase 1 model...")
    model = MOAEncoder()
    checkpoint = torch.load(Config.PHASE1_MODEL, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    print("   ✓ Model loaded")
    
    # Load data
    print("\n📂 Loading data...")
    image_df = pd.read_csv(Config.IMAGE_CSV)
    moa_df = pd.read_csv(Config.MOA_CSV)
    
    week_pattern = '|'.join([f'Week{i}/' for i in range(1, 7)])
    image_df = image_df[image_df['Image_PathName_DAPI'].str.contains(week_pattern)]
    
    data = image_df.merge(moa_df, 
                         left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
                         right_on=['compound', 'concentration'], how='inner')
    
    print(f"   ✓ {len(data)} samples loaded")
    
    # Create dataset
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, data_df, root_dir):
            self.data = data_df
            self.root_dir = Path(root_dir)
            self.transform = A.Compose([A.Resize(*Config.IMG_SIZE), A.Normalize(mean=[0.5], std=[0.5]), ToTensorV2()])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            dapi_path = self.root_dir / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
            tubulin_path = self.root_dir / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
            actin_path = self.root_dir / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
            
            def load_norm(p):
                img = np.array(Image.open(p)).astype(np.float32)
                c_min, c_max = img.min(), img.max()
                if c_max > c_min:
                    img = 255 * (img - c_min) / (c_max - c_min)
                return img.astype(np.uint8)
            
            dapi = self.transform(image=np.stack([load_norm(dapi_path)]*3, axis=-1))['image'][0:1]
            tubulin = self.transform(image=np.stack([load_norm(tubulin_path)]*3, axis=-1))['image'][0:1]
            actin = self.transform(image=np.stack([load_norm(actin_path)]*3, axis=-1))['image'][0:1]
            
            return {
                'dapi': dapi, 'tubulin': tubulin, 'actin': actin,
                'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
                'moa': row['moa'], 'compound': row['compound']
            }
    
    dataset = SimpleDataset(data, Config.DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Extract embeddings
    print("\n🔍 Extracting embeddings...")
    embeddings, moas, compounds = extract_embeddings(model, dataloader, Config.DEVICE)
    print(f"   ✓ Extracted {len(embeddings)} embeddings")
    
    # 1. t-SNE
    print("\n📊 Generating t-SNE visualization...")
    tsne_path = os.path.join(Config.OUTPUT_DIR, 'embedding_tsne.png')
    plot_tsne(embeddings, moas, tsne_path)
    
    # 2. PCA
    print("\n📊 Generating PCA visualization...")
    pca_path = os.path.join(Config.OUTPUT_DIR, 'embedding_pca.png')
    plot_pca(embeddings, moas, pca_path)
    
    # 3. Morphological deviation
    print("\n📊 Generating morphological deviation analysis...")
    morph_path = os.path.join(Config.OUTPUT_DIR, 'morphological_deviation.png')
    similarity_matrix, moa_names = plot_morphological_deviation(embeddings, moas, morph_path)
    
    # Save similarity matrix
    sim_df = pd.DataFrame(similarity_matrix, columns=moa_names, index=moa_names)
    sim_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'moa_similarity_matrix.csv'))
    print(f"   ✓ Similarity matrix CSV saved")
    
    print("\n" + "="*70)
    print("✅ EMBEDDING SPACE ANALYSIS COMPLETE!")
    print(f"   All results saved to: {Config.OUTPUT_DIR}/")
    print("="*70)

if __name__ == '__main__':
    main()
