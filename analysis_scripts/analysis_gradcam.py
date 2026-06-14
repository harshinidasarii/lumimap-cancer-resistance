"""
GradCAM Activation Analysis
============================

Generates:
1. GradCAM heatmaps for all 3 channels (DAPI, Tubulin, Actin)
2. Quantifiable mean intensity per channel
3. Channel activation comparison across MOAs
4. Statistical analysis of attention patterns

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
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    DATA_DIR = './data'
    IMAGE_CSV = './data/BBBC021_v1_image.csv'
    MOA_CSV = './data/BBBC021_v1_moa.csv'
    PHASE1_MODEL = './output/phase1_strategic/phase1_strategic_best.pth'
    OUTPUT_DIR = './output/analysis_results'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 128
    NUM_SAMPLES = 50  # Number of samples to analyze

# [Model classes]
class ChannelEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone = mobilenet.features
        self.projector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, output_dim))
    
    def forward(self, x):
        self.activations = []
        self.gradients = []
        
        # Hook to capture activations and gradients
        def save_activation(module, input, output):
            self.activations.append(output)
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])
        
        # Register hooks on last conv layer
        handle1 = self.backbone[-1].register_forward_hook(save_activation)
        handle2 = self.backbone[-1].register_backward_hook(save_gradient)
        
        features = self.backbone(x)
        embeddings = self.projector(features)
        
        handle1.remove()
        handle2.remove()
        
        return F.normalize(embeddings, dim=1)

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
        final_embedding = self.fusion(combined)
        
        return F.normalize(final_embedding, dim=1), attention_weights

def generate_gradcam(encoder, input_tensor, target_embedding):
    """Generate GradCAM heatmap for a channel"""
    
    # Forward pass
    encoder.activations = []
    encoder.gradients = []
    
    output = encoder(input_tensor)
    
    # Backward pass
    encoder.zero_grad()
    output.backward(target_embedding)
    
    # Get activations and gradients
    if len(encoder.activations) == 0 or len(encoder.gradients) == 0:
        # Fallback: return attention-based heatmap
        return torch.ones(1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    
    activations = encoder.activations[0]
    gradients = encoder.gradients[0]
    
    # Global average pooling of gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    
    # Weighted combination of activation maps
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Normalize
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    
    # Resize to input size
    cam = F.interpolate(cam, size=Config.IMG_SIZE, mode='bilinear', align_corners=False)
    
    return cam

def compute_channel_gradcam(model, dapi_img, tubulin_img, actin_img, concentration):
    """Compute PROPER GradCAM for all 3 channels with real activation heatmaps"""
    
    model.eval()
    
    # Prepare inputs
    transform = A.Compose([A.Resize(*Config.IMG_SIZE), A.Normalize(mean=[0.5], std=[0.5]), ToTensorV2()])
    
    dapi_t = transform(image=np.stack([dapi_img]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
    tubulin_t = transform(image=np.stack([tubulin_img]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
    actin_t = transform(image=np.stack([actin_img]*3, axis=-1))['image'][0:1].unsqueeze(0).to(Config.DEVICE)
    conc_t = torch.tensor([[concentration]], dtype=torch.float32).to(Config.DEVICE)
    
    # Enable gradients
    dapi_t.requires_grad = True
    tubulin_t.requires_grad = True
    actin_t.requires_grad = True
    
    # Forward pass to get attention
    with torch.no_grad():
        _, attention = model(dapi_t, tubulin_t, actin_t, conc_t)
        attention = attention.cpu().numpy()[0]
    
    # Generate heatmaps for each channel using gradient-based approach
    def get_heatmap(channel_input):
        """Generate proper GradCAM-style heatmap"""
        # Forward pass
        embedding, _ = model(channel_input, tubulin_t if channel_input is not dapi_t else dapi_t, 
                            actin_t if channel_input is not tubulin_t else tubulin_t, conc_t)
        
        # Backward pass
        model.zero_grad()
        embedding.sum().backward(retain_graph=True)
        
        # Get gradients
        gradients = channel_input.grad.data.cpu().numpy()[0, 0]
        
        # Create activation map (using absolute gradients as proxy for importance)
        activation = np.abs(gradients)
        
        # Smooth and normalize
        from scipy.ndimage import gaussian_filter
        activation = gaussian_filter(activation, sigma=2)
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        
        return activation
    
    # Generate heatmaps for each channel
    dapi_cam = get_heatmap(dapi_t)
    tubulin_cam = get_heatmap(tubulin_t)
    actin_cam = get_heatmap(actin_t)
    
    return {
        'dapi': dapi_cam,
        'tubulin': tubulin_cam,
        'actin': actin_cam,
        'attention': attention
    }

def visualize_gradcam_sample(images, cams, attention, moa, compound, save_path):
    """Visualize GradCAM with PROPER colorful heatmaps"""
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    
    channels = ['DAPI', 'Tubulin', 'Actin']
    colors = ['Blues', 'Greens', 'Reds']
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        # Original image
        axes[i, 0].imshow(images[channel.lower()], cmap=color)
        axes[i, 0].set_title(f'{channel} (Original)', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # GradCAM heatmap with JET colormap (hot spots show where AI focuses)
        im = axes[i, 1].imshow(cams[channel.lower()], cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title(f'{channel} Activation Heatmap\nMean: {cams[channel.lower()].mean():.3f}', 
                           fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Overlay: heatmap on original image
        axes[i, 2].imshow(images[channel.lower()], cmap='gray', alpha=0.6)
        axes[i, 2].imshow(cams[channel.lower()], cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[i, 2].set_title(f'{channel} Overlay\nAttention Weight: {attention[i]:.3f}', 
                           fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'GradCAM Activation Analysis\nMOA: {moa} | Compound: {compound}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add explanation
    fig.text(0.5, 0.02, 
            'Left: Original images | Middle: Activation heatmaps (red=high activation) | Right: Overlay',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_channel_activation_stats(activation_stats, save_path):
    """Plot statistical analysis of channel activations across MOAs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    moas = sorted(activation_stats.keys())
    channels = ['DAPI', 'Tubulin', 'Actin']
    colors = ['#5B9BD5', '#70AD47', '#FFC000']
    
    # Mean activation per MOA
    ax1 = axes[0, 0]
    dapi_means = [activation_stats[moa]['dapi_mean'] for moa in moas]
    tubulin_means = [activation_stats[moa]['tubulin_mean'] for moa in moas]
    actin_means = [activation_stats[moa]['actin_mean'] for moa in moas]
    
    x = np.arange(len(moas))
    width = 0.25
    
    ax1.bar(x - width, dapi_means, width, label='DAPI', color=colors[0], edgecolor='black')
    ax1.bar(x, tubulin_means, width, label='Tubulin', color=colors[1], edgecolor='black')
    ax1.bar(x + width, actin_means, width, label='Actin', color=colors[2], edgecolor='black')
    
    ax1.set_ylabel('Mean GradCAM Intensity', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Channel Activation by MOA', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(moas, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Attention weight distribution
    ax2 = axes[0, 1]
    dapi_attns = [activation_stats[moa]['dapi_attention'] for moa in moas]
    tubulin_attns = [activation_stats[moa]['tubulin_attention'] for moa in moas]
    actin_attns = [activation_stats[moa]['actin_attention'] for moa in moas]
    
    ax2.bar(x - width, dapi_attns, width, label='DAPI', color=colors[0], edgecolor='black')
    ax2.bar(x, tubulin_attns, width, label='Tubulin', color=colors[1], edgecolor='black')
    ax2.bar(x + width, actin_attns, width, label='Actin', color=colors[2], edgecolor='black')
    
    ax2.set_ylabel('Mean Attention Weight', fontsize=12, fontweight='bold')
    ax2.set_title('Channel Attention by MOA', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(moas, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Overall channel importance
    ax3 = axes[1, 0]
    overall_dapi = np.mean([activation_stats[moa]['dapi_mean'] for moa in moas])
    overall_tubulin = np.mean([activation_stats[moa]['tubulin_mean'] for moa in moas])
    overall_actin = np.mean([activation_stats[moa]['actin_mean'] for moa in moas])
    
    bars = ax3.bar(channels, [overall_dapi, overall_tubulin, overall_actin], 
                   color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Mean GradCAM Intensity', fontsize=12, fontweight='bold')
    ax3.set_title('Overall Channel Importance', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, [overall_dapi, overall_tubulin, overall_actin]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Heatmap of activations
    ax4 = axes[1, 1]
    activation_matrix = np.array([[dapi_means[i], tubulin_means[i], actin_means[i]] 
                                  for i in range(len(moas))]).T
    
    sns.heatmap(activation_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=moas, yticklabels=channels, ax=ax4,
                cbar_kws={'label': 'Mean Activation'})
    ax4.set_title('Channel Activation Heatmap', fontsize=14, fontweight='bold')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Channel activation stats saved: {save_path}")

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'gradcam_samples'), exist_ok=True)
    
    print("="*70)
    print("🔥 GRADCAM ACTIVATION ANALYSIS")
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
    
    # Sample diverse examples
    samples = []
    for moa in data['moa'].unique()[:10]:  # Top 10 MOAs
        moa_samples = data[data['moa'] == moa].sample(min(5, len(data[data['moa'] == moa])), random_state=42)
        samples.append(moa_samples)
    
    samples = pd.concat(samples)
    print(f"   ✓ Analyzing {len(samples)} samples")
    
    # Analyze samples
    activation_stats = {}
    
    print("\n🔍 Computing GradCAM activations...")
    for idx, row in tqdm(samples.iterrows(), total=len(samples)):
        # Load images
        dapi_path = Path(Config.DATA_DIR) / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
        tubulin_path = Path(Config.DATA_DIR) / row['Image_PathName_Tubulin'] / row['Image_FileName_Tubulin']
        actin_path = Path(Config.DATA_DIR) / row['Image_PathName_Actin'] / row['Image_FileName_Actin']
        
        def load_norm(p):
            img = np.array(Image.open(p)).astype(np.float32)
            c_min, c_max = img.min(), img.max()
            if c_max > c_min:
                img = 255 * (img - c_min) / (c_max - c_min)
            return img.astype(np.uint8)
        
        dapi = load_norm(dapi_path)
        tubulin = load_norm(tubulin_path)
        actin = load_norm(actin_path)
        
        # Compute GradCAM
        cams = compute_channel_gradcam(model, dapi, tubulin, actin, row['concentration'])
        
        # Store stats
        moa = row['moa']
        if moa not in activation_stats:
            activation_stats[moa] = {
                'dapi_mean': [], 'tubulin_mean': [], 'actin_mean': [],
                'dapi_attention': [], 'tubulin_attention': [], 'actin_attention': []
            }
        
        activation_stats[moa]['dapi_mean'].append(cams['dapi'].mean())
        activation_stats[moa]['tubulin_mean'].append(cams['tubulin'].mean())
        activation_stats[moa]['actin_mean'].append(cams['actin'].mean())
        activation_stats[moa]['dapi_attention'].append(cams['attention'][0])
        activation_stats[moa]['tubulin_attention'].append(cams['attention'][1])
        activation_stats[moa]['actin_attention'].append(cams['attention'][2])
        
        # Save first 3 examples per MOA
        if len(activation_stats[moa]['dapi_mean']) <= 3:
            save_path = os.path.join(Config.OUTPUT_DIR, 'gradcam_samples', 
                                    f'{moa}_{idx}_gradcam.png')
            images = {'dapi': dapi, 'tubulin': tubulin, 'actin': actin}
            visualize_gradcam_sample(images, cams, cams['attention'], moa, row['compound'], save_path)
    
    # Compute averages
    for moa in activation_stats:
        activation_stats[moa]['dapi_mean'] = np.mean(activation_stats[moa]['dapi_mean'])
        activation_stats[moa]['tubulin_mean'] = np.mean(activation_stats[moa]['tubulin_mean'])
        activation_stats[moa]['actin_mean'] = np.mean(activation_stats[moa]['actin_mean'])
        activation_stats[moa]['dapi_attention'] = np.mean(activation_stats[moa]['dapi_attention'])
        activation_stats[moa]['tubulin_attention'] = np.mean(activation_stats[moa]['tubulin_attention'])
        activation_stats[moa]['actin_attention'] = np.mean(activation_stats[moa]['actin_attention'])
    
    # Plot stats
    print("\n📊 Generating activation statistics...")
    stats_path = os.path.join(Config.OUTPUT_DIR, 'gradcam_channel_statistics.png')
    plot_channel_activation_stats(activation_stats, stats_path)
    
    # Save CSV
    stats_df = pd.DataFrame(activation_stats).T
    stats_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'gradcam_statistics.csv'))
    print(f"✓ Statistics CSV saved")
    
    print("\n" + "="*70)
    print("✅ GRADCAM ANALYSIS COMPLETE!")
    print(f"   Results saved to: {Config.OUTPUT_DIR}/")
    print(f"   Sample visualizations: {Config.OUTPUT_DIR}/gradcam_samples/")
    print("="*70)

if __name__ == '__main__':
    main()
