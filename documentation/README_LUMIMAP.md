# LUMIMAP: AI-Powered Cancer Drug Resistance Detection System

**L**earning-based **U**niversal **M**echanism **I**dentification for **M**ultidrug **A**daptation **P**henotyping

A deep learning system for detecting and classifying drug resistance in cancer cells using microscopy images.

---

## 🎯 What LUMIMAP Does

LUMIMAP analyzes microscopy images of cancer cells treated with drugs and:

1. **Detects Resistance Type** - Not just "resistant" or "sensitive", but identifies:
   - **Sensitive**: Drug working as expected
   - **Primary Resistance**: No drug effect (looks like untreated cells)
   - **Partial Resistance**: Weak drug response
   - **Cross-Resistance**: Cell using alternative pathway to bypass the drug
   - **Uncertain**: Ambiguous response

2. **Identifies Bypass Pathways** - When cross-resistance is detected, identifies which alternative cellular mechanism the cell activated

3. **Provides Therapy Recommendations** - Suggests:
   - Alternative drugs
   - Combination therapies
   - Dose adjustments

4. **Visualizes with Channel-wise GradCAM** - Shows exactly which cellular structures (nucleus, microtubules, or cytoskeleton) drove the prediction

---

## 🔬 How It Works

### The Problem We Solved

Traditional drug resistance detection had a fundamental flaw:
```
❌ Old approach: 
   Input: Image → Output: "Resistant" or "Sensitive"
   Problem: Doesn't know WHICH drug → can't know what phenotype to expect!

✅ LUMIMAP approach:
   Input: Image + Drug + Concentration → Output: Specific resistance type + Mechanism
   Innovation: Learns what each drug's MOA (Mechanism of Action) looks like
```

### Key Innovations

1. **Channel-Aware Learning**
   - Separate encoders for DAPI (nucleus), Tubulin (microtubules), Actin (cytoskeleton)
   - Model learns which channel matters for each drug type
   - Example: Microtubule drugs → high attention on Tubulin channel

2. **Multi-Drug Same-MOA Learning**
   - Learns that taxol, docetaxel, epothilone B should produce similar phenotypes
   - Generalizes across different compounds with same mechanism

3. **Contrastive Learning with DMSO Baseline**
   - DMSO (no drug) = healthy cells = what resistance looks like
   - Model learns: "Treated cells looking like DMSO = resistant"

4. **Multi-Type Resistance Classification**
   - Not binary (yes/no resistance)
   - Identifies specific resistance mechanisms
   - Enables targeted therapy recommendations

---

## 📊 Dataset

**BBBC021** (Broad Bioimage Benchmark Collection)
- 39,600 images (13,200 fields of view × 3 channels)
- 113 compounds tested at 8 concentrations each
- 12 different Mechanisms of Action (MOAs)
- MCF-7 breast cancer cells
- 3 fluorescent channels:
  - **DAPI**: Nucleus/DNA (blue)
  - **Tubulin**: Microtubules/spindle (green)  
  - **Actin**: Cytoskeleton (red)

### MOA Categories in Dataset:
1. Actin disruptors
2. Aurora kinase inhibitors
3. Eg5 inhibitors
4. Microtubule destabilizers
5. Microtubule stabilizers
6. DNA damage
7. DNA replication inhibitors
8. Protein synthesis inhibitors
9. Protein degradation
10. Kinase inhibitors
11. Cholesterol-lowering
12. Epithelial

---

## 🏗️ System Architecture

### Three-Phase Training Pipeline

#### **Phase 1: Contrastive MOA Learning** (`phase1_contrastive_moa_learning.py`)

Learns what each MOA phenotype looks like.

**Architecture:**
```python
Input: 3 separate channels (DAPI, Tubulin, Actin) + Concentration

├─ DAPI Encoder (ResNet)
├─ Tubulin Encoder (ResNet)
├─ Actin Encoder (ResNet)
├─ Concentration Encoder
├─ Channel Attention (learns which channel matters)
└─ Fusion → Final MOA embedding

Training: Triplet Loss
- Anchor: Drug A with MOA X
- Positive: Drug B with MOA X (different drug, same MOA)
- Negative: Drug C with MOA Y (different MOA)
```

**Output:** Trained encoder that creates MOA-specific embeddings

#### **Phase 2: Resistance Label Generation** (`phase2_generate_resistance_labels.py`)

Generates pseudo-labels for resistance using Phase 1 encoder.

**Algorithm:**
```python
For each drug-treated cell:
1. Extract features using Phase 1 encoder
2. Compare similarity to DMSO (healthy baseline)
3. Compare similarity to same-MOA references
4. Compare similarity to other MOAs
5. Classify:
   - High DMSO sim + Low MOA sim → PRIMARY_RESISTANCE
   - Low DMSO sim + High MOA sim → SENSITIVE
   - Medium MOA sim → PARTIAL_RESISTANCE
   - Low MOA sim + High other-MOA sim → CROSS_RESISTANCE
```

**Output:** CSV with resistance labels for all images

#### **Phase 3: Resistance Classifier** (`phase3_resistance_classifier_train.py`)

Trains final classifier using pseudo-labels.

**Architecture:**
```python
Input: Image channels + Drug + Concentration

├─ Phase 1 Encoder (frozen initially)
├─ Classification Head
└─ Output: 5-class resistance prediction

Training strategy:
- First 10 epochs: Freeze encoder, train classifier
- Next 30 epochs: Unfreeze encoder, fine-tune end-to-end
```

**Output:** Production-ready resistance classifier

---

## 🚀 Usage

### Installation

```bash
# Create environment
conda create -n lumimap python=3.9
conda activate lumimap

# Install dependencies
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install pillow opencv-python scikit-image
pip install albumentations matplotlib seaborn
pip install tqdm
```

### Training

```bash
# Phase 1: Learn MOA phenotypes (run first)
python phase1_contrastive_moa_learning.py

# Phase 2: Generate resistance labels (run after Phase 1)
python phase2_generate_resistance_labels.py

# Phase 3: Train resistance classifier (run after Phase 2)
python phase3_resistance_classifier_train.py
```

### Inference

```bash
# Analyze a single cell image
python lumimap_inference.py \
    --dapi path/to/dapi.tif \
    --tubulin path/to/tubulin.tif \
    --actin path/to/actin.tif \
    --drug taxol \
    --concentration 1.0 \
    --moa "Microtubule stabilizers" \
    --output result.png
```

**Example Output:**
```
LUMIMAP ANALYSIS COMPLETE
====================================================================

Resistance Type: CROSS_RESISTANCE
Confidence: 87.3%

Recommendation: Combination therapy targeting both Microtubule stabilizers and Actin disruptors
Rationale: Cell activated Actin disruption bypass pathway. Dual targeting recommended.

Alternative drugs: cytochalasin B, cytochalasin D, latrunculin B

✓ Full report saved to result.png
====================================================================
```

---

## 📈 Results

### Performance Metrics (Expected)

**Phase 1 (MOA Classification):**
- Accuracy: ~85-90% on MOA prediction
- Strong clustering of same-MOA compounds

**Phase 2 (Label Quality):**
- High-confidence labels: ~70-80% of dataset
- Clear separation between resistance types

**Phase 3 (Resistance Classification):**
- Overall Accuracy: ~75-85%
- Per-class F1 scores:
  - Sensitive: 0.85-0.90
  - Primary Resistance: 0.75-0.80
  - Partial Resistance: 0.70-0.75
  - Cross-Resistance: 0.65-0.75
  - Uncertain: 0.50-0.60 (intentionally low confidence)

---

## 🎨 Visualization

### GradCAM Output

The system generates channel-wise GradCAM showing:

```
Row 1: Original Images
  [DAPI]    [Tubulin]    [Actin]

Row 2: GradCAM Heatmaps (what the model looks at)
  [DAPI CAM] [Tubulin CAM] [Actin CAM]
  Attn: 0.15  Attn: 0.75    Attn: 0.10
  
Row 3: Prediction & Recommendations
  Resistance Type: CROSS_RESISTANCE
  Expected MOA: Microtubule stabilization
  Observed MOA: Actin disruption
  
  Recommendation: Combination therapy
  Alternatives: cytochalasin B, latrunculin B
```

**Interpretation:**
- High attention on Tubulin expected for microtubule drugs
- If high attention on Actin instead → cell using bypass mechanism
- Recommendations adjust based on which channel/mechanism activated

---

## 📂 File Structure

```
lumimap/
├── data/
│   ├── Week1_22123/        # Image folders
│   ├── Week1_22141/
│   ├── ...
│   ├── BBBC021_v1_image.csv
│   ├── BBBC021_v1_moa.csv
│   └── BBBC021_v1_compound.csv
│
├── outputs/
│   ├── phase1/
│   │   ├── phase1_best_model.pth
│   │   └── phase1_losses.png
│   ├── phase2/
│   │   ├── resistance_labels.csv
│   │   ├── resistance_distribution.png
│   │   └── similarity_distributions.png
│   └── phase3/
│       ├── phase3_best_model.pth
│       ├── phase3_training.png
│       ├── confusion_matrix.png
│       └── classification_report.txt
│
├── phase1_contrastive_moa_learning.py
├── phase2_generate_resistance_labels.py
├── phase3_resistance_classifier_train.py
├── lumimap_inference.py
└── README.md
```

---

## 🧬 Clinical Applications

### Personalized Therapy Selection

1. **Tumor Biopsy** → Microscopy images
2. **LUMIMAP Analysis** → Resistance profile
3. **Therapy Selection** based on:
   - Which drugs cell is sensitive to
   - Which pathways cell uses to resist
   - Combination therapy if cross-resistance detected

### Example Clinical Workflow

```
Patient: Breast cancer, failed taxol treatment

LUMIMAP Analysis on tumor cells:
→ Resistance Type: CROSS_RESISTANCE
→ Observation: Cells showing actin disruption instead of microtubule stabilization
→ Interpretation: Tumor cells bypassing microtubule pathway via cytoskeleton remodeling

Recommendation:
→ Combination: Taxol + Cytochalasin B
→ Rationale: Block both microtubule and actin pathways
→ Alternative: Switch to different MOA entirely (e.g., DNA damage agents)
```

---

## 🎓 Science Fair Presentation Tips

### Key Talking Points

1. **The Problem:**
   - Cancer cells develop resistance to drugs
   - Doctors need to know: Why resistant? What to try next?
   
2. **Why It's Hard:**
   - Different drugs work differently (different MOAs)
   - Resistance can happen multiple ways
   - Need to know which pathway cell is using
   
3. **Our Solution:**
   - AI learns what each drug's effect looks like
   - Compares treated cells to healthy cells
   - Identifies which resistance mechanism
   - Recommends specific alternative therapies
   
4. **Innovation:**
   - First system to classify resistance TYPES (not just yes/no)
   - Channel-wise learning (looks at nucleus, microtubules, cytoskeleton separately)
   - Provides actionable therapy recommendations

### Demo Script

```
1. Show microscopy images (before/after drug treatment)
2. Run LUMIMAP inference live
3. Show GradCAM - "AI is looking at microtubules, not nucleus"
4. Explain prediction - "Cross-resistance means bypass pathway"
5. Show recommendations - "Try these alternative drugs"
6. Compare to traditional approach - "Old way just says 'resistant'"
```

---

## 🔮 Future Improvements

1. **More Drug Classes**: Expand beyond BBBC021's 12 MOAs
2. **Patient Data**: Train on real tumor biopsies
3. **Time Series**: Analyze how resistance develops over time
4. **Drug Combinations**: Predict optimal combination therapies
5. **Multi-Modal**: Integrate genomic data with imaging

---

## 📚 References

1. BBBC021 Dataset: https://bbbc.broadinstitute.org/BBBC021
2. Caie et al., "High-Content Phenotypic Profiling of Drug Response", Molecular Cancer Therapeutics, 2010
3. Original inspiration: Computational drug sensitivity prediction

---

## 👥 Credits

**Harshini** - Lead researcher, model development, science fair presentation
**Sanyasi** - Technical advisor, debugging, infrastructure

ISEF 2025 Project

---

## 📝 License

Educational use only. BBBC021 dataset copyright AstraZeneca Pharmaceuticals.

---

## 🆘 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch size in Config
- Use smaller image size (e.g., 128x128)

**"Can't find image files"**
- Check DATA_DIR path
- Verify Week1/Week2/Week3 folders exist
- Check CSV file paths match actual files

**"Model not converging"**
- Increase NUM_EPOCHS
- Adjust learning rate
- Check data distribution (enough samples per class?)

**"Low accuracy on resistance detection"**
- Phase 1 might not have converged - check loss curves
- Phase 2 labels might be noisy - adjust similarity thresholds
- Try different backbone (efficientnet instead of resnet)

---

For questions: Contact via ISEF portal or science fair booth!
