# LUMIMAP Quick Start Guide for SFPROJECT

## Your Current Data Structure
```
SFPROJECT/
├── data/
│   ├── Week1/
│   │   ├── Week1_22123/
│   │   ├── Week1_22141/
│   │   └── ...
│   ├── Week2/
│   ├── Week3/
│   ├── BBBC021_v1_compound.csv  ← All weeks share these CSVs
│   ├── BBBC021_v1_image.csv
│   └── BBBC021_v1_moa.csv
├── input/
└── output/
```

## Files You Need (Updated for Your Structure)

**NEW FILES (use these):**
1. `phase1_contrastive_moa_learning_UPDATED.py`
2. `phase2_generate_resistance_labels_UPDATED.py`  
3. `phase3_resistance_classifier_train_UPDATED.py`
4. `lumimap_inference_UPDATED.py`

**OLD FILES (ignore these):**
- Any files without `_UPDATED` suffix

## Step-by-Step Training

### 1. Run Phase 1 (Learn MOA Phenotypes)
```bash
python phase1_contrastive_moa_learning_UPDATED.py
```

**What it does:**
- Learns what each MOA looks like from Week1-3 data
- Creates separate encoders for DAPI, Tubulin, Actin
- Uses triplet loss: same-MOA drugs = similar, different-MOA = different

**Output:**
- `./output/phase1/phase1_best_model.pth` (model checkpoint)
- `./output/phase1/phase1_losses.png` (training curves)

**Expected time:** 2-4 hours on GPU, 12-24 hours on CPU

**Check if it worked:**
- Loss should decrease over epochs
- Final loss should be < 0.3

### 2. Run Phase 2 (Generate Labels)
```bash
python phase2_generate_resistance_labels_UPDATED.py
```

**What it does:**
- Uses Phase 1 model to extract features
- Compares each cell to DMSO (baseline) and its MOA reference
- Generates resistance type labels

**Output:**
- `./output/phase2/resistance_labels.csv` (labels for training)
- `./output/phase2/resistance_distribution.png`
- `./output/phase2/similarity_distributions.png`

**Expected time:** 30-60 minutes

**Check if it worked:**
- CSV should have ~1000-2000 rows (for Week1-3)
- Should see distribution across all 5 resistance types
- DMSO should be mostly SENSITIVE

### 3. Run Phase 3 (Train Classifier)  
```bash
python phase3_resistance_classifier_train_UPDATED.py
```

**What it does:**
- Trains final resistance classifier using Phase 2 labels
- First 10 epochs: frozen encoder
- Next 30 epochs: fine-tune entire network

**Output:**
- `./output/phase3/phase3_best_model.pth` (final model!)
- `./output/phase3/confusion_matrix.png`
- `./output/phase3/classification_report.txt`

**Expected time:** 3-6 hours on GPU

**Check if it worked:**
- Accuracy should be > 70%
- Confusion matrix should show good diagonal

### 4. Run Inference (Demo!)
```bash
python lumimap_inference_UPDATED.py \
    --dapi ./data/Week1/Week1_22123/A01_s1_w1.tif \
    --tubulin ./data/Week1/Week1_22123/A01_s1_w2.tif \
    --actin ./data/Week1/Week1_22123/A01_s1_w4.tif \
    --drug taxol \
    --concentration 1.0 \
    --moa "Microtubule stabilizers" \
    --output demo_result.png
```

**Output:**
- Beautiful visualization with:
  - Original 3 channels
  - GradCAM heatmaps
  - Resistance prediction
  - Therapy recommendations

## Configuration (if needed)

All configs are in the script files. Key settings:

**If GPU memory issues:**
```python
# In phase1_..._UPDATED.py, line ~35
BATCH_SIZE = 16  # Reduce from 32
IMG_SIZE = (128, 128)  # Reduce from (256, 256)
```

**To use more data:**
```python
# In phase1_..._UPDATED.py, line ~55
WEEKS_TO_USE = ['Week1', 'Week2', 'Week3', 'Week4', 'Week5', 'Week6']
```

## Troubleshooting

**"File not found" error:**
- Check that CSVs are at `./data/BBBC021_v1_*.csv`
- Check that image folders are at `./data/Week1/Week1_22123/` etc.

**"CUDA out of memory":**
- Reduce `BATCH_SIZE` to 16 or 8
- Reduce `IMG_SIZE` to (128, 128)

**Phase 2 says "Model not found":**
- Make sure Phase 1 finished successfully
- Check that `./output/phase1/phase1_best_model.pth` exists

**Low accuracy:**
- Train longer (increase NUM_EPOCHS)
- Use more data (add more weeks)
- Check that Phase 1 converged (loss < 0.3)

## Expected Results

**Phase 1:**
- Train loss: ~0.2-0.3
- Val loss: ~0.25-0.35

**Phase 2:**  
- SENSITIVE: 40-50%
- PRIMARY_RESISTANCE: 20-30%
- PARTIAL_RESISTANCE: 10-20%
- CROSS_RESISTANCE: 10-15%
- UNCERTAIN: 5-15%

**Phase 3:**
- Overall accuracy: 75-85%
- Per-class F1: 0.7-0.9

## For Science Fair Demo

1. **Train all 3 phases first** (takes ~1 day total)
2. **Pick good example images** - find cells with clear phenotypes
3. **Run inference live** during presentation
4. **Show the visualizations** - especially GradCAM
5. **Explain the recommendations** - this is the coolest part!

## Quick Test (Before Full Training)

Test that everything is set up correctly:

```bash
# Just test Phase 1 for 1 epoch
python phase1_contrastive_moa_learning_UPDATED.py
# Edit line ~56 to: NUM_EPOCHS = 1
```

If this runs without errors, you're good to go!

## Questions?

Check the full README_LUMIMAP.md for detailed explanations.
