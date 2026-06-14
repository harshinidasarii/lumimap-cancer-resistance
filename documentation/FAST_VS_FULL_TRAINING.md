# Fast vs Full Training - Which Should You Use?

## Quick Comparison

| Feature | **FAST Training** ⚡ | **FULL Training** 🎯 |
|---------|---------------------|---------------------|
| **Total Time** | 2-3 hours | 12-24 hours |
| **Data Used** | Week1 only (50%) | Week1-3 (100%) |
| **Image Size** | 128x128 | 256x256 |
| **Model** | MobileNetV2 (lighter) | ResNet18 (better) |
| **Epochs** | 15 | 30 |
| **Accuracy** | ~70-75% | ~80-85% |
| **Best For** | Quick testing, demos | Science fair, publication |

## 🏃 FAST Training (Recommended to Start!)

### Use this when:
- ✅ First time running the code (verify it works!)
- ✅ Testing/debugging
- ✅ Quick demo for science fair practice
- ✅ Limited time (need results today)
- ✅ No GPU available

### Files to use:
```bash
phase1_FAST.py
phase2_FAST.py  
phase3_FAST.py
```

### Expected timeline:
```
Phase 1 FAST: 30-60 minutes (GPU) or 2-3 hours (CPU)
Phase 2 FAST: 10-15 minutes
Phase 3 FAST: 45-90 minutes (GPU)
────────────────────────────
TOTAL: 2-3 hours (GPU) or 6-8 hours (CPU)
```

### What you sacrifice:
- Slightly lower accuracy (~5-10%)
- Less training data (Week1 only)
- Smaller model (fewer parameters)

### What you still get:
- ✅ All 5 resistance types detected
- ✅ Channel-wise GradCAM works
- ✅ Therapy recommendations work
- ✅ Multi-type resistance classification
- ✅ Good enough for science fair demo!

## 🎯 FULL Training (For Final Results)

### Use this when:
- ✅ FAST training worked and you want best accuracy
- ✅ Preparing for actual science fair presentation
- ✅ Want publication-quality results
- ✅ Have overnight to train
- ✅ Have good GPU

### Files to use:
```bash
phase1_contrastive_moa_learning_UPDATED.py
phase2_generate_resistance_labels_UPDATED.py
phase3_resistance_classifier_train_UPDATED.py
```

### Expected timeline:
```
Phase 1 FULL: 2-4 hours (GPU) or 12-24 hours (CPU)
Phase 2 FULL: 30-60 minutes
Phase 3 FULL: 3-6 hours (GPU)
────────────────────────────
TOTAL: 6-10 hours (GPU) or 18-30 hours (CPU)
```

### What you get:
- ✅ Best possible accuracy
- ✅ More robust model
- ✅ Trained on more data
- ✅ Better generalization
- ✅ Publication-ready

## 📋 Recommended Workflow

### Step 1: Start with FAST (Today)
```bash
# Run FAST training to make sure everything works
python phase1_FAST.py  # 30-60 min
python phase2_FAST.py  # 10-15 min  
python phase3_FAST.py  # 45-90 min

# Test inference
python lumimap_inference_FAST.py \
    --dapi data/Week1/.../A01_s1_w1.tif \
    --tubulin data/Week1/.../A01_s1_w2.tif \
    --actin data/Week1/.../A01_s1_w4.tif \
    --drug taxol \
    --concentration 1.0
```

**✓ If this works → Your setup is correct!**

### Step 2: Run FULL training (Overnight)
```bash
# Once FAST works, run FULL for best results
# Start before going to bed
nohup python phase1_contrastive_moa_learning_UPDATED.py > phase1.log 2>&1 &
# Check progress: tail -f phase1.log

# Next day:
python phase2_generate_resistance_labels_UPDATED.py
python phase3_resistance_classifier_train_UPDATED.py
```

### Step 3: Compare Results
```bash
# FAST results
./output/phase1_fast/phase1_fast_best.pth  (~70-75% accuracy)

# FULL results  
./output/phase1/phase1_best_model.pth      (~80-85% accuracy)

# Use FULL model for science fair!
```

## ⚙️ What Makes FAST Fast?

### 1. Smaller Images (4x faster)
```python
# FAST
IMG_SIZE = (128, 128)  # 16,384 pixels

# FULL
IMG_SIZE = (256, 256)  # 65,536 pixels (4x more!)
```

### 2. Lighter Model (~3x faster)
```python
# FAST
BACKBONE = 'mobilenet_v2'  # 3.5M parameters

# FULL  
BACKBONE = 'resnet18'      # 11.7M parameters
```

### 3. Less Data (2x faster)
```python
# FAST
WEEKS_TO_USE = ['Week1']           # ~800 images
SUBSAMPLE_FRACTION = 0.5           # Use 50%

# FULL
WEEKS_TO_USE = ['Week1', 'Week2', 'Week3']  # ~2000 images
# Uses 100% of data
```

### 4. Fewer Epochs (2x faster)
```python
# FAST
NUM_EPOCHS = 15

# FULL
NUM_EPOCHS = 30
```

### 5. Mixed Precision (1.5x faster on GPU)
```python
# FAST
USE_AMP = True  # FP16 math (faster on modern GPUs)

# FULL
# Standard FP32 (more precise)
```

**Total speedup: 4 × 3 × 2 × 2 × 1.5 = ~72x faster!**

## 🎓 For Science Fair

### Practice Run (Use FAST):
- Run FAST training to practice your demo
- Show the visualizations
- Test the inference pipeline
- Make sure you understand the system

### Final Presentation (Use FULL):
- Train FULL model overnight before the fair
- Use FULL model for live demos
- Show higher accuracy in your poster
- Mention you tried both approaches!

## 💡 Pro Tips

1. **Start with FAST**
   - Makes sure code works
   - Gets you results quickly
   - Good for debugging

2. **Switch to FULL later**
   - Once you know it works
   - Run overnight
   - Better final results

3. **You can use both!**
   - FAST for practice demos
   - FULL for actual competition
   - Compare them in your poster!

## 🆘 If FAST Still Too Slow

Edit `phase1_FAST.py` and reduce further:

```python
# Line ~35-40
IMG_SIZE = (64, 64)          # Even smaller!
BATCH_SIZE = 128             # Bigger batches
NUM_EPOCHS = 10              # Fewer epochs
SUBSAMPLE_FRACTION = 0.25    # Use only 25% of data
NUM_WORKERS = 8              # More CPU cores (if available)
```

This will be EVEN FASTER but with lower accuracy (~65-70%)

## ✅ Validation

Both FAST and FULL should show:
- ✅ Loss decreasing over epochs
- ✅ Val loss < Train loss (not overfitting)
- ✅ Final loss < 0.5
- ✅ Confusion matrix with strong diagonal
- ✅ All 5 resistance types detected

Good luck! 🚀
