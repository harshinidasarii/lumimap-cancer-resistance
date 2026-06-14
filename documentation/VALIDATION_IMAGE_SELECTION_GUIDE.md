# 📁 HOW TO SELECT YOUR 5 VALIDATION IMAGES

## 🎯 STEP 1: FIND IMAGES IN YOUR DATASET

You should already have TIFF files in your `data/` folder. Let's find them!

### **Check what you have:**

```bash
# List all Week folders
ls -la data/

# List files in Week1
ls -la data/Week1/Week1_*/

# List files in Week2  
ls -la data/Week2/Week2_*/

# Count total TIFF files
find data/ -name "*.tif" | wc -l
```

---

## 🔍 STEP 2: SELECT DIVERSE TEST IMAGES

**For best validation, choose images that represent:**

1. **Different drugs** (if you have them)
   - Example: cytochalasin B, paclitaxel, cisplatin, etc.

2. **Different MOA classes**
   - Actin disruptors
   - Microtubule stabilizers
   - DNA damage
   - etc.

3. **Different weeks/batches** (if applicable)
   - Week1 vs Week2 vs Week3
   - Different experimental conditions

4. **Different concentrations** (if labeled)
   - 10 nM vs 100 nM vs 1 µM

5. **Different wells/replicates**
   - B02, C03, D04, etc.

---

## 📋 STEP 3: UPDATE THE VALIDATION SCRIPT

Open `batch_validation.py` and find this section (around line 50):

```python
VALIDATION_IMAGES = [
    # Example 1: Week2 cytochalasin B
    ('data/Week2/Week2_24141/Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    # Example 2: Week1 image
    ('data/Week1/Week1_22123/Week1_150607_B02_s4_w2EE226363-0BAC-443F-A41C-16C9C89FDDFA.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    # Add your 3 more images here:
    # ('path/to/your/image3.tif', 'drug_name', 'expected_MOA'),
    # ('path/to/your/image4.tif', 'drug_name', 'expected_MOA'),
    # ('path/to/your/image5.tif', 'drug_name', 'expected_MOA'),
]
```

**Replace with YOUR image paths!**

---

## 💡 EXAMPLE: SELECTING 5 DIVERSE IMAGES

### **Option A: All Same Drug (cytochalasin B)**
If you only have cytochalasin B images:

```python
VALIDATION_IMAGES = [
    # Different wells/replicates of cytochalasin B
    ('data/Week2/Week2_24141/Week2_180607_B02_s2_w4*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week2/Week2_24141/Week2_180607_C03_s1_w4*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week1/Week1_22123/Week1_150607_B02_s4_w2*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week1/Week1_22123/Week1_150607_D04_s3_w2*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week2/Week2_24141/Week2_180607_E05_s2_w4*.tif',
     'cytochalasin B', 'Actin disruptors'),
]
```

### **Option B: Different Drugs (if you have them)**
If you have multiple drug types:

```python
VALIDATION_IMAGES = [
    # Actin disruptor
    ('data/Week2/actin/image1.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    # Microtubule stabilizer
    ('data/Week2/microtubule/image2.tif',
     'paclitaxel', 'Microtubule stabilizers'),
    
    # DNA damage
    ('data/Week2/dna/image3.tif',
     'cisplatin', 'DNA damage'),
    
    # Control
    ('data/Week2/control/image4.tif',
     'DMSO', 'DMSO'),
    
    # Another actin disruptor
    ('data/Week2/actin/image5.tif',
     'latrunculin B', 'Actin disruptors'),
]
```

---

## 🔎 STEP 4: DECODE YOUR FILENAME

BBBC021 filenames follow this pattern:
```
Week1_150607_B02_s4_w2EE226363-0BAC-443F-A41C-16C9C89FDDFA.tif
│     │      │   │  │  │
│     │      │   │  │  └─ UUID (unique identifier)
│     │      │   │  └──── Channel (w1=Hoechst, w2=ER, w4=actin, w5=tubulin)
│     │      │   └─────── Site/field number
│     │      └─────────── Well (B02, C03, etc.)
│     └────────────────── Date (YYMMDD)
└──────────────────────── Week number
```

**Channel meanings:**
- `w1` = Hoechst (DNA/nuclei)
- `w2` = ER (endoplasmic reticulum)
- `w4` = Actin (phalloidin staining)
- `w5` = Tubulin (microtubules)

**Choose w4 (actin) or w5 (tubulin) for best morphology!**

---

## 🚀 STEP 5: RUN THE VALIDATION

```bash
python batch_validation.py
```

**Expected output:**
```
======================================================================
BATCH VALIDATION REPORT
======================================================================
Number of images: 5
Model: ResNet-50 (13 MOA classes)
Device: mps

[1/5] Processing: Week2_180607_B02_s2_w4ED129C44...
  Drug: cytochalasin B
  Expected: Actin disruptors (7.74%)
  Predicted: Microtubule stabilizers (83.62%)
  Resistance: 92.26%

[2/5] Processing: Week1_150607_B02_s4_w2EE226363...
  Drug: cytochalasin B
  Expected: Actin disruptors (0.00%)
  Predicted: Microtubule stabilizers (99.97%)
  Resistance: 100.00%

... (3 more images)

======================================================================
SUMMARY STATISTICS
======================================================================

Top-1 Accuracy: 40.0% (2/5)
Average Resistance Score: 85.42%

Resistance Distribution:
  Sensitive (<30%):        1 (20.0%)
  Partial (30-70%):        1 (20.0%)
  Resistant (>70%):        3 (60.0%)

✓ Visual report saved: batch_validation_report_20260226_203045.png
✓ JSON results saved: batch_validation_results_20260226_203045.json
```

---

## 📊 YOU'LL GET 3 OUTPUTS:

1. **Visual Report (PNG)** - Bar charts for each image
2. **JSON Results** - Machine-readable data
3. **Console Summary** - Statistics and metrics

---

## 💡 TIPS FOR SELECTING IMAGES:

### **Good Selection:**
✅ Mix of resistant and sensitive samples
✅ Different experimental conditions
✅ Both Week1 and Week2 (if available)
✅ Different wells (biological replicates)
✅ Same channel type (all w4 or all w5)

### **Poor Selection:**
❌ All from same well
❌ All from same experimental run
❌ Mixed channels (w1, w2, w4, w5 together)
❌ All expected to give same result

---

## 🔍 QUICK COMMAND TO LIST YOUR IMAGES:

```bash
# Find all TIF files and show first 10
find data/ -name "*.tif" | head -10

# Find all w4 (actin) channel images
find data/ -name "*w4*.tif"

# Find all w5 (tubulin) channel images  
find data/ -name "*w5*.tif"

# Count images by channel
echo "w4 (actin): $(find data/ -name '*w4*.tif' | wc -l)"
echo "w5 (tubulin): $(find data/ -name '*w5*.tif' | wc -l)"
```

---

## ❓ DON'T KNOW WHICH DRUG IS IN WHICH IMAGE?

**Check the BBBC021 dataset documentation!**

The dataset usually comes with:
- `image.csv` - Maps images to compounds
- `compound.csv` - Lists compound names and MOAs
- `moa.csv` - Lists MOA classes

Look for these files in your `data/` folder.

**Or**, just use cytochalasin B for all 5 images if that's what you have!

---

## 🎯 RECOMMENDED 5-IMAGE VALIDATION SET:

If you only have cytochalasin B data:

```python
VALIDATION_IMAGES = [
    # 5 different wells/sites of cytochalasin B
    # This tests consistency across replicates
    
    ('data/Week2/Week2_24141/Week2_180607_B02_s2_w4*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week2/Week2_24141/Week2_180607_B03_s1_w4*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week2/Week2_24141/Week2_180607_C02_s3_w4*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week1/Week1_22123/Week1_150607_B02_s4_w2*.tif',
     'cytochalasin B', 'Actin disruptors'),
    
    ('data/Week1/Week1_22123/Week1_150607_C03_s2_w2*.tif',
     'cytochalasin B', 'Actin disruptors'),
]
```

---

**Now you can validate on multiple images and get comprehensive statistics!** 🎉
