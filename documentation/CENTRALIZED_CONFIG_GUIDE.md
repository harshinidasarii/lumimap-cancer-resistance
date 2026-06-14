# 🎯 Centralized Test Configuration Guide

## ✅ PROBLEM SOLVED!

Both `resistance_detector_inference.py` and `gradcam_explainer.py` now use **the same configuration file** to ensure they always analyze the same image with the same metadata.

---

## 📁 **FILES:**

1. **test_config.py** - Central configuration (EDIT THIS ONE!)
2. **resistance_detector_inference.py** - Imports from test_config
3. **gradcam_explainer.py** - Imports from test_config

---

## 🔧 **HOW TO USE:**

### **Step 1: Edit `test_config.py`**

Open `test_config.py` and choose your test image:

```python
# Option 1: Week2 cytochalasin B (92.26% resistance)
TEST_IMAGE = 'data/Week2/Week2_24141/Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif'
DRUG_NAME = 'cytochalasin B'
DRUG_CONCENTRATION = '10 nM'
EXPECTED_MOA = 'Actin disruptors'

# Option 2: Week1 image (100% resistance)
# TEST_IMAGE = 'data/Week1/Week1_22123/Week1_150607_B02_s4_w2EE226363-0BAC-443F-A41C-16C9C89FDDFA.tif'
# DRUG_NAME = 'cytochalasin B'
# DRUG_CONCENTRATION = '10 nM'
# EXPECTED_MOA = 'Actin disruptors'
```

**To switch images:**
- Uncomment the image you want (remove `#`)
- Comment out the other options (add `#` at the start of each line)

### **Step 2: Validate Configuration (Optional)**

Test your configuration:
```bash
python test_config.py
```

You should see:
```
✓ Configuration validated:
  Image: Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif
  Drug: cytochalasin B (10 nM)
  Expected MOA: Actin disruptors
```

### **Step 3: Run Both Scripts**

Now run both scripts - they'll automatically use the same configuration:

```bash
python resistance_detector_inference.py
python gradcam_explainer.py
```

**Both will show:**
```
✓ Using test configuration:
  Image: Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif
  Drug: cytochalasin B (10 nM)
  Expected MOA: Actin disruptors
```

---

## ✅ **BENEFITS:**

### **Before (Manual Configuration):**
❌ Edit `resistance_detector_inference.py` → set image path, drug name
❌ Edit `gradcam_explainer.py` → set different image path, drug name  
❌ Results don't match! 😱

### **After (Centralized Configuration):**
✅ Edit `test_config.py` ONCE
✅ Both scripts automatically sync
✅ Results always match! 🎯

---

## 🎨 **ADDING NEW TEST IMAGES:**

### **Example: Add cisplatin test image**

Edit `test_config.py`:

```python
# Option 3: Week3 cisplatin (DNA damage test)
# TEST_IMAGE = 'data/Week3/Week3_cisplatin_image.tif'
# DRUG_NAME = 'cisplatin'
# DRUG_CONCENTRATION = '5 µM'
# EXPECTED_MOA = 'DNA damage'
```

Then uncomment to activate!

---

## 🔍 **WHAT GETS SYNCED:**

The following values are automatically shared between both scripts:

1. **TEST_IMAGE** - Path to the test image file
2. **DRUG_NAME** - Name of the drug (e.g., 'cytochalasin B')
3. **DRUG_CONCENTRATION** - Concentration (e.g., '10 nM')
4. **EXPECTED_MOA** - Expected MOA class (e.g., 'Actin disruptors')

---

## 📊 **VERIFICATION CHECKLIST:**

After running both scripts, verify they show:

- [ ] Same image filename
- [ ] Same drug name
- [ ] Same concentration
- [ ] Same expected MOA
- [ ] Same predicted MOA
- [ ] Same resistance score
- [ ] Same confidence percentages

If all match → **Perfect!** 🎉

---

## 🚀 **TYPICAL WORKFLOW:**

1. Edit `test_config.py` to select your test image
2. Run `python test_config.py` to validate (optional)
3. Run `python resistance_detector_inference.py`
4. Run `python gradcam_explainer.py`
5. Compare outputs - they should be identical!

---

## 💡 **TIPS:**

- **Keep all test images commented except ONE** to avoid confusion
- **Add descriptive comments** for each test image (drug, resistance score, etc.)
- **Validate config before running** if you're unsure about the path
- **Never edit image paths directly** in the analysis scripts anymore!

---

## 🎯 **FOR YOUR PRESENTATION:**

Now you can confidently say:

> "Both our resistance detector and explainability system (Grad-CAM) analyze 
> the same test image with identical parameters, ensuring complete consistency 
> and reproducibility in our results."

---

**No more mismatches! Everything stays in sync!** 🎊
