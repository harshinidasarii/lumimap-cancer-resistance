# Changes Made for Your SFPROJECT Data Structure

## What Was Wrong Before

The original code assumed:
```
❌ CSV files inside each week folder
❌ DATA_DIR needed different handling
❌ Output to ./outputs/ (not ./output/)
```

## What's Fixed Now

✅ **CSV Paths:** All CSVs are at `./data/BBBC021_v1_*.csv`
✅ **Data Directory:** Set to `./data` (where your Week folders are)
✅ **Output Directory:** Changed to `./output/phase1`, `./output/phase2`, etc. to match your structure
✅ **Image Paths:** Work correctly with your folder structure

## Your Exact Structure

```
SFPROJECT/
├── data/
│   ├── Week1/Week1_22123/  ← Images here
│   ├── Week2/
│   ├── Week3/
│   ├── BBBC021_v1_compound.csv  ← CSVs here (shared by all weeks)
│   ├── BBBC021_v1_image.csv
│   └── BBBC021_v1_moa.csv
├── input/
└── output/  ← Models will be saved here
```

## Files to Use (NEW = Updated)

### ✅ Use These (Updated for Your Structure):
1. **phase1_contrastive_moa_learning_UPDATED.py**
   - Paths: `./data/BBBC021_v1_*.csv`
   - Output: `./output/phase1/`
   
2. **phase2_generate_resistance_labels_UPDATED.py**
   - Reads: `./data/BBBC021_v1_*.csv`
   - Loads: `./output/phase1/phase1_best_model.pth`
   - Output: `./output/phase2/resistance_labels.csv`

3. **quick_start_guide.md**
   - Step-by-step instructions
   - Troubleshooting
   - Expected results

### ❌ Don't Use These (Old Versions):
- phase1_contrastive_moa_learning.py (without _UPDATED)
- phase2_generate_resistance_labels.py (without _UPDATED)
- Any files that reference `./outputs/` instead of `./output/`

## Key Code Changes

### Config Changes:
```python
# OLD
DATA_DIR = './data'
IMAGE_CSV = 'BBBC021_v1_image.csv'  # Wrong!
OUTPUT_DIR = './outputs/phase1'      # Wrong!

# NEW  
DATA_DIR = './data'
IMAGE_CSV = './data/BBBC021_v1_image.csv'  # Correct!
OUTPUT_DIR = './output/phase1'              # Correct!
```

### Path Construction:
```python
# This works because CSV has paths like "Week1/Week1_22123"
# And your folders are at "./data/Week1/Week1_22123/"

dapi_path = self.root_dir / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
# Becomes: ./data/Week1/Week1_22123/A01_s1_w1.tif ✓
```

## What to Do Now

1. **Delete or ignore old files** (ones without `_UPDATED`)
2. **Use the new `_UPDATED` versions**
3. **Follow quick_start_guide.md**
4. **Run Phase 1 → Phase 2 → Phase 3**

## Testing Your Setup

Quick test to make sure paths are correct:

```python
import pandas as pd
from pathlib import Path

# Test CSV loading
image_df = pd.read_csv('./data/BBBC021_v1_image.csv')
moa_df = pd.read_csv('./data/BBBC021_v1_moa.csv')
print(f"✓ Loaded {len(image_df)} images")
print(f"✓ Loaded {len(moa_df)} MOA entries")

# Test image path
row = image_df.iloc[0]
dapi_path = Path('./data') / row['Image_PathName_DAPI'] / row['Image_FileName_DAPI']
print(f"✓ Example path: {dapi_path}")
print(f"✓ File exists: {dapi_path.exists()}")
```

If all checks pass (✓), you're ready to train!

## Expected Training Time

With Week1-3 data (~1000-2000 images):
- **Phase 1:** 2-4 hours (GPU) or 12-24 hours (CPU)
- **Phase 2:** 30-60 minutes  
- **Phase 3:** 3-6 hours (GPU)

**Total:** About 1 day of training time

## Need Help?

1. Read `quick_start_guide.md` for step-by-step instructions
2. Check `README_LUMIMAP.md` for detailed explanations
3. Look at error messages carefully - they usually point to the problem!

Good luck with the science fair! 🚀
