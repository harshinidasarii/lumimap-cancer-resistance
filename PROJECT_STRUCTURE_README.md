# 🗂️ LUMIMAP Project Structure

## New Organization (Post-Cleanup)

After running the organization script, your project has this clean structure:

```
sfproject/
│
├── 📁 phase1_training/              Phase 1: Contrastive Learning
│   ├── phase1_STRATEGIC_SAMPLING.py    ← Main training script (USE THIS)
│   ├── phase1_FAST.py                  ← Fast version for testing
│   ├── phase1_contrastive_moa_learning.py
│   └── phase1_contrastive_moa_learning_UPDATED.py
│
├── 📁 phase2_similarity/            Phase 2: Similarity Classification
│   ├── phase2_USE_STRATEGIC_MODEL.py   ← Main script (USE THIS)
│   ├── phase2_FAST.py                  ← Fast version
│   ├── phase2_generate_resistance_labels.py
│   └── phase2_generate_resistance_labels_UPDATED.py
│
├── 📁 phase3_classifier/            Phase 3: Binary Classifier (Optional)
│   ├── phase3_BINARY_CLASSIFIER.py
│   ├── phase3_FAST.py
│   ├── phase3_USE_STRATEGIC_IMBALANCED.py
│   ├── phase3_resistance_classifier_train.py
│   └── resistance_type_classifier_train.py
│
├── 📁 analysis_scripts/             Analysis & Metrics
│   ├── analysis_ablation.py            ← Ablation study
│   ├── analysis_embedding_space.py     ← t-SNE/PCA visualization
│   ├── analysis_gradcam.py             ← Generate GradCAM examples
│   ├── analysis_moa_performance.py     ← Per-MOA metrics
│   ├── get_accuracy.py                 ← Quick accuracy check
│   ├── get_all_metrics.py              ← Complete metrics report
│   ├── create_visualizations.py        ← Generate all charts
│   ├── run_all_analyses.py             ← Run everything
│   └── test_gradcam_quick.py           ← Test GradCAM
│
├── 📁 old_deprecated/               Old/Unused Files (DELETE LATER)
│   ├── batch-output.py
│   ├── batch_predict.py
│   ├── gradcam_explainer.py            ← Has architecture mismatch
│   ├── resistance_detector_inference.py ← Has architecture mismatch
│   ├── demo_complete.py                ← Old demo version
│   └── (many other deprecated files)
│
├── 📁 documentation/                All Documentation Files
│   ├── README.md                       ← Main README
│   ├── LUMIMAP_README.md               ← Complete guide
│   ├── START_HERE.md                   ← Quick start
│   ├── implementation_guide.md         ← Full implementation
│   ├── FAST_VS_FULL_TRAINING.md
│   └── (other .md files)
│
├── 📁 data/                         Dataset (Unchanged)
│   ├── BBBC021_v1_image.csv
│   ├── BBBC021_v1_compound.csv
│   ├── BBBC021_v1_moa.csv
│   └── Week1/, Week2/, ... (image folders)
│
├── 📁 output/                       Results (Unchanged)
│   ├── phase1_strategic/               ← Phase 1 trained model
│   ├── phase2_strategic/               ← Phase 2 results
│   └── demo_results/                   ← Demo visualizations
│
├── 📁 venv/                         Virtual Environment
│
└── 🐍 DEMO SCRIPTS (Main Folder)    ← FOR SCIENCE FAIR
    ├── demo_with_gradcam.py            ← Main demo script ⭐
    ├── find_resistant_samples.py       ← Find demo indices ⭐
    ├── find_ALL_resistance_types.py    ← All 4 types ⭐
    ├── show_input_files.py             ← Show input images ⭐
    ├── view_csv.py                     ← View metadata ⭐
    ├── BBBC021_v1_image.csv            ← Metadata file
    └── README.md                       ← Main README

```

---

## 🎯 Quick Reference

### **For Science Fair Demo:**
All demo scripts are in the **MAIN FOLDER** - easy to access!

```bash
# Stay in main project folder
cd ~/harshu-repo/sfproject

# Run any demo command
python demo_with_gradcam.py --idx 36
python find_resistant_samples.py
python show_input_files.py --idx 36
python view_csv.py --idx 36
```

---

### **For Training (Already Done):**

```bash
# Phase 1: Train contrastive learning model
cd phase1_training
python phase1_STRATEGIC_SAMPLING.py

# Phase 2: Generate resistance labels
cd ../phase2_similarity
python phase2_USE_STRATEGIC_MODEL.py
```

---

### **For Analysis:**

```bash
# Run all analyses
cd analysis_scripts
python run_all_analyses.py

# Or individual analyses
python get_all_metrics.py
python analysis_gradcam.py
python analysis_moa_performance.py
```

---

## 📋 File Count by Category

| Folder | File Count | Purpose |
|--------|------------|---------|
| **phase1_training/** | 4 files | Contrastive learning training |
| **phase2_similarity/** | 4 files | Similarity calculation & classification |
| **phase3_classifier/** | 5 files | Binary classifier (optional) |
| **analysis_scripts/** | 9 files | Metrics, visualizations, analysis |
| **old_deprecated/** | ~25 files | Old/unused code (DELETE LATER) |
| **documentation/** | ~16 files | All .md documentation |
| **Main folder (demos)** | 5 files | Science fair demo scripts ⭐ |

**Total: ~68 files organized into 6 folders + 5 demo files in main**

---

## 🗑️ What Can Be Deleted

### **Safe to Delete (in old_deprecated/):**

- All batch processing scripts (old approach)
- Test scripts (test_*.py)
- Deprecated inference scripts (gradcam_explainer.py, resistance_detector_inference.py)
- Old demo versions (demo_complete.py, demo_predict.py)
- Morphology analyzer (not used in final version)
- Drug database scripts (replaced by simpler MOA alternatives)

**When to delete:** After confirming everything works with the new structure.

---

## ✅ What Stays in Main Folder

**Only 5 demo scripts + data files:**

1. **demo_with_gradcam.py** - Main demo (runs GradCAM visualization)
2. **find_resistant_samples.py** - Find demo indices
3. **find_ALL_resistance_types.py** - All 4 resistance types
4. **show_input_files.py** - Show input image filenames
5. **view_csv.py** - View metadata

**Plus:**
- BBBC021_v1_image.csv (metadata)
- README.md (main documentation)
- requirements.txt
- .gitignore

---

## 🎓 For Science Fair

**Your main folder is now CLEAN and contains only what you need for demos!**

```bash
# All commands run from main folder
python find_resistant_samples.py          # Show all options
python demo_with_gradcam.py --idx 36      # CROSS-RESISTANCE demo
python show_input_files.py --idx 36       # See input images
python view_csv.py --idx 36

python demo_with_gradcam.py --idx 41      # PARTIAL-RESISTANCE demo
python show_input_files.py --idx 41       # See input images
python view_csv.py --idx 41

python demo_with_gradcam.py --idx 1     # SENSITIVE demo
python show_input_files.py --idx 1       # See input images
python view_csv.py --idx 1

```

**No need to navigate to other folders for demos!** ✅

---

## 🔧 How to Run the Organization Script

**On your Mac:**

```bash
# Navigate to project
cd ~/harshu-repo/sfproject

# Make script executable
chmod +x organize_project.sh

# Run the script
./organize_project.sh
```

**The script will:**
1. Create 6 new folders
2. Move files to appropriate locations
3. Keep demo scripts in main folder
4. Show you the new structure

**Time: ~5 seconds**

---

## 📊 Before vs After

### **BEFORE (Cluttered):**
```
89 files mixed together in main folder
Hard to find what you need
Confusing for science fair
```

### **AFTER (Organized):**
```
Main folder: 5 demo scripts (clean!)
6 organized folders by purpose
Easy to navigate
Professional structure
```

---

## 💡 Benefits

**For Science Fair:**
- ✅ Main folder has ONLY demo scripts
- ✅ Easy to find and run commands
- ✅ Professional organization
- ✅ Clean workspace

**For Development:**
- ✅ Phase 1/2/3 scripts separated
- ✅ Analysis tools in one place
- ✅ Documentation centralized
- ✅ Old files isolated (can delete)

**For Understanding:**
- ✅ Clear what each folder does
- ✅ Easy to see project structure
- ✅ Simple to explain to others

---

## 🚀 Next Steps

1. **Run the organization script** (5 seconds)
2. **Verify demo scripts still work** (run one demo)
3. **Review old_deprecated folder** (decide what to delete)
4. **Use the clean main folder for science fair!**

---

## ⚠️ Important Notes

**Demo scripts DON'T need to be modified** - they still work the same way!

**Output folders unchanged** - ./output/ stays exactly where it is

**Data folder unchanged** - ./data/ stays exactly where it is

**Virtual environment unchanged** - ./venv/ stays exactly where it is

**Only organizational change** - files moved to folders, nothing else changed!

---

## 📞 Quick Help

**If something doesn't work after organizing:**

```bash
# Check if demo works
python demo_with_gradcam.py --idx 36

# If you get import errors, you might need to adjust paths
# But this shouldn't happen since demos are still in main folder
```

**All demo scripts remain in the SAME location (main folder), so they should work exactly as before!**

---

✅ **Your project is now professionally organized and ready for science fair!** 🎉
