#!/bin/bash

# LUMIMAP Project Organization Script
# ====================================
# This script organizes all files into proper folders

echo "=========================================="
echo "🗂️  ORGANIZING LUMIMAP PROJECT"
echo "=========================================="
echo ""

# Create directory structure
echo "📁 Creating folder structure..."
mkdir -p phase1_training
mkdir -p phase2_similarity
mkdir -p phase3_classifier
mkdir -p analysis_scripts
mkdir -p old_deprecated
mkdir -p documentation

echo "✓ Folders created"
echo ""

# ==========================================
# PHASE 1: Training Files
# ==========================================
echo "📦 Moving Phase 1 files..."
mv phase1_FAST.py phase1_training/ 2>/dev/null
mv phase1_STRATEGIC_SAMPLING.py phase1_training/ 2>/dev/null
mv phase1_contrastive_moa_learning.py phase1_training/ 2>/dev/null
mv phase1_contrastive_moa_learning_UPDATED.py phase1_training/ 2>/dev/null
echo "✓ Phase 1 files moved"

# ==========================================
# PHASE 2: Similarity/Classification Files
# ==========================================
echo "📦 Moving Phase 2 files..."
mv phase2_FAST.py phase2_similarity/ 2>/dev/null
mv phase2_USE_STRATEGIC_MODEL.py phase2_similarity/ 2>/dev/null
mv phase2_generate_resistance_labels.py phase2_similarity/ 2>/dev/null
mv phase2_generate_resistance_labels_UPDATED.py phase2_similarity/ 2>/dev/null
echo "✓ Phase 2 files moved"

# ==========================================
# PHASE 3: Binary Classifier Files
# ==========================================
echo "📦 Moving Phase 3 files..."
mv phase3_BINARY_CLASSIFIER.py phase3_classifier/ 2>/dev/null
mv phase3_FAST.py phase3_classifier/ 2>/dev/null
mv phase3_USE_STRATEGIC_IMBALANCED.py phase3_classifier/ 2>/dev/null
mv phase3_resistance_classifier_train.py phase3_classifier/ 2>/dev/null
mv resistance_type_classifier_train.py phase3_classifier/ 2>/dev/null
echo "✓ Phase 3 files moved"

# ==========================================
# ANALYSIS Scripts
# ==========================================
echo "📦 Moving analysis scripts..."
mv analysis_ablation.py analysis_scripts/ 2>/dev/null
mv analysis_embedding_space.py analysis_scripts/ 2>/dev/null
mv analysis_gradcam.py analysis_scripts/ 2>/dev/null
mv analysis_moa_performance.py analysis_scripts/ 2>/dev/null
mv get_accuracy.py analysis_scripts/ 2>/dev/null
mv get_all_metrics.py analysis_scripts/ 2>/dev/null
mv create_visualizations.py analysis_scripts/ 2>/dev/null
mv run_all_analyses.py analysis_scripts/ 2>/dev/null
mv test_gradcam_quick.py analysis_scripts/ 2>/dev/null
echo "✓ Analysis files moved"

# ==========================================
# OLD/DEPRECATED Files
# ==========================================
echo "📦 Moving old/deprecated files..."
mv batch-output.py old_deprecated/ 2>/dev/null
mv batch_predict.py old_deprecated/ 2>/dev/null
mv batch_validation.py old_deprecated/ 2>/dev/null
mv check_checkpoint.py old_deprecated/ 2>/dev/null
mv check_img_size.py old_deprecated/ 2>/dev/null
mv compare_models.py old_deprecated/ 2>/dev/null
mv compare_preprocessing.py old_deprecated/ 2>/dev/null
mv discover_images.py old_deprecated/ 2>/dev/null
mv enhanced_batch_processor.py old_deprecated/ 2>/dev/null
mv gradcam_explainer.py old_deprecated/ 2>/dev/null
mv gradcam_visualization.py old_deprecated/ 2>/dev/null
mv load_checkpoint_model.py old_deprecated/ 2>/dev/null
mv lumimap_inference.py old_deprecated/ 2>/dev/null
mv moa_classifier_train.py old_deprecated/ 2>/dev/null
mv resistance_detector_inference.py old_deprecated/ 2>/dev/null
mv resistance_inference.py old_deprecated/ 2>/dev/null
mv test_config.py old_deprecated/ 2>/dev/null
mv test_image_sizes.py old_deprecated/ 2>/dev/null
mv test_model_load.py old_deprecated/ 2>/dev/null
mv morphology_analyzer.py old_deprecated/ 2>/dev/null
mv drug_database.py old_deprecated/ 2>/dev/null
mv therapy_recommendation_database.py old_deprecated/ 2>/dev/null
mv view_drug_database.py old_deprecated/ 2>/dev/null
mv demo_complete.py old_deprecated/ 2>/dev/null
mv demo_predict.py old_deprecated/ 2>/dev/null
mv SYSTEM_WORKFLOW.py old_deprecated/ 2>/dev/null

# Move old output images
mv gradcam_explanation.png old_deprecated/ 2>/dev/null
mv resistance_report.json old_deprecated/ 2>/dev/null
mv resistance_report.png old_deprecated/ 2>/dev/null
echo "✓ Old files moved"

# ==========================================
# DOCUMENTATION Files
# ==========================================
echo "📦 Moving documentation files..."
mv CENTRALIZED_CONFIG_GUIDE.md documentation/ 2>/dev/null
mv CHANGES_FOR_YOUR_DATA_STRUCTURE.md documentation/ 2>/dev/null
mv CHECKPOINT_FIX_README.md documentation/ 2>/dev/null
mv FAST_VS_FULL_TRAINING.md documentation/ 2>/dev/null
mv FIXES_SUMMARY.md documentation/ 2>/dev/null
mv LUMIMAP_README.md documentation/ 2>/dev/null
mv MORPHOLOGY_BUG_EXPLANATION.md documentation/ 2>/dev/null
mv MORPHOLOGY_INTEGRATION_COMPLETE.md documentation/ 2>/dev/null
mv PHASE3_PROBLEM_AND_SOLUTIONS.md documentation/ 2>/dev/null
mv README_LUMIMAP.md documentation/ 2>/dev/null
mv START_HERE.md documentation/ 2>/dev/null
mv VALIDATION_IMAGE_SELECTION_GUIDE.md documentation/ 2>/dev/null
mv cancer_resistance_data_requirements.md documentation/ 2>/dev/null
mv implementation_guide.md documentation/ 2>/dev/null
mv overallproject.md documentation/ 2>/dev/null
mv quick_start_guide.md documentation/ 2>/dev/null
echo "✓ Documentation moved"

# ==========================================
# DEMO FILES - STAY IN MAIN FOLDER
# ==========================================
echo ""
echo "✓ Demo files staying in main folder:"
echo "  • demo_with_gradcam.py"
echo "  • find_resistant_samples.py"
echo "  • find_ALL_resistance_types.py"
echo "  • show_input_files.py"
echo "  • view_csv.py"

# ==========================================
# CLEANUP
# ==========================================
echo ""
echo "=========================================="
echo "✅ ORGANIZATION COMPLETE!"
echo "=========================================="
echo ""
echo "📂 New folder structure:"
echo ""
echo "sfproject/"
echo "├── 📁 phase1_training/          (Phase 1 contrastive learning)"
echo "├── 📁 phase2_similarity/        (Phase 2 similarity classification)"
echo "├── 📁 phase3_classifier/        (Phase 3 binary classifier)"
echo "├── 📁 analysis_scripts/         (Analysis & metrics)"
echo "├── 📁 old_deprecated/           (Can delete later)"
echo "├── 📁 documentation/            (All .md files)"
echo "├── 📁 data/                     (Dataset - unchanged)"
echo "├── 📁 output/                   (Results - unchanged)"
echo "├── 📁 venv/                     (Virtual env - unchanged)"
echo "└── 🐍 Demo scripts (main folder)"
echo "    ├── demo_with_gradcam.py"
echo "    ├── find_resistant_samples.py"
echo "    ├── find_ALL_resistance_types.py"
echo "    ├── show_input_files.py"
echo "    └── view_csv.py"
echo ""
echo "🗑️  You can safely delete 'old_deprecated/' folder when ready"
echo ""
