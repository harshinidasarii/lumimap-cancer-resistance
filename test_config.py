"""
Centralized Test Configuration
===============================
This file contains test image paths and metadata to ensure both
resistance_detector_inference.py and gradcam_explainer.py analyze
the same image with consistent metadata.

Usage:
    from test_config import TEST_IMAGE, DRUG_NAME, DRUG_CONCENTRATION, EXPECTED_MOA
"""

# =============================================================================
# TEST IMAGE CONFIGURATION
# =============================================================================

# Choose which test image to use (uncomment ONE):

# Option 1: Week2 cytochalasin B (92.26% resistance)
# TEST_IMAGE = 'data/Week1/Week1_22123/Week1_150607_B02_s4_w2EE226363-0BAC-443F-A41C-16C9C89FDDFA.tif'
# DRUG_NAME = 'cytochalasin B'
# DRUG_CONCENTRATION = '10 nM'
# EXPECTED_MOA = 'Actin disruptors'

# Option 2: Week1 image (100% resistance)

TEST_IMAGE = 'data/image.png'
DRUG_NAME = 'paclitaxel'
DRUG_CONCENTRATION = '10 nM'
EXPECTED_MOA = 'Actin disruptors'

# Option 3: Add your own test image here
# TEST_IMAGE = 'path/to/your/image.tif'
# DRUG_NAME = 'your_drug_name'
# DRUG_CONCENTRATION = 'your_concentration'
# EXPECTED_MOA = 'expected_MOA_class'

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that configuration is properly set"""
    import os
    
    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(
            f"Test image not found: {TEST_IMAGE}\n"
            f"Please check the path in test_config.py"
        )
    
    print(f"✓ Configuration validated:")
    print(f"  Image: {TEST_IMAGE}")
    print(f"  Drug: {DRUG_NAME} ({DRUG_CONCENTRATION})")
    print(f"  Expected MOA: {EXPECTED_MOA}")
    return True

if __name__ == "__main__":
    # Test the configuration
    validate_config()
