"""
Morphological Feature Analyzer
===============================
Analyzes actual cell morphology to classify resistance mechanisms
Complements MOA-based classification with real shape analysis
"""

import cv2
import numpy as np
from scipy import ndimage

def analyze_cell_morphology(image_path):
    """
    Analyze cell morphology features from microscopy image
    
    Returns:
        dict: Morphological features including:
            - elongation: How spindle-like vs round cells are (0=round, 1=spindle)
            - compactness: How tightly packed cells are
            - circularity: Average circularity of cells
            - cell_separation: How scattered vs clustered cells are
            - morphology_type: 'epithelial' or 'mesenchymal'
    """
    
    # Load image
    from PIL import Image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to 8-bit if needed
    if img_array.dtype in [np.uint16, np.int16]:
        img_array = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Threshold to segment cells
    # Use Otsu's method for automatic thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours (cell boundaries)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (noise)
    min_area = 100  # pixels
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if len(contours) == 0:
        return {
            'elongation': 0.0,
            'compactness': 0.0,
            'circularity': 0.0,
            'cell_separation': 0.0,
            'morphology_type': 'unknown',
            'num_cells': 0
        }
    
    # Analyze each cell
    elongations = []
    circularities = []
    areas = []
    
    for contour in contours:
        # Get moments
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        
        # Area
        area = cv2.contourArea(contour)
        areas.append(area)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (1.0 = perfect circle, <1.0 = elongated)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(min(circularity, 1.0))  # Cap at 1.0
        
        # Fit ellipse to get elongation
        if len(contour) >= 5:  # Need at least 5 points to fit ellipse
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse  # MA=major axis, ma=minor axis
            
            if ma > 0:
                # Aspect ratio: >2 = spindle-like, ~1 = round
                aspect_ratio = MA / ma
                # Convert to elongation score (0=round, 1=very elongated)
                elongation = min((aspect_ratio - 1) / 3, 1.0)  # Normalize
                elongations.append(elongation)
    
    # Calculate overall metrics
    avg_elongation = np.mean(elongations) if elongations else 0.0
    avg_circularity = np.mean(circularities) if circularities else 0.0
    
    # Cell separation: measure spacing between cells
    # Higher = more scattered (mesenchymal-like)
    # Lower = more clustered (epithelial-like)
    if len(contours) > 1:
        # Get centroids
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append([cx, cy])
        
        # Calculate average nearest-neighbor distance
        from scipy.spatial import distance_matrix
        if len(centroids) > 1:
            dist_matrix = distance_matrix(centroids, centroids)
            # Get minimum non-zero distance for each cell
            np.fill_diagonal(dist_matrix, np.inf)
            min_distances = np.min(dist_matrix, axis=1)
            avg_cell_distance = np.mean(min_distances)
            
            # Normalize by average cell size
            avg_cell_diameter = np.sqrt(np.mean(areas) / np.pi) * 2
            cell_separation = avg_cell_distance / avg_cell_diameter if avg_cell_diameter > 0 else 0
        else:
            cell_separation = 0.0
    else:
        cell_separation = 0.0
    
    # Compactness: how tightly cells fill space
    total_cell_area = sum(areas)
    image_area = gray.shape[0] * gray.shape[1]
    compactness = total_cell_area / image_area
    
    # Classify morphology type based on features
    # EMT/Mesenchymal indicators:
    # - High elongation (>0.4)
    # - Low circularity (<0.6)
    # - High cell separation (>2.0)
    
    # Epithelial indicators:
    # - Low elongation (<0.3)
    # - High circularity (>0.6)
    # - Low cell separation (<1.5)
    
    emt_score = 0
    epithelial_score = 0
    
    # Elongation
    if avg_elongation > 0.4:
        emt_score += 1
    elif avg_elongation < 0.3:
        epithelial_score += 1
    
    # Circularity
    if avg_circularity < 0.6:
        emt_score += 1
    elif avg_circularity > 0.6:
        epithelial_score += 1
    
    # Cell separation
    if cell_separation > 2.0:
        emt_score += 1
    elif cell_separation < 1.5:
        epithelial_score += 1
    
    # Determine type
    if emt_score > epithelial_score:
        morphology_type = 'mesenchymal'
        confidence = emt_score / 3.0
    elif epithelial_score > emt_score:
        morphology_type = 'epithelial'
        confidence = epithelial_score / 3.0
    else:
        morphology_type = 'intermediate'
        confidence = 0.33
    
    return {
        'elongation': float(avg_elongation),
        'circularity': float(avg_circularity),
        'compactness': float(compactness),
        'cell_separation': float(cell_separation),
        'morphology_type': morphology_type,
        'morphology_confidence': float(confidence),
        'num_cells': len(contours),
        'emt_score': emt_score,
        'epithelial_score': epithelial_score
    }


def classify_resistance_with_morphology(predicted_moa, expected_moa, image_path, resistance_score):
    """
    Improved resistance mechanism classification using both MOA and morphology
    
    Args:
        predicted_moa: Predicted MOA from model
        expected_moa: Expected MOA from drug
        image_path: Path to cell image for morphology analysis
        resistance_score: Resistance score (0-100)
    
    Returns:
        dict: Classification with mechanism, confidence, and morphological evidence
    """
    
    # Analyze morphology
    morph_features = analyze_cell_morphology(image_path)
    
    # Get MOA-based hypothesis
    moa_mechanism = get_moa_based_hypothesis(predicted_moa, expected_moa)
    
    # Combine MOA hypothesis with morphology evidence
    final_mechanism = {
        'mechanism': None,
        'confidence': 'low',
        'morphology_support': 'unknown',
        'moa_hypothesis': moa_mechanism,
        'morphology_evidence': morph_features
    }
    
    # Check if morphology supports MOA hypothesis
    if 'EMT' in moa_mechanism or 'cytoskeletal reorganization' in moa_mechanism:
        # MOA suggests EMT-like, check morphology
        if morph_features['morphology_type'] == 'mesenchymal':
            final_mechanism['mechanism'] = 'EMT-like Resistance (cytoskeletal reorganization)'
            final_mechanism['confidence'] = 'high'
            final_mechanism['morphology_support'] = 'confirmed'
        elif morph_features['morphology_type'] == 'epithelial':
            final_mechanism['mechanism'] = 'Epithelial Resistance (maintaining epithelial phenotype)'
            final_mechanism['confidence'] = 'high'
            final_mechanism['morphology_support'] = 'contradicts MOA pattern'
        else:
            final_mechanism['mechanism'] = 'Cytoskeletal Remodeling (type unclear)'
            final_mechanism['confidence'] = 'medium'
            final_mechanism['morphology_support'] = 'intermediate'
    
    elif 'DNA damage' in moa_mechanism:
        final_mechanism['mechanism'] = moa_mechanism
        final_mechanism['confidence'] = 'medium'
        final_mechanism['morphology_support'] = 'not applicable'
    
    else:
        final_mechanism['mechanism'] = moa_mechanism
        final_mechanism['confidence'] = 'medium'
        final_mechanism['morphology_support'] = 'not analyzed'
    
    return final_mechanism


def get_moa_based_hypothesis(predicted_moa, expected_moa):
    """Original MOA-based hypothesis (rule-based)"""
    
    # Cytoskeletal remodeling patterns
    cytoskeletal_moas = ['Actin disruptors', 'Microtubule stabilizers', 
                         'Microtubule destabilizers']
    
    if predicted_moa in cytoskeletal_moas and expected_moa in cytoskeletal_moas:
        if predicted_moa != expected_moa:
            return 'EMT-like Resistance (cytoskeletal reorganization) [MOA hypothesis]'
    
    # DNA damage patterns
    if 'DNA' in predicted_moa or 'DNA' in expected_moa:
        return 'DNA Damage Response Resistance'
    
    # Default
    return 'General Resistance (mechanism unclear)'


if __name__ == '__main__':
    # Test on your image
    test_image = 'data/Week2/Week2_24141/Week2_180607_B02_s2_w4ED129C44-C20B-4ACC-BFA6-15AB8F608B75.tif'
    
    print("Analyzing cell morphology...")
    features = analyze_cell_morphology(test_image)
    
    print("\n" + "="*60)
    print("MORPHOLOGICAL ANALYSIS")
    print("="*60)
    print(f"Number of cells detected: {features['num_cells']}")
    print(f"\nShape Metrics:")
    print(f"  Elongation: {features['elongation']:.3f} (0=round, 1=spindle)")
    print(f"  Circularity: {features['circularity']:.3f} (1=perfect circle)")
    print(f"  Cell separation: {features['cell_separation']:.3f} (distance/diameter)")
    print(f"  Compactness: {features['compactness']:.3f} (cells/total area)")
    
    print(f"\nMorphology Classification:")
    print(f"  Type: {features['morphology_type'].upper()}")
    print(f"  Confidence: {features['morphology_confidence']:.1%}")
    print(f"  EMT score: {features['emt_score']}/3")
    print(f"  Epithelial score: {features['epithelial_score']}/3")
    
    print(f"\nInterpretation:")
    if features['morphology_type'] == 'epithelial':
        print("  ✓ Cells show EPITHELIAL morphology (round/polygonal)")
        print("  ✓ NOT showing EMT characteristics (no spindles/needles)")
        print("  → Resistance mechanism: Epithelial Resistance")
        print("    (cells 'doubling down' on epithelial phenotype)")
    elif features['morphology_type'] == 'mesenchymal':
        print("  ✓ Cells show MESENCHYMAL morphology (spindle-like)")
        print("  ✓ EMT characteristics present (elongated/scattered)")
        print("  → Resistance mechanism: EMT-like Resistance")
    else:
        print("  ⚠ Intermediate morphology detected")
        print("  → Need more analysis to determine type")
    
    print("="*60)
