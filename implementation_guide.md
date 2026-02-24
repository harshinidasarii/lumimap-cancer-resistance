# Cancer Resistance AI System: Complete Implementation Guide

## Your Conceptual Approach - Validated ✅

**Core Insight:** Resistance = Morphological Mismatch
```
Expected MOA (what drug should do) ≠ Observed MOA (what cells actually show)
                                    ↓
                            RESISTANCE DETECTED
```

This is brilliant because:
1. ✅ Can use BBBC021 immediately (has MOA labels!)
2. ✅ Don't need explicit "resistant" labels initially
3. ✅ Quantifiable mismatch = quantifiable resistance
4. ✅ MOA-based recommendations are natural

---

## REFINED System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 1: MOA PREDICTOR                      │
│               (Train on BBBC021 first)                       │
└─────────────────────────────────────────────────────────────┘

INPUT: 3-Channel Image (DAPI, Tubulin, Actin)
  ↓
┌─────────────────────────────────────────────────────────────┐
│  CNN Feature Extractor (ResNet50/EfficientNet/ViT)         │
│  Pre-trained on ImageNet → Fine-tuned on BBBC021            │
│  Output: 2048-D feature vector                              │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  MOA Classification Head                                     │
│  12 Classes: Actin disruptors, Aurora kinase inhibitors,    │
│  Eg5 inhibitors, Epithelial, Protein degradation,           │
│  Protein synthesis, Microtubule destabilizers,              │
│  Microtubule stabilizers, Cholesterol-lowering,             │
│  Kinase inhibitors, DNA damage, DNA replication             │
│  Output: Probability distribution over 12 MOAs              │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
                   [MOA Prediction]
                   e.g., "Actin disruptor" with 0.92 confidence

───────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: RESISTANCE DETECTION                   │
│         (Requires resistant cell data)                       │
└─────────────────────────────────────────────────────────────┘

For each test image:
  1. Know which drug was applied → Expected MOA
  2. Run through MOA Predictor → Predicted MOA
  3. Compare:
  
┌─────────────────────────────────────────────────────────────┐
│  Comparison Logic:                                           │
│                                                              │
│  IF Predicted MOA == Expected MOA:                          │
│      → Cell is SENSITIVE (drug worked as expected)          │
│                                                              │
│  IF Predicted MOA ≈ DMSO control:                           │
│      → Cell is RESISTANT (no effect, looks normal)          │
│                                                              │
│  IF Predicted MOA == Different MOA:                         │
│      → Cell is CROSS-RESISTANT or has compensatory pathway  │
│                                                              │
│  Resistance Score = 1 - (Probability of Expected MOA)       │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
              [Resistance Detection]

───────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│           PHASE 3: RESISTANCE CLASSIFICATION                 │
└─────────────────────────────────────────────────────────────┘

Analyze the pattern:
  
  Predicted MOA = "DMSO-like" → Efflux pump resistance (MDR1)
                                 Drug never reached target
  
  Predicted MOA = "Different pathway" → Target mutation
                                         Drug hit wrong target
  
  Predicted MOA = "Weak response" → Partial resistance
                                     Apoptosis evasion
  
  Additional classifier trained on known mechanisms

───────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│            PHASE 4: DRUG RECOMMENDATION                      │
└─────────────────────────────────────────────────────────────┘

MOA-based recommendation engine:
  
  Current: Drug A (Actin disruptor) → Resistant
  Predicted: DMSO-like morphology
  
  Recommendations:
  1. Different MOA drugs (Aurora kinase inhibitor, Eg5 inhibitor)
  2. MDR1 inhibitors + Drug A
  3. Drugs known to work in MDR1-overexpressing cells
```

---

## Component Breakdown

### Component 1: MOA Predictor (Foundation)

**Training Dataset:** BBBC021
- 13,200 fields of view
- 113 compounds
- 12 MOA classes
- Already labeled!

**Architecture:**
```python
class MOAPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone - use pre-trained CNN
        self.backbone = torchvision.models.resnet50(pretrained=True)
        # Modify first conv layer for 3 channels (already RGB-like)
        # Keep pretrained weights
        
        # Replace final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 12)  # 12 MOA classes
        )
    
    def forward(self, x):
        # x shape: (batch, 3, H, W) - DAPI, Tubulin, Actin
        features = self.backbone(x)
        return features  # logits for 12 classes
```

**Training Strategy:**
```python
# Loss function
criterion = nn.CrossEntropyLoss()

# Data augmentation (critical for microscopy)
transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])

# Training loop
for epoch in range(epochs):
    for images, moa_labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, moa_labels)
        # ... standard training
```

**Expected Performance on BBBC021:**
- Published results: 89-96% accuracy (per-treatment)
- Your target: >90% accuracy

---

### Component 2: Resistance Detector (Core Innovation)

**Key Insight:** Measure prediction confidence for expected MOA

```python
class ResistanceDetector:
    def __init__(self, moa_predictor):
        self.moa_predictor = moa_predictor
        
    def detect_resistance(self, image, drug_info):
        """
        Args:
            image: 3-channel microscopy image
            drug_info: dict with 'name', 'expected_moa', 'concentration'
        
        Returns:
            resistance_score: float [0, 1] where 1 = highly resistant
            predicted_moa: string
            confidence: float
        """
        # Get MOA prediction
        with torch.no_grad():
            logits = self.moa_predictor(image)
            probs = F.softmax(logits, dim=1)
        
        # Get expected MOA index
        expected_moa_idx = MOA_CLASSES.index(drug_info['expected_moa'])
        expected_moa_prob = probs[0, expected_moa_idx].item()
        
        # Get predicted MOA
        predicted_moa_idx = probs.argmax().item()
        predicted_moa = MOA_CLASSES[predicted_moa_idx]
        confidence = probs[0, predicted_moa_idx].item()
        
        # Calculate resistance score
        # High resistance = low probability of expected MOA
        resistance_score = 1.0 - expected_moa_prob
        
        # Refine based on prediction pattern
        if predicted_moa == "DMSO" or similarity_to_control(image) > 0.8:
            # Looks like untreated cells
            resistance_score = max(resistance_score, 0.8)
            resistance_type = "complete_resistance"
        elif predicted_moa == drug_info['expected_moa']:
            # Correct response
            resistance_score = min(resistance_score, 0.2)
            resistance_type = "sensitive"
        else:
            # Wrong MOA
            resistance_type = "cross_resistance_or_adaptation"
        
        return {
            'resistance_score': resistance_score,
            'predicted_moa': predicted_moa,
            'expected_moa': drug_info['expected_moa'],
            'confidence': confidence,
            'resistance_type': resistance_type
        }
```

---

### Component 3: Resistance Quantification (Enhanced)

**Multi-signal integration:**

```python
class ResistanceQuantifier:
    def __init__(self, moa_predictor, baseline_features):
        self.moa_predictor = moa_predictor
        self.baseline_features = baseline_features  # DMSO control stats
        
    def quantify_resistance(self, image, drug_info, viability_data=None):
        """
        Combines multiple signals for resistance quantification
        """
        # Signal 1: MOA prediction confidence
        moa_signal = self._get_moa_signal(image, drug_info)
        
        # Signal 2: Distance from DMSO baseline
        baseline_signal = self._get_baseline_similarity(image)
        
        # Signal 3: Morphological features
        morph_signal = self._get_morphological_features(image)
        
        # Signal 4: Viability data (if available)
        if viability_data:
            viability_signal = viability_data['survival_fraction']
        else:
            viability_signal = None
        
        # Combine signals
        if viability_signal is not None:
            # Supervised: use viability as ground truth
            # Train a regression model
            resistance_degree = self.regression_model.predict(
                np.concatenate([moa_signal, baseline_signal, morph_signal])
            )
        else:
            # Unsupervised: weighted combination
            resistance_degree = (
                0.4 * moa_signal +
                0.3 * baseline_signal +
                0.3 * morph_signal
            )
        
        return {
            'resistance_degree': resistance_degree,  # 0-1 scale
            'moa_confidence': moa_signal,
            'baseline_similarity': baseline_signal,
            'morphology_score': morph_signal
        }
    
    def _get_morphological_features(self, image):
        """
        Extract interpretable features:
        - Cell count (resistant cells may proliferate)
        - Cell area (apoptosis resistance)
        - Nuclear morphology (DNA damage resistance)
        - Actin organization (cytoskeleton resistance)
        """
        features = extract_cell_features(image)
        # Compare to expected drug response
        # Return deviation score
        return deviation_score
```

---

### Component 4: Drug Recommender (MOA-based)

**Strategy:** Recommend drugs with different MOAs

```python
class DrugRecommender:
    def __init__(self, drug_database, moa_similarity_matrix):
        self.drug_database = drug_database
        self.moa_similarity = moa_similarity_matrix
        
    def recommend_alternatives(self, failed_drug, resistance_profile):
        """
        Args:
            failed_drug: dict with 'name', 'moa', 'target'
            resistance_profile: output from ResistanceQuantifier
        
        Returns:
            ranked_recommendations: list of alternative drugs
        """
        failed_moa = failed_drug['moa']
        resistance_type = resistance_profile['resistance_type']
        
        recommendations = []
        
        # Strategy 1: Different MOA
        if resistance_type == "complete_resistance":
            # Drug never reached target → try different mechanism
            candidates = self.drug_database[
                self.drug_database['moa'] != failed_moa
            ]
            
            # Prefer MOAs that are mechanistically distant
            for _, drug in candidates.iterrows():
                similarity = self.moa_similarity[failed_moa][drug['moa']]
                score = 1.0 - similarity  # Less similar = better
                recommendations.append({
                    'drug': drug['name'],
                    'moa': drug['moa'],
                    'score': score,
                    'rationale': f"Different mechanism from {failed_moa}"
                })
        
        # Strategy 2: Combination therapy
        if resistance_type in ["efflux_mediated", "complete_resistance"]:
            # Add efflux pump inhibitors
            inhibitors = self.drug_database[
                self.drug_database['class'] == 'MDR1_inhibitor'
            ]
            for _, inhibitor in inhibitors.iterrows():
                recommendations.append({
                    'drug': f"{failed_drug['name']} + {inhibitor['name']}",
                    'moa': f"{failed_moa} + MDR1 inhibition",
                    'score': 0.8,
                    'rationale': "Overcome efflux-mediated resistance"
                })
        
        # Strategy 3: Next-generation agents
        if resistance_type == "target_mutation":
            # Find drugs that work against mutated targets
            next_gen = self.drug_database[
                (self.drug_database['target'] == failed_drug['target']) &
                (self.drug_database['generation'] > failed_drug['generation'])
            ]
            for _, drug in next_gen.iterrows():
                recommendations.append({
                    'drug': drug['name'],
                    'moa': drug['moa'],
                    'score': 0.85,
                    'rationale': f"Active against mutated {drug['target']}"
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:5]  # Top 5
```

---

## Visualization: Explainability with Grad-CAM

**Why cells look resistant - show heatmaps**

```python
class ResistanceExplainer:
    def __init__(self, model):
        self.model = model
        self.gradcam = GradCAM(model, target_layer='layer4')
    
    def generate_heatmap(self, image, drug_info):
        """
        Generate Grad-CAM heatmap showing which regions
        contribute to resistance detection
        """
        # Get prediction
        output = self.model(image)
        expected_moa_idx = MOA_CLASSES.index(drug_info['expected_moa'])
        
        # Generate Grad-CAM for expected MOA
        heatmap = self.gradcam(image, class_idx=expected_moa_idx)
        
        # Overlay on original image
        visualization = overlay_heatmap(image, heatmap)
        
        return visualization
    
    def explain_resistance(self, image, drug_info, resistance_result):
        """
        Provide human-readable explanation
        """
        heatmap = self.generate_heatmap(image, drug_info)
        
        explanation = {
            'visual': heatmap,
            'text': self._generate_text_explanation(resistance_result),
            'key_features': self._identify_key_features(image, heatmap)
        }
        
        return explanation
    
    def _generate_text_explanation(self, result):
        resistance_score = result['resistance_score']
        predicted_moa = result['predicted_moa']
        expected_moa = result['expected_moa']
        
        if resistance_score > 0.7:
            text = f"Cells show HIGH resistance ({resistance_score:.2f}). "
            text += f"Expected {expected_moa} phenotype, but cells display "
            text += f"{predicted_moa} morphology instead. "
            
            if predicted_moa == "DMSO":
                text += "Cells appear untreated, suggesting complete drug evasion."
            else:
                text += f"Cells activated alternative {predicted_moa} pathway."
        
        elif resistance_score > 0.4:
            text = f"Cells show MODERATE resistance ({resistance_score:.2f}). "
            text += "Partial drug response observed."
        
        else:
            text = f"Cells appear SENSITIVE ({1-resistance_score:.2f}). "
            text += f"Normal {expected_moa} response detected."
        
        return text
```

---

## Training Pipeline (Step-by-Step)

### STEP 1: Train MOA Classifier on BBBC021

```python
# Data preparation
train_data = BBBC021Dataset(
    image_dir='/path/to/BBBC021/images',
    metadata_file='BBBC021_v1_image.csv',
    moa_file='BBBC021_v1_moa.csv',
    transform=train_transforms
)

# Model
model = MOAPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(50):
    for images, moa_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, moa_labels)
        loss.backward()
        optimizer.step()

# Evaluate
accuracy = evaluate(model, test_loader)
print(f"MOA Classification Accuracy: {accuracy:.2%}")
# Target: >90%
```

### STEP 2: Collect Resistance Data

**Option A: Use your own resistant cell lines**
```python
# Image resistant and sensitive cells with same drugs
resistant_data = {
    'MCF7-DoxR': {
        'drug': 'doxorubicin',
        'expected_moa': 'DNA damage',
        'resistance_mechanism': 'MDR1',
        'images': [...],
        'ic50': 2000  # nM
    },
    'MCF7-parental': {
        'drug': 'doxorubicin',
        'expected_moa': 'DNA damage',
        'resistance_mechanism': 'none',
        'images': [...],
        'ic50': 50  # nM
    }
}
```

**Option B: Simulate resistance for proof-of-concept**
```python
# Use BBBC021 + noise/modifications to simulate resistance
# Not scientifically valid but useful for pipeline development
def simulate_resistance(sensitive_image, drug_info):
    # Make image look more like DMSO control
    dmso_control = load_dmso_images()
    resistant_sim = 0.7 * sensitive_image + 0.3 * dmso_control
    return resistant_sim
```

### STEP 3: Fine-tune Resistance Detector

```python
# Use the trained MOA predictor
resistance_detector = ResistanceDetector(model)

# Validate on resistant cell data
for cell_line, data in resistant_data.items():
    images = data['images']
    drug_info = {
        'name': data['drug'],
        'expected_moa': data['expected_moa']
    }
    
    results = []
    for img in images:
        result = resistance_detector.detect_resistance(img, drug_info)
        results.append(result['resistance_score'])
    
    avg_resistance = np.mean(results)
    print(f"{cell_line}: Resistance Score = {avg_resistance:.2f}")
    
    # Validate: resistant lines should have high scores
    if 'resistant' in cell_line.lower():
        assert avg_resistance > 0.6, "Failed to detect resistance!"
```

### STEP 4: Train Resistance Classifier

```python
# If you have labeled resistance mechanisms
resistance_classifier = ResistanceTypeClassifier(model.backbone)

# Training data: images + mechanism labels
train_data = ResistanceMechanismDataset(
    images=resistance_images,
    labels=mechanism_labels  # MDR1, EGFR_mut, EMT, etc.
)

# Train classifier
for epoch in range(30):
    for images, mechanism_labels in train_loader:
        outputs = resistance_classifier(images)
        loss = criterion(outputs, mechanism_labels)
        # ... standard training
```

### STEP 5: Build Drug Recommendation System

```python
# Create drug database from BBBC021 + literature
drug_db = pd.DataFrame({
    'name': ['doxorubicin', 'paclitaxel', 'cisplatin', ...],
    'moa': ['DNA damage', 'Microtubule stabilizer', 'DNA damage', ...],
    'target': ['TOP2A', 'TUBB', 'DNA', ...],
    'mdr1_substrate': [True, True, False, ...],
})

# Build MOA similarity matrix using BBBC021 morphology
moa_similarity = compute_moa_similarity_from_morphology(model, bbbc021_data)

# Initialize recommender
recommender = DrugRecommender(drug_db, moa_similarity)

# Test
alternatives = recommender.recommend_alternatives(
    failed_drug={'name': 'doxorubicin', 'moa': 'DNA damage'},
    resistance_profile={'resistance_type': 'efflux_mediated'}
)

for alt in alternatives:
    print(f"{alt['drug']}: {alt['rationale']} (score: {alt['score']:.2f})")
```

---

## Complete System Integration

```python
class CancerResistanceAI:
    def __init__(self):
        # Load trained MOA predictor
        self.moa_predictor = load_model('moa_predictor.pth')
        
        # Initialize components
        self.resistance_detector = ResistanceDetector(self.moa_predictor)
        self.resistance_quantifier = ResistanceQuantifier(self.moa_predictor)
        self.resistance_classifier = ResistanceTypeClassifier(self.moa_predictor)
        self.drug_recommender = DrugRecommender(drug_database, moa_similarity)
        self.explainer = ResistanceExplainer(self.moa_predictor)
    
    def analyze_image(self, image_path, drug_info, viability_data=None):
        """
        Complete pipeline: from image to recommendations
        """
        # Load and preprocess image
        image = load_microscopy_image(image_path)
        image_tensor = preprocess(image)
        
        # Component 1: Detect resistance
        detection = self.resistance_detector.detect_resistance(
            image_tensor, drug_info
        )
        
        # Component 2: Classify resistance type (if resistant)
        if detection['resistance_score'] > 0.5:
            resistance_type = self.resistance_classifier.classify(
                image_tensor
            )
        else:
            resistance_type = 'sensitive'
        
        # Component 3: Quantify resistance degree
        quantification = self.resistance_quantifier.quantify_resistance(
            image_tensor, drug_info, viability_data
        )
        
        # Component 4: Recommend alternative drugs
        if detection['resistance_score'] > 0.5:
            recommendations = self.drug_recommender.recommend_alternatives(
                drug_info, 
                {'resistance_type': resistance_type, **quantification}
            )
        else:
            recommendations = []
        
        # Generate explanation
        explanation = self.explainer.explain_resistance(
            image_tensor, drug_info, detection
        )
        
        # Compile complete report
        report = {
            'is_resistant': detection['resistance_score'] > 0.5,
            'resistance_score': detection['resistance_score'],
            'resistance_type': resistance_type,
            'resistance_degree': quantification['resistance_degree'],
            'predicted_moa': detection['predicted_moa'],
            'expected_moa': drug_info['expected_moa'],
            'recommendations': recommendations,
            'explanation': explanation
        }
        
        return report

# Usage
ai_system = CancerResistanceAI()

result = ai_system.analyze_image(
    image_path='patient_sample.tif',
    drug_info={
        'name': 'doxorubicin',
        'expected_moa': 'DNA damage',
        'concentration': 100  # nM
    },
    viability_data={'survival_fraction': 0.85}  # Optional
)

print(f"Resistance detected: {result['is_resistant']}")
print(f"Resistance score: {result['resistance_score']:.2f}")
print(f"Resistance type: {result['resistance_type']}")
print(f"\nRecommended alternatives:")
for rec in result['recommendations']:
    print(f"  - {rec['drug']}: {rec['rationale']}")
```

---

## Next Steps for Implementation

1. **Start with BBBC021 MOA classifier** (can do this NOW)
   - Download dataset
   - Train model
   - Achieve >90% accuracy
   - This is your foundation

2. **Collect pilot resistance data** (2-3 months)
   - Minimum: 2 cell line pairs
   - 4 drugs
   - Both DAPI/Tubulin/Actin imaging

3. **Implement resistance detection** (1 month)
   - Use MOA mismatch approach
   - Validate on pilot data

4. **Build recommendation engine** (1 month)
   - Drug database
   - MOA similarity from BBBC021
   - Ranking algorithm

5. **Create user interface** (1 month)
   - Upload image
   - View heatmaps
   - Get recommendations

Total timeline: ~6 months for working prototype!
