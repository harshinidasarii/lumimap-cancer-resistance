# LUMIMAP: Multi-Type Cancer Drug Resistance Detection System

## 🎯 Innovation: Resistance TYPE Classification, Not Just Yes/No

LUMIMAP is **not** just a binary resistance detector. It classifies **7 different resistance mechanisms** based on morphological features, enabling precision therapy recommendations.

## 🔬 The 7 Resistance Types

LUMIMAP detects these resistance mechanisms based on cellular morphology:

### 1. **Endocrine/Hormone Resistance**
**Features:**
- Larger nucleus, spindle-shaped cells
- Rough membrane texture
- Presence of endosomes/lysosomes

**Therapy:** CDK4/6 inhibitors, mTOR inhibitors, ER degraders

---

### 2. **Drug Efflux Resistance** 
**Features:**
- Reduced nucleus-to-cytoplasm ratio
- Assembled actin filaments
- Vesicle-rich cytoplasm

**Therapy:** P-gp inhibitors, liposomal formulations, ADCs

---

### 3. **Apoptosis Resistance**
**Features:**
- Large, smooth nuclei
- Mitochondrial swelling
- Low apoptotic bodies

**Therapy:** BCL-2 inhibitors, TRAIL agonists, ferroptosis inducers

---

### 4. **Metabolic Rewiring**
**Features:**
- Perinuclear mitochondrial clustering
- Spindle shapes
- High nucleus-to-cytoplasm ratio

**Therapy:** Glutaminase inhibitors, FASN inhibitors, metformin

---

### 5. **ncRNA-Mediated Resistance**
**Features:**
- Subtle nuclear texture changes
- Increased heterochromatin
- Irregular elongated nuclei

**Therapy:** ASOs, epigenetic modifiers, miRNA mimics

---

### 6. **EMT-Like Resistance**
**Features:**
- Elongated cells
- Loss of cell-cell adhesion
- Irregular cell shapes

**Therapy:** TGF-β inhibitors, Src inhibitors, FAK inhibitors

---

### 7. **Target Therapy Resistance**
**Features:**
- Intracellular heterogeneity
- Increased mitotic cells
- Altered chromatin compaction

**Therapy:** Alternative pathway inhibitors, multi-kinase inhibitors

---

## 🏗️ System Architecture

### Key Innovations

#### 1. **Channel-Aware Learning**
Instead of stacking channels as RGB, we process DAPI, Tubulin, and Actin **separately** to learn channel-specific features:

```python
DAPI (Nucleus):
- Nuclear size, shape, texture
- Chromatin organization
- Cell division markers

Tubulin (Microtubules):
- Spindle formation
- Mitotic structures
- Microtubule stability

Actin (Cytoskeleton):
- Cell shape and elongation
- Cell-cell adhesion
- Cytoskeletal organization
```

#### 2. **Concentration-Aware**
Drug concentration affects phenotype:
- Low dose: Weak/partial phenotype
- Medium dose: Clear MOA phenotype  
- High dose: Toxic/off-target effects

The model learns concentration-dependent patterns.

#### 3. **Multi-Type Classification**
Not binary (resistant/sensitive) but **8-class** (7 resistance types + sensitive).

#### 4. **Therapy Recommendation Engine**
Each resistance type maps to specific therapeutic strategies:

```
Resistance Type → Therapeutic Strategy → Drug Recommendations
```

#### 5. **Channel-Wise GradCAM**
Separate explanations for each channel:
- Which cellular compartment is relevant?
- What morphological features led to the classification?

---

## 📁 File Structure

```
lumimap/
├── therapy_recommendation_database.py  # Resistance types & therapy mappings
├── resistance_type_classifier_train.py # Training script
├── resistance_inference.py             # Inference & visualization
└── README.md                          # This file
```

---

## 🚀 Usage

### Step 1: Training

```bash
python resistance_type_classifier_train.py
```

**Requirements:**
- BBBC021 dataset (Weeks 1-3)
- BBBC021_v1_image.csv
- BBBC021_v1_moa.csv

**Output:**
- `best_resistance_model.pth` - Trained model
- `training_curves.png` - Training progress
- `confusion_matrix.png` - Classification performance
- `classification_report.txt` - Detailed metrics

### Step 2: Inference

```bash
python resistance_inference.py \
  --model outputs_resistance/best_resistance_model.pth \
  --dapi path/to/dapi.tif \
  --tubulin path/to/tubulin.tif \
  --actin path/to/actin.tif \
  --compound taxol \
  --concentration 1.0 \
  --output results/taxol_analysis
```

**Output:**
- `resistance_analysis_visualization.png` - Visual report
- `resistance_analysis.json` - Machine-readable results
- `resistance_analysis.txt` - Clinical report with therapy recommendations

---

## 📊 What the Output Looks Like

### Visualization Components

**1. Channel-Specific GradCAMs**
- DAPI: Highlights nuclear features
- Tubulin: Highlights microtubule structures
- Actin: Highlights cytoskeletal features  
- Combined: Overall important regions

**2. Resistance Type Probabilities**
- Bar chart showing probability for each of the 8 classes
- Predicted class highlighted

**3. Therapy Recommendations**
- Detected resistance type
- First-line drug recommendations
- Combination therapy suggestions
- Biomarkers to test
- Monitoring parameters

### Example Output

```
================================================================================
                    LUMIMAP RESISTANCE ANALYSIS REPORT
================================================================================

CURRENT THERAPY:
  Drug: taxol
  Mechanism of Action: Microtubule stabilizers

RESISTANCE ANALYSIS:
  Detected Resistance Type: DRUG_EFFLUX
  Confidence: 87.3%

MORPHOLOGICAL FEATURES DETECTED:

  STRUCTURAL:
    • Reduced nucleus-to-cytoplasm ratio
    • Assembled actin filaments

  CYTOPLASMIC:
    • Higher membrane signal
    • Vesicle-rich cytoplasm

THERAPEUTIC STRATEGY:
  Inhibit drug efflux pumps or use pump-independent drugs

FIRST-LINE RECOMMENDATIONS:
  • P-glycoprotein inhibitors (Valspodar, Tariquidar)
  • BCRP inhibitors (Ko143)
  • MRP1 inhibitors

COMBINATION THERAPY:
  Efflux inhibitor + Original chemotherapy

RATIONALE:
  Increased vesicular activity and membrane pumps detected

BIOMARKERS TO TEST:
  • MDR1/P-gp expression
  • BCRP
  • MRPs

MONITORING PARAMETERS:
  • Intracellular drug accumulation
  • Vesicle formation
  • Membrane transporter expression

================================================================================
```

---

## 🔬 How It Works: The Data Flow

### Training Phase

```
1. Load BBBC021 Images
   ↓
2. Generate Pseudo-Labels (temporary - will be replaced with contrastive learning)
   - Compare to DMSO baseline
   - Extract morphological features
   - Assign resistance type based on phenotype
   ↓
3. Train Channel-Aware Model
   - DAPI encoder → nuclear features
   - Tubulin encoder → microtubule features
   - Actin encoder → cytoskeletal features
   - Channel attention → weight important channels
   - Concentration encoding → dose-response
   ↓
4. Multi-class classification (8 classes)
   ↓
5. Save trained model
```

### Inference Phase

```
1. Load Patient Image (DAPI + Tubulin + Actin)
   ↓
2. Preprocess & Normalize
   ↓
3. Forward Pass
   - Extract channel-specific features
   - Apply channel attention
   - Encode drug concentration
   - Classify resistance type
   ↓
4. Generate GradCAM for Each Channel
   ↓
5. Map Resistance Type → Therapy Recommendations
   ↓
6. Generate Clinical Report
```

---

## 🎓 Key Differences from Previous Approach

### ❌ OLD (moa_classifier_train.py)

```python
Input: Image (3 channels stacked as RGB)
Output: MOA label (e.g., "Microtubule stabilizer")

Problems:
- Just classifies drug MOA, not resistance
- No resistance labels
- No therapy recommendations
- Doesn't use drug/concentration info
- No baseline comparison (DMSO)
```

### ✅ NEW (resistance_type_classifier_train.py)

```python
Input: 
  - Image (DAPI, Tubulin, Actin processed separately)
  - Drug name
  - Concentration

Output:
  - Resistance TYPE (7 types + sensitive)
  - Confidence score
  - Therapy recommendations
  - Channel-wise GradCAM explanations

Innovations:
- Channel-aware architecture
- Concentration encoding
- Multi-type resistance classification
- Therapy recommendation engine
- Clinical actionability
```

---

## 🔮 Next Steps: Improving the System

### Phase 1: Contrastive Learning (TODO)

Replace pseudo-labels with unsupervised contrastive learning:

```python
# Learn what each MOA phenotype looks like
Anchor: taxol (Microtubule stabilizer)
Positive: docetaxel (same MOA, different drug)
Negative: cytochalasin (different MOA)

→ Model learns: "This is what microtubule stabilization looks like"
```

### Phase 2: DMSO Baseline Comparison

```python
# Resistance = looks like DMSO when treated
For each drug-treated cell:
  similarity_to_dmso = compare(cell, dmso_reference)
  similarity_to_moa = compare(cell, expected_moa_phenotype)
  
  if high_dmso_similarity & low_moa_similarity:
    → PRIMARY RESISTANCE
  elif shows_different_moa:
    → CROSS-RESISTANCE
```

### Phase 3: Feature Engineering

Extract quantitative morphological features:
- Nucleus size, shape, texture
- Nucleus-to-cytoplasm ratio
- Mitochondrial distribution
- Cell elongation
- Cell-cell adhesion
- Vesicle count

Use these features as additional inputs.

---

## 📈 Expected Performance

With proper training:
- **Sensitivity classification**: 75-85% accuracy
- **Multi-type resistance**: 60-75% accuracy (harder task)
- **Therapy recommendation**: Clinical validation needed

---

## 🧪 Clinical Use Case Example

**Scenario:** Patient with breast cancer on taxol (microtubule stabilizer)

**LUMIMAP Analysis:**
1. Takes cell images: DAPI, Tubulin, Actin
2. Detects: **Drug Efflux Resistance** (87% confidence)
3. GradCAM shows: Vesicle-rich cytoplasm, high membrane signal
4. Recommendation: 
   - Add P-glycoprotein inhibitor (Valspodar)
   - Or switch to liposomal formulation
   - Test MDR1/P-gp expression
5. Monitor: Intracellular drug accumulation

**Outcome:** Precision therapy based on resistance mechanism!

---

## 📚 References

**BBBC021 Dataset:**
- [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC021)
- Caie et al., *Molecular Cancer Therapeutics*, 2010

**Resistance Mechanisms:**
- Based on clinical oncology literature
- Morphological signatures from cell biology research

---

## 🤝 Contributing

This is a research project. Key areas for improvement:

1. **Better pseudo-label generation** (contrastive learning)
2. **More training data** (additional BBBC021 weeks)
3. **Clinical validation** (real patient samples)
4. **Feature engineering** (quantitative morphology)
5. **Therapy database expansion** (more drugs, combinations)

---

## ⚠️ Important Notes

**Current Limitations:**

1. **Pseudo-labels are temporary** - Using simple heuristics for now
2. **Limited training data** - Only Weeks 1-3 of BBBC021
3. **Not clinically validated** - Requires validation on patient samples
4. **Drug database incomplete** - MOA mappings are simplified

**This is a RESEARCH SYSTEM** - Not for clinical use without proper validation!

---

## 📞 Contact

For questions about LUMIMAP:
- Check the code comments
- Review the therapy_recommendation_database.py for resistance types
- Examine the GradCAM outputs to understand model decisions

---

## 🎯 Summary: Why LUMIMAP is Innovative

**Traditional Approach:**
"Is this cell resistant?" → Yes/No → No actionable information

**LUMIMAP Approach:**
"What TYPE of resistance?" → 7 specific mechanisms → Precision therapy recommendations

**The difference:** Going from binary classification to **mechanism-based precision oncology**!

---

## License

Research use only. Dataset from Broad Institute BBBC021.
