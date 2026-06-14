# Cancer Resistance Detection: Data Requirements

## Component 1: Resistance Detection (Binary Classification)

### What You Need:
**Paired Cell Populations:**
- **Sensitive cells** + Drug → Shows dramatic morphological changes (death, rounding, detachment)
- **Resistant cells** + Same Drug → Shows minimal/no changes (survival, continued growth)
- **Both populations** + DMSO → Baseline control (ESSENTIAL)

**Example Setup:**
```
MCF-7 Parental (sensitive) + Paclitaxel 10nM → 80% death, rounded cells
MCF-7-TaxR (resistant) + Paclitaxel 10nM → <10% death, normal morphology
Both + DMSO → Healthy, proliferating cells
```

**Minimum Dataset:**
- 3-5 sensitive cell lines
- 3-5 resistant counterparts (same genetic background)
- 5-10 drugs with different mechanisms
- 3-4 concentrations per drug
- 3 replicates per condition
- **DMSO controls for everything**
- Time points: 24h, 48h, 72h post-treatment

**Total Images Needed:** ~10,000-20,000 images

---

## Component 2: Resistance Type Classification

### Resistance Mechanism Categories:

1. **Drug Efflux (ABC Transporters)**
   - MDR1/ABCB1 (P-glycoprotein)
   - MRP1/ABCC1
   - BCRP/ABCG2
   - Morphology: Often normal-looking despite drug presence

2. **Target Mutation**
   - EGFR mutations (gefitinib/erlotinib resistance)
   - BRAF mutations (vemurafenib resistance)
   - BCR-ABL mutations (imatinib resistance)
   - Morphology: Varies by target

3. **Apoptosis Evasion**
   - BCL-2 overexpression
   - TP53 mutations
   - Caspase defects
   - Morphology: Cells survive but may show stress without dying

4. **DNA Repair Enhancement**
   - PARP upregulation
   - MMR pathway changes
   - Morphology: Recovery after initial damage

5. **EMT (Epithelial-Mesenchymal Transition)**
   - Morphology: Elongated, spindle-shaped cells
   - Loss of cell-cell contacts
   - Increased motility

6. **Metabolic Reprogramming**
   - Altered glucose/glutamine metabolism
   - Morphology: May show different mitochondrial patterns

### What You Need:
- **Characterized cell lines** with known resistance mechanisms
- Each mechanism represented by 2-3 cell line models
- **Genetic/molecular validation**: Western blots, qPCR, sequencing confirming mechanism
- Images at multiple drug concentrations
- **Controls:** Sensitive parental cells with same drug

**Labeling Requirement:**
Each image must be annotated with:
- Resistance mechanism (ground truth from molecular data)
- Cell line identifier
- Drug and concentration
- Treatment duration

---

## Component 3: Resistance Quantification

### What You Need:
**Dose-Response Data Paired with Images:**
```
For each cell line + drug combination:
├── IC50 value (concentration killing 50% of cells)
├── Resistance factor (RF = IC50_resistant / IC50_sensitive)
├── Images at 5-8 concentrations spanning IC50
└── Viability/survival assay data
```

**Example:**
```
MCF-7 Parental:
  Doxorubicin IC50 = 50 nM
  Images at: 0, 10, 25, 50, 100, 200 nM
  
MCF-7-DoxR:
  Doxorubicin IC50 = 2000 nM (RF = 40)
  Images at: 0, 100, 500, 1000, 2000, 4000 nM
  
Resistance Score = 0.95 (very high)
```

**Quantification Metrics:**
- **IC50 ratio** (resistant/sensitive)
- **Survival percentage** at standard dose
- **Recovery rate** after drug removal
- **Cross-resistance profile** (resistance to related drugs)

**Minimum Data:**
- 10+ cell lines with varying resistance levels (RF from 2x to 100x)
- 8-10 drugs
- Complete dose-response curves
- Matched imaging and viability data

---

## Component 4: Drug Recommendation

### What You Need:
**Drug-Drug Relationship Database:**

1. **Mechanism of Action (MOA) Matrix**
   - BBBC021 can help here! It has 12 MOA categories
   - Build similarity scores between drugs based on morphology
   - Compounds with different MOAs may overcome resistance

2. **Cross-Resistance Profiles**
   ```
   If resistant to Drug A (MDR1-mediated):
   ├── Also resistant to: Drugs B, C, D (all MDR1 substrates)
   └── Likely sensitive to: Drugs X, Y, Z (non-substrates or MDR1 inhibitors)
   ```

3. **Pathway-Drug Mapping**
   ```
   Resistance Mechanism → Recommend:
   ├── MDR1 overexpression → Non-substrate drugs + MDR1 inhibitors (verapamil)
   ├── EGFR mutation → 3rd gen EGFR inhibitors (osimertinib)
   ├── Apoptosis defect → BH3 mimetics, alternative death pathways
   └── EMT → MET inhibitors, combination therapies
   ```

4. **Literature Database**
   - Published drug combinations that overcome specific resistance
   - Clinical trial data on resistance mechanisms
   - Drug approval status (FDA-approved vs experimental)

**Data Structure:**
```python
{
  "resistance_mechanism": "MDR1_overexpression",
  "original_drug": "doxorubicin",
  "recommended_alternatives": [
    {
      "drug": "cisplatin",
      "rationale": "Non-MDR1 substrate",
      "evidence_level": "high",
      "expected_efficacy": 0.85
    },
    {
      "drug": "doxorubicin + verapamil",
      "rationale": "MDR1 inhibitor combination",
      "evidence_level": "moderate",
      "expected_efficacy": 0.75
    }
  ]
}
```

---

## How to Use BBBC021 in Your Pipeline

### Current BBBC021 Assets:
✅ 113 compounds with MOA labels (12 categories)
✅ 3-channel high-quality imaging (DAPI, Tubulin, Actin)
✅ Multiple concentrations
✅ Established analysis pipelines

### BBBC021 Limitations:
❌ Single cell line (MCF-7 parental only - sensitive)
❌ No resistant cell populations
❌ No resistance mechanism labels
❌ Limited to morphological profiling

### Strategic Use of BBBC021:

**Use Case 1: Pre-training for Feature Extraction**
- Train a deep learning model to recognize cellular features
- Learn representations of nucleus, cytoskeleton, morphology
- Transfer learning: Use these features for resistance detection

**Use Case 2: MOA Classification for Drug Recommendations**
- Build a model to classify drug MOAs from morphology
- Use this to suggest drugs with different MOAs for resistant cells
- "If cell is resistant to Aurora kinase inhibitor, try Eg5 inhibitor"

**Use Case 3: Baseline Morphology Database**
- Use as reference for "normal" MCF-7 morphology
- Compare resistant cells to this baseline
- Identify morphological signatures unique to resistance

---

## Practical Implementation Pathway

### Phase 1: Pilot Dataset (Achievable with modest resources)
**Minimum Viable Dataset:**
- 2 cell line pairs (sensitive + resistant)
  - MCF-7 parental + MCF-7-DoxR (doxorubicin resistant, MDR1)
  - MCF-7 parental + MCF-7-TaxR (paclitaxel resistant, tubulin mutation)
- 4 drugs (doxorubicin, paclitaxel, cisplatin, 5-FU)
- 4 concentrations + DMSO control
- 3 replicates
- **Total: ~400 images**

This proves concept for all 4 components!

### Phase 2: Expanded Dataset
- 5 cell line pairs
- 10 drugs spanning different MOAs
- More resistance mechanisms
- **Total: ~5,000 images**

### Phase 3: Full Scale
- 10+ cell lines with multiple resistance mechanisms
- 20+ drugs
- Temporal data
- Public dataset contribution
- **Total: 20,000+ images**

---

## Critical Success Factors

### Must-Haves:
1. ✅ **DMSO controls** - Absolutely essential for baseline
2. ✅ **Matched pairs** - Same parental line, developed resistance
3. ✅ **Molecular validation** - Confirm resistance mechanism by protein/RNA
4. ✅ **Dose-response data** - Quantitative viability measurements
5. ✅ **Consistent imaging** - Same microscope, settings, staining protocol

### Nice-to-Haves:
- Time-lapse imaging showing resistance development
- Single-cell tracking
- Additional markers (drug accumulation, apoptosis markers)
- 3D imaging
- Live/dead staining

---

## Expected Model Performance Targets

Based on similar published work:

**Component 1: Resistance Detection**
- Target Accuracy: 85-95%
- Challenge: Partial resistance (intermediate phenotypes)

**Component 2: Resistance Type**
- Target Accuracy: 70-85%
- Challenge: Multiple concurrent mechanisms

**Component 3: Resistance Quantification**
- Target Correlation with IC50: R² > 0.7
- Challenge: Non-linear dose-response relationships

**Component 4: Drug Recommendations**
- Target Top-3 Accuracy: 60-75%
- Challenge: Limited training data for rare combinations

---

## Timeline Estimate

**With existing lab access:**
- Phase 1 pilot: 3-4 months
- Model development: 2-3 months
- Validation: 1-2 months
- **Total: ~8-10 months**

**Starting from scratch:**
- Cell line generation: 6-12 months (resistance development takes time)
- Rest of pipeline: +8-10 months
- **Total: 14-22 months**
