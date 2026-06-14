"""
Therapy Recommendation Database for LUMIMAP
Maps resistance types to therapeutic strategies based on mechanism
"""

from enum import Enum
from typing import List, Dict, Optional
import json

class ResistanceType(Enum):
    """7 types of cancer drug resistance + sensitive state"""
    SENSITIVE = 0
    ENDOCRINE_HORMONE = 1
    DRUG_EFFLUX = 2
    APOPTOSIS_RESISTANCE = 3
    METABOLIC_REWIRING = 4
    NCRNA_MEDIATED = 5
    EMT_LIKE = 6
    TARGET_THERAPY = 7

class MorphologicalFeatures:
    """Morphological signatures for each resistance type"""
    
    FEATURES = {
        ResistanceType.SENSITIVE: {
            "nuclear": ["Normal nucleus size", "Regular shape", "Normal chromatin"],
            "cytoplasmic": ["Normal cytoplasm", "Regular cell shape"],
            "organelle": ["Normal mitochondria", "Normal vesicles"],
            "population": ["Organized arrangement", "Normal confluence"]
        },
        
        ResistanceType.ENDOCRINE_HORMONE: {
            "nuclear": [
                "Larger nucleus",
                "Wider distribution of cell size and shape",
                "Spindle-shaped cells",
                "Isolated and disorganized arrangement",
                "High confluence"
            ],
            "membrane": [
                "Rough membrane texture",
                "More defined structures",
                "Presence of endosomes and lysosomes"
            ]
        },
        
        ResistanceType.DRUG_EFFLUX: {
            "structural": [
                "Reduced nucleus-to-cytoplasm ratio",
                "Assembled actin filaments"
            ],
            "cytoplasmic": [
                "Higher membrane signal",
                "Vesicle-rich cytoplasm"
            ]
        },
        
        ResistanceType.APOPTOSIS_RESISTANCE: {
            "nuclear": [
                "Large, smooth nuclei",
                "Irregular-shaped nuclei",
                "Rounded morphology"
            ],
            "organelle": [
                "Mitochondrial swelling",
                "Uniform cell shape",
                "Low number of apoptotic bodies",
                "Increased surface roughness"
            ]
        },
        
        ResistanceType.METABOLIC_REWIRING: {
            "mitochondrial": [
                "Increased or decreased mitochondrial brightness",
                "Perinuclear mitochondrial clustering"
            ],
            "morphology": [
                "Spindle shapes",
                "High nucleus-to-cytoplasm ratio",
                "Small, compact cells"
            ]
        },
        
        ResistanceType.NCRNA_MEDIATED: {
            "nuclear": [
                "Subtle nuclear texture changes",
                "Slight shifts in cell size distribution",
                "Larger, irregular elongated nuclei",
                "Increased heterochromatin"
            ],
            "molecular": [
                "Altered expression of RNA-associated proteins"
            ]
        },
        
        ResistanceType.EMT_LIKE: {
            "morphological": [
                "Elongated cells",
                "Loss of cell-cell adhesion",
                "Low nucleus-to-cytoplasm ratio",
                "Irregular cell shapes"
            ]
        },
        
        ResistanceType.TARGET_THERAPY: {
            "population": [
                "Intracellular heterogeneity",
                "Partial confluence"
            ],
            "nuclear": [
                "Large irregular nuclei",
                "High nucleus-to-cytoplasm ratio",
                "Altered chromatin compaction"
            ],
            "cellular": [
                "Increased mitotic cells",
                "Abnormal signaling organelles",
                "Small-volume cells"
            ]
        }
    }

class TherapyDatabase:
    """Database of therapeutic recommendations for each resistance type"""
    
    RECOMMENDATIONS = {
        ResistanceType.SENSITIVE: {
            "strategy": "Continue current therapy",
            "drugs": [],
            "rationale": "Cell is responding to current treatment",
            "monitoring": "Continue monitoring response",
            "combination": None
        },
        
        ResistanceType.ENDOCRINE_HORMONE: {
            "strategy": "Endocrine therapy modulation or bypass",
            "first_line": [
                "CDK4/6 inhibitors (Palbociclib, Ribociclib, Abemaciclib)",
                "mTOR inhibitors (Everolimus)",
                "PI3K inhibitors (Alpelisib)"
            ],
            "second_line": [
                "Fulvestrant (ER degrader)",
                "Aromatase inhibitors (Letrozole, Anastrozole)",
                "Switch endocrine agent class"
            ],
            "rationale": "Cells showing hormone-independent growth patterns",
            "biomarkers": ["ER status", "PR status", "HER2 status", "CDK4/6 pathway"],
            "combination": "CDK4/6 inhibitor + Aromatase inhibitor or Fulvestrant",
            "monitoring": [
                "ER/PR expression levels",
                "Cell proliferation markers (Ki-67)",
                "Nuclear morphology changes"
            ]
        },
        
        ResistanceType.DRUG_EFFLUX: {
            "strategy": "Inhibit drug efflux pumps or use pump-independent drugs",
            "first_line": [
                "P-glycoprotein inhibitors (Valspodar, Tariquidar)",
                "BCRP inhibitors (Ko143)",
                "MRP1 inhibitors"
            ],
            "second_line": [
                "Liposomal formulations (bypass efflux)",
                "Antibody-drug conjugates (ADCs)",
                "Small molecule inhibitors not affected by efflux"
            ],
            "rationale": "Increased vesicular activity and membrane pumps detected",
            "biomarkers": ["MDR1/P-gp expression", "BCRP", "MRPs"],
            "combination": "Efflux inhibitor + Original chemotherapy",
            "monitoring": [
                "Intracellular drug accumulation",
                "Vesicle formation",
                "Membrane transporter expression"
            ]
        },
        
        ResistanceType.APOPTOSIS_RESISTANCE: {
            "strategy": "Pro-apoptotic therapy or alternative cell death pathways",
            "first_line": [
                "BCL-2 inhibitors (Venetoclax)",
                "MCL-1 inhibitors",
                "IAP inhibitors (Birinapant)"
            ],
            "second_line": [
                "TRAIL receptor agonists",
                "p53 pathway activators",
                "Ferroptosis inducers (Erastin, RSL3)",
                "Necroptosis inducers"
            ],
            "rationale": "Anti-apoptotic machinery overactive, mitochondrial dysfunction",
            "biomarkers": ["BCL-2 family proteins", "p53 status", "Caspase activity"],
            "combination": "BCL-2 inhibitor + DNA damaging agent",
            "monitoring": [
                "Apoptotic bodies",
                "Mitochondrial membrane potential",
                "Caspase activation",
                "Nuclear fragmentation"
            ]
        },
        
        ResistanceType.METABOLIC_REWIRING: {
            "strategy": "Target altered metabolic pathways",
            "first_line": [
                "Glutaminase inhibitors (CB-839)",
                "FASN inhibitors (TVB-2640)",
                "Metformin (OXPHOS inhibitor)",
                "2-DG (glucose metabolism inhibitor)"
            ],
            "second_line": [
                "LDHA inhibitors",
                "IDH inhibitors (Ivosidenib, Enasidenib)",
                "Mitochondrial complex inhibitors"
            ],
            "rationale": "Altered mitochondrial distribution and metabolic reprogramming",
            "biomarkers": [
                "Mitochondrial activity",
                "Glucose uptake (FDG-PET)",
                "Lactate levels",
                "Glutamine dependence"
            ],
            "combination": "Metabolic inhibitor + Standard chemotherapy",
            "monitoring": [
                "Mitochondrial morphology and clustering",
                "Cell size and compaction",
                "Metabolic activity markers"
            ]
        },
        
        ResistanceType.NCRNA_MEDIATED: {
            "strategy": "Target ncRNA or downstream pathways",
            "first_line": [
                "Antisense oligonucleotides (ASOs)",
                "LncRNA inhibitors",
                "miRNA mimics or antagomirs"
            ],
            "second_line": [
                "Epigenetic modifiers (HDAC inhibitors, DNMTi)",
                "RNA splicing modulators",
                "Transcription factor inhibitors"
            ],
            "rationale": "Altered chromatin structure and RNA-mediated regulation",
            "biomarkers": [
                "ncRNA expression profiling",
                "Chromatin state",
                "Histone modifications"
            ],
            "combination": "Epigenetic modifier + Standard therapy",
            "monitoring": [
                "Nuclear texture changes",
                "Heterochromatin levels",
                "ncRNA expression"
            ]
        },
        
        ResistanceType.EMT_LIKE: {
            "strategy": "Reverse EMT or target mesenchymal phenotype",
            "first_line": [
                "TGF-β inhibitors (Galunisertib)",
                "Src inhibitors (Dasatinib)",
                "FAK inhibitors (Defactinib)"
            ],
            "second_line": [
                "HDAC inhibitors (restore epithelial phenotype)",
                "Wnt pathway inhibitors",
                "Integrin inhibitors",
                "Anti-fibrotic agents"
            ],
            "rationale": "Mesenchymal transition with loss of cell adhesion",
            "biomarkers": [
                "E-cadherin (decreased)",
                "Vimentin (increased)",
                "EMT transcription factors (SNAIL, SLUG, TWIST)"
            ],
            "combination": "EMT inhibitor + Anti-migratory agent",
            "monitoring": [
                "Cell morphology (elongation)",
                "Cell-cell adhesion",
                "Nucleus-to-cytoplasm ratio",
                "Migration/invasion markers"
            ]
        },
        
        ResistanceType.TARGET_THERAPY: {
            "strategy": "Switch target or use multi-targeted approach",
            "first_line": [
                "Alternative pathway inhibitors",
                "Multi-kinase inhibitors (Sorafenib, Regorafenib)",
                "Immune checkpoint inhibitors (if applicable)"
            ],
            "second_line": [
                "Combination targeted therapy",
                "Immunotherapy combinations",
                "Cell cycle inhibitors"
            ],
            "rationale": "Compensatory pathway activation and heterogeneous response",
            "biomarkers": [
                "Pathway phosphorylation status",
                "Receptor expression",
                "Downstream effector activation"
            ],
            "combination": "Dual pathway inhibition",
            "monitoring": [
                "Cell heterogeneity",
                "Mitotic activity",
                "Chromatin compaction",
                "Signaling organelle abnormalities"
            ]
        }
    }
    
    @classmethod
    def get_recommendation(cls, resistance_type: ResistanceType, 
                          current_drug: Optional[str] = None,
                          drug_moa: Optional[str] = None) -> Dict:
        """
        Get therapeutic recommendation based on resistance type
        
        Args:
            resistance_type: Type of resistance detected
            current_drug: Current drug being used
            drug_moa: Mechanism of action of current drug
            
        Returns:
            Dictionary with detailed therapeutic recommendations
        """
        base_recommendation = cls.RECOMMENDATIONS[resistance_type].copy()
        
        # Add context about current therapy
        if current_drug and resistance_type != ResistanceType.SENSITIVE:
            base_recommendation['current_therapy_status'] = {
                'drug': current_drug,
                'moa': drug_moa,
                'recommendation': 'Consider switching or adding agents as listed above'
            }
        
        # Add resistance type name
        base_recommendation['resistance_type'] = resistance_type.name
        
        # Add morphological features
        base_recommendation['detected_features'] = MorphologicalFeatures.FEATURES.get(
            resistance_type, {}
        )
        
        return base_recommendation
    
    @classmethod
    def generate_report(cls, resistance_type: ResistanceType,
                       current_drug: str,
                       drug_moa: str,
                       confidence: float) -> str:
        """Generate a clinical report with recommendations"""
        
        rec = cls.get_recommendation(resistance_type, current_drug, drug_moa)
        
        report = f"""
================================================================================
                    LUMIMAP RESISTANCE ANALYSIS REPORT
================================================================================

CURRENT THERAPY:
  Drug: {current_drug}
  Mechanism of Action: {drug_moa}

RESISTANCE ANALYSIS:
  Detected Resistance Type: {resistance_type.name}
  Confidence: {confidence:.1%}

MORPHOLOGICAL FEATURES DETECTED:
"""
        for category, features in rec['detected_features'].items():
            report += f"\n  {category.upper()}:\n"
            for feature in features:
                report += f"    • {feature}\n"
        
        if resistance_type != ResistanceType.SENSITIVE:
            report += f"""
THERAPEUTIC STRATEGY:
  {rec['strategy']}

FIRST-LINE RECOMMENDATIONS:
"""
            for drug in rec['first_line']:
                report += f"  • {drug}\n"
            
            report += f"""
COMBINATION THERAPY:
  {rec['combination']}

RATIONALE:
  {rec['rationale']}

BIOMARKERS TO TEST:
"""
            for biomarker in rec['biomarkers']:
                report += f"  • {biomarker}\n"
            
            report += f"""
MONITORING PARAMETERS:
"""
            for param in rec['monitoring']:
                report += f"  • {param}\n"
        else:
            report += f"""
RECOMMENDATION:
  {rec['strategy']}
  {rec['rationale']}

MONITORING:
  {rec['monitoring']}
"""
        
        report += """
================================================================================
NOTE: This AI-generated report should be reviewed by oncology specialists.
Recommendations are based on morphological analysis and should be confirmed
with molecular biomarker testing and clinical correlation.
================================================================================
"""
        return report

# Export for use in other modules
__all__ = ['ResistanceType', 'MorphologicalFeatures', 'TherapyDatabase']
