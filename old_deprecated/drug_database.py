"""
Comprehensive Drug Database
============================
Central database of anti-cancer drugs with their Mechanisms of Action (MOA)
and typical concentrations for cell-based assays.

Based on BBBC021 dataset and common chemotherapy compounds.

Usage:
    from drug_database import DRUG_DATABASE, get_drug_info, list_drugs_by_moa
"""

# =============================================================================
# COMPREHENSIVE DRUG DATABASE
# =============================================================================

DRUG_DATABASE = {
    # =========================================================================
    # ACTIN DISRUPTORS
    # =========================================================================
    'cytochalasin B': {
        'moa': 'Actin disruptors',
        'concentration': '10 nM',
        'description': 'Inhibits actin polymerization, disrupts cytoskeleton'
    },
    'cytochalasin D': {
        'moa': 'Actin disruptors',
        'concentration': '1 μM',
        'description': 'Potent actin filament disruptor'
    },
    'latrunculin A': {
        'moa': 'Actin disruptors',
        'concentration': '0.5 μM',
        'description': 'Binds actin monomers, prevents polymerization'
    },
    'latrunculin B': {
        'moa': 'Actin disruptors',
        'concentration': '1 μM',
        'description': 'Sequesters actin monomers'
    },
    
    # =========================================================================
    # AURORA KINASE INHIBITORS
    # =========================================================================
    'alisertib': {
        'moa': 'Aurora kinase inhibitors',
        'concentration': '100 nM',
        'description': 'Aurora A kinase inhibitor, mitotic arrest'
    },
    'barasertib': {
        'moa': 'Aurora kinase inhibitors',
        'concentration': '10 nM',
        'description': 'Aurora B kinase inhibitor'
    },
    'VX-680': {
        'moa': 'Aurora kinase inhibitors',
        'concentration': '50 nM',
        'description': 'Pan-Aurora kinase inhibitor'
    },
    
    # =========================================================================
    # CHOLESTEROL-LOWERING
    # =========================================================================
    'lovastatin': {
        'moa': 'Cholesterol-lowering',
        'concentration': '10 μM',
        'description': 'HMG-CoA reductase inhibitor, statin'
    },
    'simvastatin': {
        'moa': 'Cholesterol-lowering',
        'concentration': '10 μM',
        'description': 'Statin, inhibits cholesterol synthesis'
    },
    'atorvastatin': {
        'moa': 'Cholesterol-lowering',
        'concentration': '5 μM',
        'description': 'Potent statin'
    },
    
    # =========================================================================
    # DMSO (CONTROL)
    # =========================================================================
    'DMSO': {
        'moa': 'DMSO',
        'concentration': '0.1%',
        'description': 'Negative control, vehicle only'
    },
    
    # =========================================================================
    # DNA DAMAGE
    # =========================================================================
    'doxorubicin': {
        'moa': 'DNA damage',
        'concentration': '1 μM',
        'description': 'Topoisomerase II inhibitor, DNA intercalator'
    },
    'cisplatin': {
        'moa': 'DNA damage',
        'concentration': '10 μM',
        'description': 'Platinum compound, DNA crosslinker'
    },
    'carboplatin': {
        'moa': 'DNA damage',
        'concentration': '50 μM',
        'description': 'Platinum analog, less toxic than cisplatin'
    },
    'etoposide': {
        'moa': 'DNA damage',
        'concentration': '10 μM',
        'description': 'Topoisomerase II inhibitor'
    },
    'mitomycin C': {
        'moa': 'DNA damage',
        'concentration': '1 μM',
        'description': 'DNA crosslinker, alkylating agent'
    },
    'bleomycin': {
        'moa': 'DNA damage',
        'concentration': '10 μM',
        'description': 'Induces DNA strand breaks'
    },
    
    # =========================================================================
    # DNA REPLICATION
    # =========================================================================
    'gemcitabine': {
        'moa': 'DNA replication',
        'concentration': '10 nM',
        'description': 'Nucleoside analog, inhibits DNA synthesis'
    },
    '5-fluorouracil': {
        'moa': 'DNA replication',
        'concentration': '10 μM',
        'description': '5-FU, thymidylate synthase inhibitor'
    },
    'cytarabine': {
        'moa': 'DNA replication',
        'concentration': '1 μM',
        'description': 'Ara-C, DNA polymerase inhibitor'
    },
    'methotrexate': {
        'moa': 'DNA replication',
        'concentration': '100 nM',
        'description': 'Antifolate, inhibits DHFR'
    },
    'hydroxyurea': {
        'moa': 'DNA replication',
        'concentration': '100 μM',
        'description': 'Ribonucleotide reductase inhibitor'
    },
    
    # =========================================================================
    # EG5 INHIBITORS (Kinesin spindle protein)
    # =========================================================================
    'monastrol': {
        'moa': 'Eg5 inhibitors',
        'concentration': '100 μM',
        'description': 'Eg5/KSP inhibitor, monopolar spindles'
    },
    'ispinesib': {
        'moa': 'Eg5 inhibitors',
        'concentration': '10 nM',
        'description': 'Potent Eg5 inhibitor'
    },
    'STLC': {
        'moa': 'Eg5 inhibitors',
        'concentration': '10 μM',
        'description': 'S-trityl-L-cysteine, Eg5 inhibitor'
    },
    
    # =========================================================================
    # KINASE INHIBITORS
    # =========================================================================
    'imatinib': {
        'moa': 'Kinase inhibitors',
        'concentration': '10 μM',
        'description': 'BCR-ABL tyrosine kinase inhibitor (Gleevec)'
    },
    'erlotinib': {
        'moa': 'Kinase inhibitors',
        'concentration': '10 μM',
        'description': 'EGFR tyrosine kinase inhibitor'
    },
    'sorafenib': {
        'moa': 'Kinase inhibitors',
        'concentration': '10 μM',
        'description': 'Multi-kinase inhibitor'
    },
    'sunitinib': {
        'moa': 'Kinase inhibitors',
        'concentration': '10 μM',
        'description': 'VEGFR and PDGFR inhibitor'
    },
    'dasatinib': {
        'moa': 'Kinase inhibitors',
        'concentration': '100 nM',
        'description': 'BCR-ABL and Src family kinase inhibitor'
    },
    'lapatinib': {
        'moa': 'Kinase inhibitors',
        'concentration': '10 μM',
        'description': 'HER2/EGFR dual kinase inhibitor'
    },
    
    # =========================================================================
    # MICROTUBULE DESTABILIZERS
    # =========================================================================
    'nocodazole': {
        'moa': 'Microtubule destabilizers',
        'concentration': '1 μM',
        'description': 'Microtubule depolymerization, mitotic arrest'
    },
    'colchicine': {
        'moa': 'Microtubule destabilizers',
        'concentration': '10 nM',
        'description': 'Binds tubulin, prevents polymerization'
    },
    'vinblastine': {
        'moa': 'Microtubule destabilizers',
        'concentration': '10 nM',
        'description': 'Vinca alkaloid, inhibits microtubule assembly'
    },
    'vincristine': {
        'moa': 'Microtubule destabilizers',
        'concentration': '10 nM',
        'description': 'Vinca alkaloid, microtubule poison'
    },
    'vinorelbine': {
        'moa': 'Microtubule destabilizers',
        'concentration': '10 nM',
        'description': 'Semi-synthetic vinca alkaloid'
    },
    
    # =========================================================================
    # MICROTUBULE STABILIZERS
    # =========================================================================
    'paclitaxel': {
        'moa': 'Microtubule stabilizers',
        'concentration': '10 nM',
        'description': 'Taxane, stabilizes microtubules (Taxol)'
    },
    'docetaxel': {
        'moa': 'Microtubule stabilizers',
        'concentration': '10 nM',
        'description': 'Semi-synthetic taxane (Taxotere)'
    },
    'cabazitaxel': {
        'moa': 'Microtubule stabilizers',
        'concentration': '10 nM',
        'description': 'Next-generation taxane'
    },
    'ixabepilone': {
        'moa': 'Microtubule stabilizers',
        'concentration': '10 nM',
        'description': 'Epothilone B analog'
    },
    
    # =========================================================================
    # PKC ACTIVATORS (Protein Kinase C)
    # =========================================================================
    'phorbol 12-myristate 13-acetate': {
        'moa': 'PKC activators',
        'concentration': '100 nM',
        'description': 'PMA, potent PKC activator'
    },
    'bryostatin 1': {
        'moa': 'PKC activators',
        'concentration': '10 nM',
        'description': 'PKC modulator'
    },
    
    # =========================================================================
    # PROTEIN DEGRADATION
    # =========================================================================
    'bortezomib': {
        'moa': 'Protein degradation',
        'concentration': '100 nM',
        'description': 'Proteasome inhibitor (Velcade)'
    },
    'carfilzomib': {
        'moa': 'Protein degradation',
        'concentration': '10 nM',
        'description': 'Irreversible proteasome inhibitor'
    },
    'MG-132': {
        'moa': 'Protein degradation',
        'concentration': '10 μM',
        'description': 'Proteasome inhibitor'
    },
    'lactacystin': {
        'moa': 'Protein degradation',
        'concentration': '10 μM',
        'description': 'Proteasome inhibitor'
    },
    
    # =========================================================================
    # PROTEIN SYNTHESIS
    # =========================================================================
    'cycloheximide': {
        'moa': 'Protein synthesis',
        'concentration': '10 μM',
        'description': 'Inhibits eukaryotic protein synthesis'
    },
    'puromycin': {
        'moa': 'Protein synthesis',
        'concentration': '10 μM',
        'description': 'Aminonucleoside antibiotic, protein synthesis inhibitor'
    },
    'anisomycin': {
        'moa': 'Protein synthesis',
        'concentration': '10 μM',
        'description': 'Inhibits peptidyl transferase'
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_drug_info(drug_name):
    """Get drug information by name (case-insensitive)"""
    # Try exact match first
    if drug_name in DRUG_DATABASE:
        return DRUG_DATABASE[drug_name]
    
    # Try case-insensitive match
    drug_lower = drug_name.lower()
    for key, value in DRUG_DATABASE.items():
        if key.lower() == drug_lower:
            return value
    
    return None

def list_drugs_by_moa(moa_class=None):
    """List all drugs, optionally filtered by MOA class"""
    if moa_class is None:
        return sorted(DRUG_DATABASE.keys())
    
    drugs = [name for name, info in DRUG_DATABASE.items() 
             if info['moa'] == moa_class]
    return sorted(drugs)

def list_all_moa_classes():
    """Get list of all unique MOA classes"""
    moa_classes = set(info['moa'] for info in DRUG_DATABASE.values())
    return sorted(moa_classes)

def get_drug_count_by_moa():
    """Get count of drugs per MOA class"""
    from collections import Counter
    moa_list = [info['moa'] for info in DRUG_DATABASE.values()]
    return dict(Counter(moa_list))

def print_database_summary():
    """Print a summary of the drug database"""
    print("="*70)
    print("DRUG DATABASE SUMMARY")
    print("="*70)
    print(f"Total drugs: {len(DRUG_DATABASE)}")
    print(f"\nDrugs by MOA class:")
    
    counts = get_drug_count_by_moa()
    for moa, count in sorted(counts.items()):
        print(f"  {moa:35s} {count:3d} drugs")
    
    print("\n" + "="*70)

def add_custom_drug(name, moa, concentration, description=""):
    """Add a custom drug to the database (runtime only, not persistent)"""
    DRUG_DATABASE[name] = {
        'moa': moa,
        'concentration': concentration,
        'description': description
    }
    print(f"✓ Added drug: {name} ({moa}, {concentration})")

# =============================================================================
# VALIDATION
# =============================================================================

def validate_database():
    """Validate that all drugs have valid MOA classes"""
    
    # Valid MOA classes (must match your model's training classes)
    VALID_MOA_CLASSES = [
        'Actin disruptors',
        'Aurora kinase inhibitors',
        'Cholesterol-lowering',
        'DMSO',
        'DNA damage',
        'DNA replication',
        'Eg5 inhibitors',
        'Kinase inhibitors',
        'Microtubule destabilizers',
        'Microtubule stabilizers',
        'PKC activators',
        'Protein degradation',
        'Protein synthesis'
    ]
    
    errors = []
    for drug_name, info in DRUG_DATABASE.items():
        if info['moa'] not in VALID_MOA_CLASSES:
            errors.append(f"  ✗ {drug_name}: Invalid MOA '{info['moa']}'")
    
    if errors:
        print("Validation FAILED:")
        for error in errors:
            print(error)
        return False
    else:
        print(f"✓ Database validated: {len(DRUG_DATABASE)} drugs, all have valid MOA classes")
        return True

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Print summary
    print_database_summary()
    
    # Validate
    print()
    validate_database()
    
    # Example: List drugs by MOA
    print("\n" + "="*70)
    print("EXAMPLE: Microtubule stabilizers")
    print("="*70)
    for drug in list_drugs_by_moa('Microtubule stabilizers'):
        info = DRUG_DATABASE[drug]
        print(f"  {drug:30s} {info['concentration']:10s} - {info['description']}")
