"""
View All Drugs in Database
===========================
Lists all available drugs organized by MOA
"""

from drug_database import DRUG_DATABASE, list_all_moa_classes, list_drugs_by_moa

print("="*70)
print("COMPLETE DRUG DATABASE")
print("="*70)
print(f"Total drugs: {len(DRUG_DATABASE)}")
print()

# Get all MOA classes
moa_classes = list_all_moa_classes()

for moa in moa_classes:
    # Get drugs for this MOA
    drugs = list_drugs_by_moa(moa)
    
    print(f"\n{'='*70}")
    print(f"{moa.upper()}")
    print(f"{'='*70}")
    
    for i, drug_name in enumerate(drugs, 1):
        drug_info = DRUG_DATABASE[drug_name]
        print(f"{i:2d}. {drug_name:30s} ({drug_info['concentration']})")
        print(f"    {drug_info['description']}")

print("\n" + "="*70)
print(f"Total: {len(DRUG_DATABASE)} drugs across {len(moa_classes)} MOA categories")
print("="*70)
