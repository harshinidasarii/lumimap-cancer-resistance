"""
Master Analysis Runner
======================

Runs all comprehensive analyses in sequence:
1. MOA Performance Analysis
2. Embedding Space Analysis
3. GradCAM Activation Analysis
4. Ablation Study

Creates complete publication-ready analysis suite.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and report status"""
    print("\n" + "="*70)
    print(f"🚀 RUNNING: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"✅ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR")
        print(f"   Error: {e}")
        return False

def main():
    start_time = datetime.now()
    
    print("="*70)
    print("📊 LUMIMAP COMPREHENSIVE ANALYSIS SUITE")
    print("="*70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Create output directory
    os.makedirs('./output/analysis_results', exist_ok=True)
    
    # Analysis scripts to run
    analyses = [
        ('analysis_moa_performance.py', 'MOA Performance Analysis'),
        ('analysis_embedding_space.py', 'Embedding Space Analysis'),
        ('analysis_gradcam.py', 'GradCAM Activation Analysis'),
        ('analysis_ablation.py', 'Ablation Study')
    ]
    
    results = {}
    
    # Run each analysis
    for script, description in analyses:
        success = run_script(script, description)
        results[description] = success
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("📊 ANALYSIS SUITE SUMMARY")
    print("="*70)
    
    for description, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status}: {description}")
    
    print()
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()
    print(f"All results saved to: ./output/analysis_results/")
    print("="*70)
    
    # List all generated files
    print("\n📁 Generated Files:")
    analysis_dir = './output/analysis_results'
    if os.path.exists(analysis_dir):
        for file in sorted(os.listdir(analysis_dir)):
            if file.endswith(('.png', '.csv')):
                print(f"   • {file}")
    
    print("\n✅ COMPLETE!")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count == total_count:
        print(f"\n🎉 All {total_count} analyses completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {success_count}/{total_count} analyses completed successfully")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
