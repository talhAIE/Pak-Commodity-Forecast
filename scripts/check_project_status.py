"""
Project Status Analysis Script
Analyzes what has been completed in the project
"""

import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'Data'
MODELS_DIR = PROJECT_ROOT / 'models'
FORECASTS_DIR = PROJECT_ROOT / 'forecasts'
DASHBOARD_DIR = PROJECT_ROOT / 'dashboard'

print("=" * 80)
print("PROJECT STATUS ANALYSIS")
print("=" * 80)

# 1. Check Raw Data Files
print("\n1. RAW DATA FILES")
print("-" * 80)
if DATA_DIR.exists():
    data_files = list(DATA_DIR.glob("*"))
    print(f"   Found {len(data_files)} files in Data/ directory:")
    for f in sorted(data_files):
        size = f.stat().st_size / (1024 * 1024)  # MB
        print(f"   [OK] {f.name} ({size:.2f} MB)")
else:
    print("   ❌ Data/ directory not found")

# 2. Check Merged Datasets
print("\n2. MERGED DATASETS")
print("-" * 80)
long_file = PROJECT_ROOT / "merged_export_dataset_2010_2025.csv"
wide_file = PROJECT_ROOT / "merged_export_dataset_wide_2010_2025.csv"

if long_file.exists():
    df_long = pd.read_csv(long_file)
    print(f"   ✅ Long Format Dataset: {long_file.name}")
    print(f"      - Rows: {len(df_long):,}")
    print(f"      - Columns: {list(df_long.columns)}")
    print(f"      - Date range: {df_long['Date'].min()} to {df_long['Date'].max()}")
    print(f"      - Commodities: {df_long['Commodity'].unique().tolist()}")
    print(f"      - Missing values: {df_long.isnull().sum().sum()}")
else:
    print(f"   ❌ Long format dataset not found")

if wide_file.exists():
    df_wide = pd.read_csv(wide_file)
    print(f"\n   ✅ Wide Format Dataset: {wide_file.name}")
    print(f"      - Rows: {len(df_wide):,}")
    print(f"      - Columns: {list(df_wide.columns)}")
    print(f"      - Missing values: {df_wide.isnull().sum().sum()}")
else:
    print(f"   ❌ Wide format dataset not found")

# 3. Check Scripts
print("\n3. IMPLEMENTED SCRIPTS")
print("-" * 80)
scripts_dir = PROJECT_ROOT / "scripts"
if scripts_dir.exists():
    scripts = list(scripts_dir.glob("*.py"))
    print(f"   Found {len(scripts)} Python scripts:")
    for s in sorted(scripts):
        print(f"   ✅ {s.name}")
else:
    print("   ❌ scripts/ directory not found")

# 4. Check Models
print("\n4. TRAINED MODELS")
print("-" * 80)
if MODELS_DIR.exists():
    model_files = list(MODELS_DIR.glob("*"))
    if model_files:
        print(f"   Found {len(model_files)} model files:")
        for m in sorted(model_files):
            size = m.stat().st_size / (1024)  # KB
            print(f"   ✅ {m.name} ({size:.2f} KB)")
    else:
        print("   ⏭️  No trained models found (models/ directory is empty)")
else:
    print("   ❌ models/ directory not found")

# 5. Check Forecasts
print("\n5. GENERATED FORECASTS")
print("-" * 80)
if FORECASTS_DIR.exists():
    forecast_files = list(FORECASTS_DIR.glob("*"))
    if forecast_files:
        print(f"   Found {len(forecast_files)} forecast files:")
        for f in sorted(forecast_files):
            size = f.stat().st_size / (1024)  # KB
            print(f"   ✅ {f.name} ({size:.2f} KB)")
    else:
        print("   ⏭️  No forecast files found (forecasts/ directory is empty)")
else:
    print("   ❌ forecasts/ directory not found")

# 6. Check Dashboard
print("\n6. WEB DASHBOARD")
print("-" * 80)
if DASHBOARD_DIR.exists():
    dashboard_files = list(DASHBOARD_DIR.rglob("*.*"))
    python_files = [f for f in dashboard_files if f.suffix == '.py']
    html_files = [f for f in dashboard_files if f.suffix in ['.html', '.htm']]
    
    if python_files or html_files:
        print(f"   Found dashboard files:")
        if python_files:
            for f in python_files:
                print(f"   ✅ {f.relative_to(PROJECT_ROOT)}")
        if html_files:
            for f in html_files:
                print(f"   ✅ {f.relative_to(PROJECT_ROOT)}")
    else:
        print("   ⏭️  No dashboard implementation found (structure exists but empty)")
else:
    print("   ❌ dashboard/ directory not found")

# 7. Check Documentation
print("\n7. DOCUMENTATION")
print("-" * 80)
doc_dir = PROJECT_ROOT / "Documentation"
if doc_dir.exists():
    doc_files = list(doc_dir.glob("*.md"))
    print(f"   Found {len(doc_files)} documentation files:")
    for d in sorted(doc_files):
        print(f"   ✅ {d.name}")
else:
    print("   ❌ Documentation/ directory not found")

# 8. Check Jupyter Notebooks
print("\n8. JUPYTER NOTEBOOKS")
print("-" * 80)
notebooks = list(PROJECT_ROOT.glob("*.ipynb"))
if notebooks:
    print(f"   Found {len(notebooks)} notebook(s):")
    for n in sorted(notebooks):
        print(f"   ✅ {n.name}")
else:
    print("   ⏭️  No Jupyter notebooks found (Forecasting_Pipeline.ipynb not created yet)")

# 9. Check Requirements
print("\n9. DEPENDENCIES")
print("-" * 80)
req_file = PROJECT_ROOT / "requirements.txt"
if req_file.exists():
    with open(req_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
    print(f"   ✅ requirements.txt found with {len(lines)} packages listed")
else:
    print("   ❌ requirements.txt not found")

# 10. Summary
print("\n" + "=" * 80)
print("PROJECT STATUS SUMMARY")
print("=" * 80)

completed = []
pending = []

# Data Collection & Preprocessing
if long_file.exists() and wide_file.exists():
    completed.append("✅ Data Collection & Preprocessing")
else:
    pending.append("⏭️  Data Collection & Preprocessing")

# Model Development
if MODELS_DIR.exists() and list(MODELS_DIR.glob("*")):
    completed.append("✅ Model Development")
else:
    pending.append("⏭️  Model Development (No models trained yet)")

# Jupyter Notebook
if notebooks:
    completed.append("✅ Forecasting Pipeline Notebook")
else:
    pending.append("⏭️  Forecasting Pipeline Notebook (Not created)")

# Dashboard
if DASHBOARD_DIR.exists() and (python_files or html_files):
    completed.append("✅ Web Dashboard")
else:
    pending.append("⏭️  Web Dashboard (Structure exists, implementation pending)")

# Documentation
if doc_dir.exists() and doc_files:
    completed.append("✅ Documentation")
else:
    pending.append("⏭️  Documentation")

print("\nCOMPLETED:")
for item in completed:
    print(f"   {item}")

print("\nPENDING:")
for item in pending:
    print(f"   {item}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
