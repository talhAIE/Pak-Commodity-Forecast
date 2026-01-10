# 🚀 Quick Deployment Guide

## Fastest Way: Streamlit Community Cloud (5 minutes!)

### Step 1: Push to GitHub
```bash
# If you haven't initialized git yet:
git init
git add .
git commit -m "Ready for deployment"

# Create a new repository on GitHub.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to: https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. **Main file path**: `dashboard/app.py`
6. Click "Deploy"

### Step 3: Share!
Your dashboard will be live at: `https://YOUR-APP-NAME.streamlit.app`

---

## ⚠️ Important Before Deploying

Make sure these files are in your GitHub repository:
- ✅ `dashboard/app.py`
- ✅ `dashboard/utils/` (all files)
- ✅ `models/best_model_lgbm.pkl`
- ✅ `models/feature_names_lgbm.json`
- ✅ `models/model_metadata.json`
- ✅ `merged_export_dataset_2010_2025.csv`
- ✅ `merged_export_dataset_wide_2010_2025.csv`
- ✅ `requirements.txt`

---

## 📝 Full Instructions

See `DEPLOYMENT_GUIDE.md` for detailed instructions and troubleshooting.
