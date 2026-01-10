git Deployment Guide - Streamlit Dashboard

This guide will help you deploy your Streamlit dashboard so your friends can access it online.

## Option 1: Streamlit Community Cloud (Recommended - FREE & Easiest)

### Prerequisites
1. **GitHub Account** (free) - https://github.com
2. **Streamlit Account** (free) - https://share.streamlit.io

### Step-by-Step Instructions

#### Step 1: Prepare Your Project for GitHub

1. **Create a `.gitignore` file** (if you don't have one):
   ```gitignore
   # Python
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .Python
   venv/
   env/
   ENV/
   
   # Jupyter Notebook
   .ipynb_checkpoints
   
   # Data files (optional - you may want to exclude large files)
   # *.csv
   # *.xlsx
   # *.pkl
   
   # IDE
   .vscode/
   .idea/
   *.swp
   *.swo
   ```

2. **Initialize Git Repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Streamlit dashboard ready for deployment"
   ```

#### Step 2: Push to GitHub

1. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name it: `commodity-forecasting-dashboard` (or any name you like)
   - Make it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (you already have files)

2. **Push your code to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/commodity-forecasting-dashboard.git
   git branch -M main
   git push -u origin main
   ```

#### Step 3: Deploy on Streamlit Cloud

1. **Sign up/Login to Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Sign in with your GitHub account

2. **Deploy Your App**:
   - Click "New app"
   - Select your repository: `commodity-forecasting-dashboard`
   - **Main file path**: `dashboard/app.py`
   - **App URL**: Choose a custom name (e.g., `commodity-forecasting`)
   - Click "Deploy"

3. **Wait for Deployment**:
   - Streamlit will automatically:
     - Install dependencies from `requirements.txt`
     - Run your app
     - Provide you with a public URL

4. **Share the URL**:
   - Your dashboard will be live at: `https://YOUR-APP-NAME.streamlit.app`
   - Share this URL with your friends!

### Important Notes for Streamlit Cloud

- **File Paths**: Make sure all file paths in your code are relative (they already are!)
- **Data Files**: All CSV files and models must be in the repository
- **File Size Limit**: Free tier has limits, but your files should be fine
- **Auto-updates**: Every time you push to GitHub, Streamlit will redeploy automatically

---

## Option 2: Alternative Deployment Options

### A. Heroku (Free tier available)

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. Deploy using Heroku CLI

### B. AWS/Azure/GCP (Paid, more control)

- More complex setup
- Better for production apps
- Requires cloud account

### C. Local Network Sharing (For testing)

If you just want to share on your local network:

1. **Find your local IP**:
   ```bash
   # Windows
   ipconfig
   # Look for IPv4 Address (e.g., 192.168.1.100)
   ```

2. **Run Streamlit with network access**:
   ```bash
   streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501
   ```

3. **Share the URL**:
   - Friends on same network: `http://YOUR_IP:8501`
   - Example: `http://192.168.1.100:8501`

---

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Make sure all dependencies are in `requirements.txt`
   - Check that file paths are relative

2. **File Not Found**:
   - Verify all data files are committed to GitHub
   - Check file paths in `dashboard/utils/data_loader.py`

3. **Deployment Fails**:
   - Check Streamlit Cloud logs
   - Verify `requirements.txt` is correct
   - Ensure `dashboard/app.py` is the correct main file

### Need Help?

- Streamlit Community: https://discuss.streamlit.io
- Streamlit Docs: https://docs.streamlit.io

---

## Quick Checklist Before Deployment

- [ ] All files committed to Git
- [ ] `requirements.txt` is up to date
- [ ] All data files (CSV) are in repository
- [ ] All model files (.pkl, .json) are in repository
- [ ] File paths are relative (not absolute)
- [ ] Tested locally: `streamlit run dashboard/app.py`

---

**Recommended**: Use **Streamlit Community Cloud** - it's free, easy, and perfect for sharing with friends!
