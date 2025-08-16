# 🚀 Deployment Guide for AI-Powered Career Platform

## 📋 Pre-Deployment Checklist

✅ **Files Ready:**
- `app.py` - Main application file
- `requirements.txt` - Dependencies
- `README.md` - Project documentation
- `.streamlit/config.toml` - Streamlit configuration
- `.streamlit/secrets.toml.example` - Secrets template

## 🔧 Step 1: GitHub Repository Setup

1. **Create a new GitHub repository:**
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name: `ai-powered-career-platform`
   - Make it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (you already have one)

2. **Upload your code:**
   ```bash
   # In your project directory
   git init
   git add .
   git commit -m "Initial commit: AI-Powered Career Platform"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ai-powered-career-platform.git
   git push -u origin main
   ```

## ☁️ Step 2: Streamlit Cloud Deployment

1. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy your app:**
   - Click "New app"
   - **Repository:** `YOUR_USERNAME/ai-powered-career-platform`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Choose a custom name (e.g., `ai-career-hub`)

3. **Configure secrets:**
   - In your app dashboard, go to "Settings" → "Secrets"
   - Add your API key in TOML format:
   ```toml
   [api_keys]
   OPENROUTER_API_KEY = "your_actual_openrouter_api_key_here"
   ```

## 🔑 Step 3: Get OpenRouter API Key

1. **Visit OpenRouter:**
   - Go to [openrouter.ai](https://openrouter.ai)
   - Sign up for an account
   - Go to "Keys" section
   - Create a new API key
   - Copy the key for use in secrets

## 🎯 Step 4: Test Your Deployment

1. **Wait for deployment** (usually 2-3 minutes)
2. **Visit your app URL** (provided by Streamlit Cloud)
3. **Test key features:**
   - Upload a resume
   - Try AI analysis features
   - Check all navigation pages

## 🔧 Step 5: Post-Deployment Updates

**To update your app:**
1. Make changes to your local code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push
   ```
3. Streamlit Cloud will automatically redeploy

## 🆘 Troubleshooting

**Common Issues:**

1. **App won't start:**
   - Check requirements.txt for correct package versions
   - Verify all imports are available

2. **API key not working:**
   - Ensure secrets are properly formatted in TOML
   - Check API key is valid and has credits

3. **Memory issues:**
   - Streamlit Cloud has memory limits
   - Consider optimizing large data processing

## 📱 Your App Will Be Available At:
`https://YOUR_APP_NAME.streamlit.app`

## 🎉 Success!
Your AI-Powered Career Platform is now live and accessible worldwide!
