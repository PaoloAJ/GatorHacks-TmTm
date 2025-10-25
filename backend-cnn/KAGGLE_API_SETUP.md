# Kaggle API Setup Guide

Complete guide to setting up Kaggle API for automated dataset downloads.

---

## Why Use Kaggle API?

Instead of manually downloading 32GB from the website:
- ✅ **Faster**: Direct download via CLI
- ✅ **Resumable**: Can resume if interrupted
- ✅ **Scriptable**: Automate the entire pipeline
- ✅ **Easier**: One command to download + extract

---

## Setup Steps

### 1. Install Dependencies (Already Done!)

The `kaggle` package is already in [requirements.txt](requirements.txt):

```bash
cd backend-cnn
pip install -r requirements.txt
```

### 2. Get Kaggle API Credentials

**Method A: Automated Setup (Recommended)**

```bash
bash scripts/setup_kaggle_api.sh
```

This script will:
- Check if kaggle is installed
- Create `~/.kaggle` directory
- Guide you through credential setup
- Test the API connection

**Method B: Manual Setup**

1. **Login to Kaggle**: https://www.kaggle.com/

2. **Go to Settings**: https://www.kaggle.com/settings

3. **Create API Token**:
   - Scroll to "API" section
   - Click **"Create New API Token"**
   - This downloads `kaggle.json` to your Downloads folder

4. **Move credentials to correct location**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

5. **Test it works**:
   ```bash
   kaggle datasets list --max-size 1
   ```

   If you see a list of datasets, it's working! ✅

### 3. Find Your Dataset ID

Your Kaggle dataset has a unique ID in its URL:

```
https://www.kaggle.com/datasets/USERNAME/DATASET-NAME
                                    ^^^^^^^^^^^^^^^^^^^
                                    This is the dataset ID
```

**Examples:**
- URL: `https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time`
- Dataset ID: `ikarus777/best-artworks-of-all-time`

**How to find it:**
1. Go to your dataset on Kaggle
2. Look at the URL in your browser
3. Copy everything after `/datasets/`

### 4. Download Dataset

Now use the automated download script:

```bash
python scripts/download_kaggle_dataset.py \
    --dataset YOUR_DATASET_ID \
    --output ./datasets
```

**Example with a real dataset:**
```bash
python scripts/download_kaggle_dataset.py \
    --dataset ikarus777/best-artworks-of-all-time \
    --output ./datasets
```

**What it does:**
1. ✅ Downloads dataset from Kaggle
2. ✅ Automatically extracts/unzips
3. ✅ Analyzes folder structure
4. ✅ Shows statistics (# of artists, # of images)
5. ✅ Tells you the exact path for next step

**Time estimate:**
- Small datasets (<1GB): 5-10 minutes
- Medium datasets (1-10GB): 30 min - 1 hour
- Large datasets (10-50GB): 1-3 hours

---

## Troubleshooting

### Error: "401 - Unauthorized"

**Problem**: Invalid or missing API credentials

**Solution**:
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# Should output: -rw------- ... kaggle.json

# If missing, re-download from Kaggle settings
# If exists, check contents:
cat ~/.kaggle/kaggle.json

# Should look like:
# {"username":"YOUR_USERNAME","key":"YOUR_KEY"}
```

### Error: "403 - Forbidden"

**Problem**: Dataset is private or requires competition acceptance

**Solution**:
1. Go to the dataset page on Kaggle
2. Click "Download" button once (accepts terms)
3. Try again

### Error: "404 - Not Found"

**Problem**: Invalid dataset ID

**Solution**:
- Double-check the dataset ID from the URL
- Make sure format is: `username/dataset-name`
- Example: `ikarus777/best-artworks-of-all-time`

### Error: "No space left on device"

**Problem**: Not enough disk space for 32GB dataset

**Solution**:
```bash
# Check available space
df -h

# Free up space or use different output directory:
python scripts/download_kaggle_dataset.py \
    --dataset YOUR_DATASET_ID \
    --output /path/to/larger/disk
```

### Download interrupted

**Problem**: Network issue or stopped mid-download

**Solution**:
```bash
# Just run the command again!
# Kaggle CLI will resume from where it left off
python scripts/download_kaggle_dataset.py \
    --dataset YOUR_DATASET_ID \
    --output ./datasets
```

---

## What Happens After Download?

The download script automatically:

1. **Extracts** the dataset
2. **Analyzes** folder structure:
   ```
   Dataset Structure Analysis
   ========================================
   📁 Root: ./datasets/best-artworks-of-all-time
   📊 Artist folders found: 235

   Top 5 artists:
     1. Pablo_Picasso: 439 images
     2. Vincent_Van_Gogh: 877 images
     3. Claude_Monet: 732 images
     ...

   🖼️  Total images: 80000
   ```

3. **Shows next steps**:
   ```
   Next steps:
     1. Test with: python scripts/upload_kaggle_to_pinecone.py --dataset-path ./datasets/best-artworks-of-all-time --dry-run
     2. Upload: python scripts/upload_kaggle_to_pinecone.py --dataset-path ./datasets/best-artworks-of-all-time
   ```

Just copy-paste the commands! 🚀

---

## Alternative: Manual Download

If you prefer not to use the API:

1. Go to your dataset on Kaggle
2. Click **"Download"** button
3. Wait for download to complete (~32GB)
4. Extract the ZIP file:
   ```bash
   unzip your-dataset.zip -d ./datasets
   ```
5. Continue with upload script

**Note**: API method is recommended for large datasets!

---

## Security Note

Your `kaggle.json` contains your API key. Keep it secure:

```bash
# Correct permissions (only you can read)
chmod 600 ~/.kaggle/kaggle.json

# Never commit to git
echo ".kaggle/" >> .gitignore
```

The setup script automatically sets correct permissions.

---

## Complete Workflow

```bash
# 1. Setup Kaggle API (one-time)
bash scripts/setup_kaggle_api.sh

# 2. Download dataset
python scripts/download_kaggle_dataset.py \
    --dataset YOUR_DATASET_ID \
    --output ./datasets

# 3. Test dataset structure
python scripts/upload_kaggle_to_pinecone.py \
    --dataset-path ./datasets/YOUR_DATASET_FOLDER \
    --dry-run

# 4. Upload to Pinecone
python scripts/upload_kaggle_to_pinecone.py \
    --dataset-path ./datasets/YOUR_DATASET_FOLDER

# 5. Start API server
python main.py
```

That's it! 🎉

---

## Quick Reference

```bash
# Test Kaggle API
kaggle datasets list --max-size 1

# List your downloaded datasets
ls -la ./datasets

# Check dataset info before downloading
kaggle datasets files -d DATASET_ID

# Download specific size
kaggle datasets download -d DATASET_ID --unzip

# Get help
python scripts/download_kaggle_dataset.py --help
```

---

See [QUICK_START.md](QUICK_START.md) for the complete pipeline overview.
