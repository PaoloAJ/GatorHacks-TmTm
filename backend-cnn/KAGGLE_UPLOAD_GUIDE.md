# Kaggle Dataset → Pinecone Upload Guide

Complete walkthrough for processing your 32GB Kaggle artist dataset and uploading embeddings to Pinecone.

## 📋 Overview

**What you're doing:**
- Processing 80,000 images (~32GB) from Kaggle
- Converting each to 512-dimensional embeddings
- Uploading ~164MB of embeddings to Pinecone (free tier)
- Querying for similar artists via API

**Time estimate:** 2-4 hours for full upload (depends on CPU/GPU)

---

## 🚀 Step-by-Step Process

### 1. Install Dependencies

```bash
cd backend-cnn
pip install -r requirements.txt
```

This installs:
- `pinecone-client` - Pinecone SDK
- `tqdm` - Progress bars
- All existing dependencies

### 2. Setup Pinecone Account (5 minutes)

1. **Create account:** https://www.pinecone.io/
   - No credit card required for free tier

2. **Get API key:**
   - Login → "API Keys" section
   - Copy your API key (starts with `pcsk_...`)

3. **Update `.env` file:**
   ```bash
   # Open backend-cnn/.env and add:
   PINECONE_API_KEY=pcsk_your_actual_key_here
   PINECONE_INDEX_NAME=artist-styles
   ```

### 3. Prepare Your Kaggle Dataset

**Expected folder structure (most common):**
```
/path/to/kaggle/dataset/
├── Vincent_Van_Gogh/
│   ├── painting1.jpg
│   ├── painting2.jpg
│   └── ...
├── Pablo_Picasso/
│   ├── painting1.jpg
│   └── ...
└── ...
```

**If your structure is different:**
- Edit `scripts/upload_kaggle_to_pinecone.py`
- Modify the `get_image_files()` method
- Or let me know and I'll help adjust it!

### 4. Test Run (Dry Run)

**Before uploading everything, do a dry run to verify:**

```bash
cd backend-cnn
python scripts/upload_kaggle_to_pinecone.py \
    --dataset-path /path/to/your/kaggle/dataset \
    --dry-run
```

This will:
- ✅ Scan your dataset structure
- ✅ Count total images
- ✅ Show artist distribution
- ❌ NOT upload anything

**Expected output:**
```
Found 235 artist folders
Found 80000 images

Artist distribution (top 10):
  Pablo Picasso: 439 images
  Vincent Van Gogh: 877 images
  ...
```

### 5. Full Upload (2-4 hours)

**Once dry run looks good, do the full upload:**

```bash
python scripts/upload_kaggle_to_pinecone.py \
    --dataset-path /path/to/your/kaggle/dataset \
    --batch-size 100
```

**What happens:**
1. ✅ Creates Pinecone index (if doesn't exist)
2. 🔄 Processes images in batches of 100
3. 🧠 Encodes each image → 512-dim embedding
4. ☁️ Uploads to Pinecone
5. 📊 Shows progress bar

**Monitoring:**
- Progress bar shows: `Processing: 45%|████▌     | 36000/80000 [01:23:45<01:36:12, 7.6it/s]`
- Errors are logged to `upload_errors.log`
- You can safely `Ctrl+C` and resume later (Pinecone keeps uploaded data)

**Performance tips:**
- **GPU recommended** for faster encoding (30-40 images/sec)
- **CPU only:** 5-10 images/sec (slower but works)
- Use `--batch-size 50` if memory is limited

### 6. Verify Upload

After upload completes, check Pinecone:

**Option A: Dashboard**
- Login to Pinecone
- Go to your `artist-styles` index
- Should show ~80,000 vectors

**Option B: API Check**
```python
# In Python console
from services.pinecone_service import PineconeService
pc = PineconeService()
stats = pc.get_index_stats()
print(stats)  # Should show total_vector_count: 80000
```

### 7. Test the API

**Start your server:**
```bash
cd backend-cnn
python main.py
```

**Try the new endpoint:**
1. Go to http://localhost:8000/docs
2. Find `/api/v1/images/find-similar-artist`
3. Upload any artwork image
4. Get back top 10 similar artists!

**Example response:**
```json
{
  "query_filename": "my_artwork.jpg",
  "matches": [
    {
      "artist_name": "Vincent Van Gogh",
      "similarity_score": 0.89,
      "image_filename": "starry_night.jpg",
      "image_path": "Vincent_Van_Gogh/starry_night.jpg"
    },
    ...
  ],
  "total_results": 10
}
```

---

## 🛠 Troubleshooting

### "ModuleNotFoundError: No module named 'pinecone'"
```bash
pip install pinecone-client
```

### "Unauthorized: Invalid API key"
- Check `.env` file has correct API key
- Restart server after updating `.env`

### "No images found"
- Verify `--dataset-path` is correct
- Check folder structure matches expected format
- Run with `--dry-run` first

### Upload interrupted (Ctrl+C)
- **No problem!** Pinecone keeps uploaded data
- Just run the script again - it will skip existing IDs
- Or use `pc.delete_all()` to start fresh

### Out of memory
- Reduce `--batch-size` to 50 or 25
- Close other applications
- Process smaller subset first

---

## 📊 Cost & Limits

**Pinecone Free Tier:**
- ✅ 100,000 vectors (you need 80K)
- ✅ 1 index
- ✅ Unlimited queries
- ✅ No time limit
- ❌ No credit card required

**Your usage:**
- 80,000 embeddings × 512 floats × 4 bytes = **164MB**
- Fits comfortably in free tier! 🎉

---

## 🎯 Next Steps

After upload completes:

1. **Integrate with frontend:**
   - Call `/find-similar-artist` from Next.js
   - Display artist recommendations

2. **Advanced features:**
   - Filter by artist: `?artist=Van+Gogh`
   - Adjust `top_k` parameter
   - Aggregate by artist (group results)

3. **Optimize queries:**
   - Cache popular queries
   - Pre-compute embeddings for user uploads
   - Add Redis for faster repeated queries

---

## 📝 Quick Command Reference

```bash
# Dry run (test dataset scanning)
python scripts/upload_kaggle_to_pinecone.py --dataset-path /path/to/dataset --dry-run

# Full upload
python scripts/upload_kaggle_to_pinecone.py --dataset-path /path/to/dataset

# Custom batch size (if low memory)
python scripts/upload_kaggle_to_pinecone.py --dataset-path /path/to/dataset --batch-size 50

# Check index stats
python -c "from services.pinecone_service import PineconeService; print(PineconeService().get_index_stats())"

# Delete all vectors (start fresh)
python -c "from services.pinecone_service import PineconeService; PineconeService().delete_all()"
```

---

## 📞 Need Help?

If you encounter issues:
1. Check error logs in `upload_errors.log`
2. Verify dataset structure with `--dry-run`
3. Test with a small subset first (one artist folder)
4. Ask me for help! 🤖

Good luck! 🚀
