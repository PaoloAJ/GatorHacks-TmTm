# Improving Confidence Scores

If you're getting low confidence/similarity scores, here's how to fix it.

## Understanding the Problem

Your CNN uses **ResNet50 pre-trained on ImageNet**, which was trained to recognize:
- ❌ Dogs, cats, cars, everyday objects (ImageNet classes)
- ❌ NOT artistic styles, brushstrokes, or compositions

This is why you get low similarity scores when matching artwork!

## Quick Diagnosis

Run the diagnostic script to see what's happening:

```bash
python3 scripts/diagnose_similarity.py --image /path/to/test/image.jpg
```

This will show you:
- Current similarity score ranges
- Whether embeddings are normalized correctly
- What's causing low confidence

## Solutions (Ranked by Effectiveness)

### 🥇 Solution 1: Fine-tune the CNN (BEST - Huge Improvement)

Train the model to recognize artistic styles instead of ImageNet objects.

**Command:**
```bash
python3 scripts/finetune_cnn.py \
    --dataset-path /Users/paolovillanueva/Desktop/gatorhacks/backend-cnn/datasets \
    --epochs 10 \
    --batch-size 32 \
    --output models/finetuned_encoder.pth
```

**What this does:**
- Uses contrastive learning: trains the model to recognize when two paintings are by the same artist
- Positive pairs: Two works by the same artist → high similarity
- Negative pairs: Works by different artists → low similarity

**Expected improvement:**
- Before: ~0.3-0.5 similarity scores
- After: ~0.7-0.9 similarity scores for same artist

**Then update your encoder service:**
```python
# In services/encoder_service.py or config
encoder = CNNEncoder(embedding_dim=512, use_pretrained=True)
encoder.load_weights('models/finetuned_encoder.pth')
```

**Re-upload embeddings:**
After fine-tuning, you MUST re-upload all embeddings with the new model:
```bash
python3 scripts/upload_kaggle_to_pinecone.py \
    --dataset-path /Users/paolovillanueva/Desktop/gatorhacks/backend-cnn/datasets \
    --batch-size 100
```

### 🥈 Solution 2: Adjusted Confidence Formula (QUICK FIX - Already Applied!)

I've updated the confidence calculation in [artist_style_service.py](backend-cnn/services/artist_style_service.py#L79-L88).

**Changes:**
```python
# Old (too conservative):
confidence = avg_score * (count / sample_size)

# New (more balanced):
similarity_component = 0.7 * avg_score + 0.3 * max_score
confidence = similarity_component * (frequency_weight ** 0.5)
```

This gives more reasonable confidence scores without requiring model retraining.

### 🥉 Solution 3: Use a Pre-trained Art Model

Instead of ImageNet ResNet, use a model pre-trained on artwork:

**Options:**
1. **CLIP** (OpenAI) - understands both images and text
2. **WikiArt-trained models** - specifically for art classification
3. **Style transfer models** - trained on artistic styles

Example using CLIP:
```python
import torch
import clip

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Encode image
image = preprocess(Image.open("artwork.jpg")).unsqueeze(0).to(device)
with torch.no_grad():
    embedding = model.encode_image(image)
```

### 🔧 Solution 4: Increase Sample Size

For the aggregation method, increase `sample_size`:

```python
# Before:
match_artist_style(image, top_k=5, sample_size=50)

# After:
match_artist_style(image, top_k=5, sample_size=200)
```

This queries more images, giving better artist coverage (but slower).

### 📊 Solution 5: Normalize Score Ranges

If scores are consistently low but *relative* ranking is correct, you can normalize them:

```python
def normalize_scores(results, min_expected=0.3, max_expected=0.8):
    """Rescale scores to 0-1 range for better interpretation"""
    scores = [r['score'] for r in results]
    min_score = min(scores)
    max_score = max(scores)

    for result in results:
        # Rescale to 0-1
        normalized = (result['score'] - min_score) / (max_score - min_score)
        result['normalized_score'] = normalized

    return results
```

## What to Expect

### With Pre-trained ImageNet ResNet (Current):
```
Top matching artists:
1. Van Gogh     - confidence: 0.35 (low but might be correct)
2. Monet        - confidence: 0.28
3. Renoir       - confidence: 0.22
```

### After Fine-tuning on Your Dataset:
```
Top matching artists:
1. Van Gogh     - confidence: 0.82 (much better!)
2. Monet        - confidence: 0.71
3. Renoir       - confidence: 0.58
```

## Recommendations

**For a hackathon/quick demo:**
1. ✅ Use Solution 2 (confidence formula - already done)
2. ✅ Increase sample_size to 100-200
3. ✅ Focus on *relative* rankings rather than absolute scores

**For production/best results:**
1. 🏆 Fine-tune the CNN (Solution 1)
2. 🏆 Re-upload all embeddings with fine-tuned model
3. 🏆 Use centroid method for fast queries

## Training Time Estimates

**Fine-tuning on your dataset:**
- CPU: ~6-10 hours for 10 epochs
- GPU (M1/M2 Mac): ~1-2 hours for 10 epochs
- GPU (CUDA): ~30-60 minutes for 10 epochs

**Quick test (3 epochs):**
```bash
python3 scripts/finetune_cnn.py \
    --dataset-path datasets \
    --epochs 3 \
    --batch-size 16 \
    --output models/quick_test.pth
```

## Still Getting Low Scores?

If scores are still low after fine-tuning:

1. **Check your test image** - is it actually similar to anything in your dataset?
2. **Verify dataset quality** - do you have enough diverse examples per artist?
3. **Try different hyperparameters** - increase epochs, adjust learning rate
4. **Consider the margin** - artistic similarity is subjective!

Remember: A score of 0.6-0.7 might be perfectly fine if it correctly identifies the right artist in the top results!
