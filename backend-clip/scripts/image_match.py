import torch
import clip
from PIL import Image
import json
import argparse
import os

def load_data(filepath):
    """Load precomputed embeddings from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_input_embedding(image_path, model, preprocess, device):
    """Generate normalized embedding for an input image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    except Exception as e:
        print(f"Error processing input image: {e}")
        return None

def find_top_matches(input_embedding, db_data, alpha, device, top_k=5):
    """Compute hybrid similarity with score normalization."""
    db_image_embs = torch.tensor([item["image_embedding"] for item in db_data], dtype=torch.float32).to(device)
    db_text_embs = torch.tensor([item["text_embedding"] for item in db_data], dtype=torch.float32).to(device)

    db_image_embs /= db_image_embs.norm(dim=-1, keepdim=True)
    db_text_embs /= db_text_embs.norm(dim=-1, keepdim=True)

    input_embedding = input_embedding.float()
    db_image_embs = db_image_embs.float()

    # Cosine similarity
    visual_scores = (input_embedding @ db_image_embs.T).squeeze()
    semantic_scores = (input_embedding @ db_text_embs.T).squeeze()

    # Normalize scores to avoid scale bias
    visual_scores = (visual_scores - visual_scores.mean()) / (visual_scores.std() + 1e-8)
    semantic_scores = (semantic_scores - semantic_scores.mean()) / (semantic_scores.std() + 1e-8)

    # Hybrid scoring
    final_scores = alpha * visual_scores + (1 - alpha) * semantic_scores
    top_indices = torch.argsort(final_scores, descending=True)[:top_k]

    return [
        {"rank": i + 1, "score": final_scores[idx].item(), "data": db_data[idx]}
        for i, idx in enumerate(top_indices)
    ]

def main():
    parser = argparse.ArgumentParser(description="Find similar art using hybrid CLIP embeddings.")
    parser.add_argument("image_path", type=str, help="Path to the query image.")
    parser.add_argument("--embeddings_file", type=str, default="art_embeddings_improved.json")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for visual similarity (0-1).")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("Error: alpha must be between 0.0 and 1.0")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    db_data = load_data(args.embeddings_file)
    if not db_data:
        return

    print(f"\nProcessing input: {args.image_path}")
    input_emb = get_input_embedding(args.image_path, model, preprocess, device)
    if input_emb is None:
        return

    print(f"\nSearching for top {args.top_k} matches (alpha={args.alpha})...")
    results = find_top_matches(input_emb, db_data, args.alpha, device, top_k=args.top_k)

    print("\n--- Top Matches ---")
    for res in results:
        data = res["data"]
        print(f"\nRank {res['rank']} | Score: {res['score']:.4f}")
        print(f"  Artist: {data['artist']}")
        print(f"  Title:  {data['title']}")
        print(f"  Year:   {data['year']}")
        print(f"  Style:  {data['style']}")
        print(f"  File:   {data['filepath']}")

if __name__ == "__main__":
    main()
