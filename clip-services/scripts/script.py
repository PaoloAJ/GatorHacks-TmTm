import torch
import clip
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import re
import json
from tqdm import tqdm

# --- Configuration ---
METADATA_CSV_PATH = "classes.csv"
IMAGE_BASE_PATH = "./"
OUTPUT_JSON_FILE = "art_embeddings_improved.json"
MODEL_NAME = "ViT-L/14"  # You can try ViT-H/14 or ViT-g/14 for higher fidelity
# ----------------------

def extract_year(description_string):
    """Attempts to extract a 4-digit year from the end of the description string."""
    if not isinstance(description_string, str):
        return "Unknown"
    match = re.search(r'(\d{4})$', description_string)
    return match.group(1) if match else "Unknown"

def build_prompts(title, artist, style, year):
    """Creates diverse prompts for text embedding generation."""
    return [
        f"A {style} painting by {artist}, titled '{title}', created in {year}.",
        f"Artwork titled '{title}' by {artist} in the style of {style}.",
        f"Visual depiction of '{title}', characterized by {style} aesthetics.",
        f"A {style} piece called '{title}' ({year}) by {artist}, vivid and detailed.",
        f"Painting '{title}' ({year}), {style} style, made by {artist}.",
    ]

def main():
    print(f"--- Starting Improved Embedding Generation ---")

    # --- Load CLIP Model ---
    print(f"Loading CLIP model ({MODEL_NAME})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load(MODEL_NAME, device=device)


    # --- Load metadata ---
    print(f"Loading metadata from {METADATA_CSV_PATH}...")
    try:
        df = pd.read_csv(METADATA_CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    required_cols = ['filename', 'artist', 'genre', 'description']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns {required_cols}")
        return

    results_data = []
    print(f"Processing {len(df)} images...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Embedding generation"):
        raw_path = str(row['filename']).strip().replace("\\", "/")

        # Primary attempt (relative to IMAGE_BASE_PATH)
        full_image_path = os.path.join(IMAGE_BASE_PATH, raw_path)

        # If that fails, try a few fallbacks
        if not os.path.exists(full_image_path):
            alt_candidates = [
                os.path.join(IMAGE_BASE_PATH, os.path.basename(raw_path)),              # just filename
                os.path.join(IMAGE_BASE_PATH, os.path.dirname(raw_path).title(), os.path.basename(raw_path)),  # e.g. pop_art → Pop_Art
                os.path.join(IMAGE_BASE_PATH, raw_path.lower()),                       # lowercase
                os.path.join(IMAGE_BASE_PATH, raw_path.title()),                       # TitleCase
            ]
            for alt_path in alt_candidates:
                if os.path.exists(alt_path):
                    full_image_path = alt_path
                    break

        # Final existence check
        if not os.path.exists(full_image_path):
            print(f"⚠️ Skipping missing file (not found after normalization): {raw_path}")
            continue

        try:
            # --- Image Embedding ---
            image = Image.open(full_image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_embedding = image_features.cpu().numpy().flatten().tolist()

            # --- Text Embedding (multi-prompt averaging) ---
            style = str(row['genre']).strip("[]'\"")
            title = str(row['description'])
            artist = str(row['artist'])
            year = extract_year(title)
            prompts = build_prompts(title, artist, style, year)

            text_embeddings = []
            with torch.no_grad():
                for prompt in prompts:
                    text_input = clip.tokenize([prompt]).to(device)
                    text_features = model.encode_text(text_input)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_embeddings.append(text_features)

            avg_text_features = torch.mean(torch.stack(text_embeddings), dim=0)
            avg_text_features /= avg_text_features.norm(dim=-1, keepdim=True)
            text_embedding = avg_text_features.cpu().numpy().flatten().tolist()

            results_data.append({
                "title": title,
                "artist": artist,
                "year": year,
                "style": style,
                "filepath": row['filename'],
                "image_embedding": image_embedding,
                "text_embedding": text_embedding
            })

        except Exception as e:
            print(f"Error processing {full_image_path}: {e}")

    # --- Save Results ---
    if results_data:
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"\n✅ Saved embeddings to {OUTPUT_JSON_FILE}")
    else:
        print("No embeddings generated — check input paths.")

if __name__ == "__main__":
    main()
