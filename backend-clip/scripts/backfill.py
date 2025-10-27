import requests
import json
import os
import time

# --- Main Script ---
OUTPUT_JSON = "uploaded_map.json"

def decode_id_to_path(hex_id):
    """Decodes a hex-encoded ID back into its original UTF-8 file path."""
    try:
        return bytes.fromhex(hex_id).decode('utf-8')
    except (ValueError, UnicodeDecodeError) as e:
        print(f"    ⚠️ Could not decode ID: {hex_id[:20]}... ({e})")
        return None

def fetch_and_rebuild_map():
    """
    Fetches all images from Cloudflare and rebuilds the
    path-to-URL map from the hex-encoded IDs.
    """
    print("Authenticating to get batch token...")
    try:
        BATCH_TOKEN = "REDACTED"
    except Exception as e:
        print(f"❌ Failed to get batch token: {e}")
        return
        
    LIST_URL = "https://batch.imagedelivery.net/images/v2"
    HEADERS = {"Authorization": f"Bearer {BATCH_TOKEN}"}
    
    print("📡 Fetching ALL existing image records from Cloudflare...")
    print("   This may take a few minutes if you have many pages...")
    
    rebuilt_map = {}
    continuation = None
    page_count = 0
    
    with requests.Session() as session:
        session.headers.update(HEADERS)
        
        while True:
            page_count += 1
            params = {"per_page": 1000} # Use 1k, 10k can timeout
            if continuation:
                params["continuation_token"] = continuation

            try:
                resp = session.get(LIST_URL, params=params, timeout=60)
                
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    print(f"⚠️ Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to fetch page {page_count}: {e}")
                time.sleep(5)
                continue
            
            result = data.get("result", {})
            images = result.get("images", [])
            
            if not images and not result.get("continuation_token"):
                if page_count == 1:
                    print("...No images found on Cloudflare.")
                break # All done

            for img in images:
                hex_id = img.get("id")
                variants = img.get("variants")
                
                if not (hex_id and variants):
                    continue

                # Get the first (usually public) URL
                url = variants[0]
                rel_path = decode_id_to_path(hex_id)
                
                if rel_path:
                    rebuilt_map[rel_path] = url
            
            continuation = result.get("continuation_token")
            if not continuation:
                break # This was the last page
            
            print(f"   ...Fetched page {page_count} (found {len(rebuilt_map)} images so far)")

    print(f"\n✅ Rebuilt map with {len(rebuilt_map)} total entries from Cloudflare.")

    # --- Merge with existing JSON map ---
    if os.path.exists(OUTPUT_JSON):
        print(f"💾 Loading existing '{OUTPUT_JSON}' to merge...")
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            try:
                old_map = json.load(f)
            except json.JSONDecodeError:
                print(f"   ⚠️ Warning: '{OUTPUT_JSON}' was corrupt. Overwriting.")
                old_map = {}
    else:
        print(f"💾 Creating new map file '{OUTPUT_JSON}'...")
        old_map = {}

    # Update the old map with the new, complete map.
    # This adds missing keys and overwrites existing ones
    # with the latest data from Cloudflare.
    old_map.update(rebuilt_map)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(old_map, f, indent=2, ensure_ascii=False)

    print(f"🎉 Success! '{OUTPUT_JSON}' is now complete with {len(old_map)} entries.")


if __name__ == "__main__":
    fetch_and_rebuild_map()