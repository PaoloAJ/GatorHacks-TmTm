import json
import psycopg2
from psycopg2.extras import execute_batch

DB_URL = "REDACTED"
JSON_FILE = "art_embeddings_improved.json"
BATCH_SIZE = 200

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

def stream_json_array(filepath):
    """Robustly stream large JSON array: [ {...}, {...}, ... ]"""
    decoder = json.JSONDecoder()
    buf = ""
    with open(filepath, "r", encoding="utf-8") as f:
        # Skip any whitespace before '['
        for line in f:
            if "[" in line:
                buf = line[line.find("[") + 1:]
                break

        # Stream parse each object
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            buf += chunk

            while True:
                buf = buf.lstrip(", \n\r\t")
                if not buf:
                    break
                if buf[0] == "]":
                    return  # End of array

                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    # Need more data
                    break

                yield obj
                buf = buf[idx:]

batch = []
count = 0
for item in stream_json_array(JSON_FILE):
    batch.append((
        item.get("artist"),
        item.get("title"),
        item.get("year"),
        item.get("style"),
        item.get("filepath"),
        item.get("image_embedding"),
        item.get("text_embedding")
    ))

    if len(batch) >= BATCH_SIZE:
        execute_batch(cur, """
            INSERT INTO art_embeddings
            (artist, title, year, style, filepath, image_embedding, text_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filepath) DO NOTHING;
        """, batch)
        conn.commit()
        count += len(batch)
        print(f"✅ Inserted {count} rows...")
        batch.clear()

# Final flush
if batch:
    execute_batch(cur, """
        INSERT INTO art_embeddings
        (artist, title, year, style, filepath, image_embedding, text_embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (filepath) DO NOTHING;
    """, batch)
    conn.commit()
    count += len(batch)

cur.close()
conn.close()
print(f"🎉 Done! Inserted total {count} rows.")
