# file: load_cdn_update_all.py
import os, json, sys, time
import psycopg2
from psycopg2.extras import execute_values

# ---- CONFIG ----
DB_URL = "REDACTED"
JSON_PATH = "uploaded_map.json"

# If your DB stores backslashes in filepath, set this to False
NORMALIZE_TO_FORWARD_SLASH = True

# Insert batch size for temp-table load (tune if needed)
PAGE_SIZE = 1000

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def norm(p: str) -> str:
    return p.replace("\\", "/") if NORMALIZE_TO_FORWARD_SLASH else p

def main():
    log("Loading JSON...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = [(norm(k), v) for k, v in raw.items()]
    log(f"Loaded {len(pairs)} mappings.")

    log("Connecting to Postgres...")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # Fail reasonably fast on unexpected locks during setup steps
            log("Setting session timeouts...")
            cur.execute("SET lock_timeout = '5s';")
            cur.execute("SET statement_timeout = '2min';")  # short for small steps

            # Confirm column exists (no ALTER to avoid DDL lock)
            log("Verifying cdn_url column exists...")
            cur.execute("""
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema='public'
                  AND table_name='art_embeddings'
                  AND column_name='cdn_url';
            """)
            if cur.fetchone() is None:
                raise RuntimeError("Column cdn_url does not exist. Add it first and rerun.")

            # Temp table
            log("Creating temp table...")
            cur.execute("CREATE TEMP TABLE tmp_cdn_map (filepath text, cdn_url text) ON COMMIT DROP;")

            log("Bulk inserting into temp table...")
            execute_values(
                cur,
                "INSERT INTO tmp_cdn_map (filepath, cdn_url) VALUES %s",
                pairs,
                page_size=PAGE_SIZE
            )

            # Diagnostics
            log("Counting rows...")
            cur.execute("SELECT COUNT(*) FROM public.art_embeddings;")
            total_rows = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM tmp_cdn_map;")
            tmp_rows = cur.fetchone()[0]
            log(f"art_embeddings rows={total_rows}, tmp_cdn_map rows={tmp_rows}")

            # How many will actually change?
            log("Estimating rows to update...")
            cur.execute("""
                SELECT COUNT(*)
                FROM public.art_embeddings a
                JOIN tmp_cdn_map t ON t.filepath = a.filepath
                WHERE a.cdn_url IS DISTINCT FROM t.cdn_url;
            """)
            to_change = cur.fetchone()[0]
            log(f"Rows that need update: {to_change}")

            # Lift timeout for the big update so it can run to completion
            log("Removing statement timeout for the big update...")
            cur.execute("SET statement_timeout = '0';")  # no timeout

            # One big update (will take minutes if many rows + large indexes)
            log("Running UPDATE ... FROM (this may take several minutes)...")
            cur.execute("""
                UPDATE public.art_embeddings a
                SET cdn_url = t.cdn_url
                FROM tmp_cdn_map t
                WHERE a.filepath = t.filepath
                  AND a.cdn_url IS DISTINCT FROM t.cdn_url;
            """)

            # Post-check
            log("Counting populated rows...")
            cur.execute("SELECT COUNT(*) FROM public.art_embeddings WHERE cdn_url IS NOT NULL;")
            populated = cur.fetchone()[0]
            log(f"Rows with cdn_url populated: {populated}")

        conn.commit()
        log("✅ Update completed.")
    except Exception as e:
        conn.rollback()
        print("ERROR:", e, file=sys.stderr)
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
