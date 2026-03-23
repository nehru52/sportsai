import csv
import os

VALID_TECHNIQUES = {"spike", "serve", "block", "dig"}
VALID_SKILL_LEVELS = {"beginner", "intermediate", "advanced", "elite"}

def load_urls(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    valid_rows = []
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"url", "technique", "skill_level", "source_channel"}
        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(f"CSV missing columns. Required: {required_cols}")

        for i, row in enumerate(reader, start=1):
            technique = row["technique"].strip().lower()
            skill_level = row["skill_level"].strip().lower()

            if technique not in VALID_TECHNIQUES:
                print(f"[SKIP] Row {i}: invalid technique '{technique}'")
                skipped += 1
                continue

            if skill_level not in VALID_SKILL_LEVELS:
                print(f"[SKIP] Row {i}: invalid skill_level '{skill_level}'")
                skipped += 1
                continue

            if not row["url"].strip().startswith("http"):
                print(f"[SKIP] Row {i}: invalid URL")
                skipped += 1
                continue

            valid_rows.append({
                "url": row["url"].strip(),
                "technique": technique,
                "skill_level": skill_level,
                "source_channel": row["source_channel"].strip()
            })

    print(f"[url_loader] Loaded {len(valid_rows)} valid rows, skipped {skipped}")
    return valid_rows


