import os
import json
import numpy as np
from datetime import datetime

from url_loader import load_urls
from youtube_downloader import download_video
from pose_extractor import extract_pose

CSV_PATH = "C:/sportsai-backend/data/input/volleyball_urls.csv"
POSE_OUTPUT_DIR = "C:/sportsai-backend/data/pose_data/volleyball"
METADATA_PATH = "C:/sportsai-backend/data/pose_data/volleyball/metadata.json"

def load_metadata() -> dict:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {"processed": [], "failed": [], "last_updated": None}

def save_metadata(meta: dict):
    meta["last_updated"] = datetime.now().isoformat()
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def process_batch():
    rows = load_urls(CSV_PATH)
    total = len(rows)
    metadata = load_metadata()
    processed_urls = {r["url"] for r in metadata["processed"]}

    for i, row in enumerate(rows, start=1):
        url = row["url"]
        technique = row["technique"]
        skill_level = row["skill_level"]
        channel = row["source_channel"]

        if url in processed_urls:
            print(f"[SKIP] {i}/{total}: already processed - {url}")
            continue

        safe_name = f"vb_{technique}_{skill_level}_{i}"
        print(f"[START] Processing {i}/{total}: volleyball {technique} [{skill_level}] from {channel}")

        try:
            dl = download_video(url, safe_name)
            file_path = dl["file_path"]
            print(f"  Downloaded: {dl['resolution']}, {dl['duration']}s")

            result = extract_pose(file_path, technique)
            frames = len(result["pose_sequence_3d"])
            conf = result["average_confidence"]

            print(f"  Processing {i}/{total}: volleyball {technique} [{skill_level}] - {frames} frames, {conf:.2f} confidence")

            out_dir = os.path.join(POSE_OUTPUT_DIR, technique)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"{safe_name}_pose3d.npy"), result["pose_sequence_3d"])
            with open(os.path.join(out_dir, f"{safe_name}_biomechanics.json"), "w") as f:
                json.dump(result["biomechanics"], f, indent=2)

            metadata["processed"].append({
                "url": url, "technique": technique, "skill_level": skill_level,
                "source_channel": channel, "frames": frames, "confidence": conf,
                "pose_file": f"{safe_name}_pose3d.npy",
                "biomechanics_file": f"{safe_name}_biomechanics.json",
                "timestamp": datetime.now().isoformat()
            })

            os.remove(file_path)
            print(f"  [DONE] Saved pose data, deleted raw video")

        except Exception as e:
            print(f"  [ERROR] {i}/{total}: {e}")
            metadata["failed"].append({"url": url, "error": str(e), "timestamp": datetime.now().isoformat()})

        save_metadata(metadata)

    print(f"\n[COMPLETE] {len(metadata['processed'])} processed, {len(metadata['failed'])} failed")

if __name__ == "__main__":
    process_batch()


