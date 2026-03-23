import subprocess
import json
import os

RAW_VIDEO_DIR = "C:/sportsai-backend/data/raw_videos"

def download_video(url: str, output_filename: str) -> dict:
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    output_path = os.path.join(RAW_VIDEO_DIR, f"{output_filename}.%(ext)s")

    cmd = ["yt-dlp",
           "--cookies", "C:/sportsai-backend/cookies.txt",
           "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best[height<=480]",
           "--merge-output-format", "mp4",
           "--no-playlist",
           "-o", output_path,
           url]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {url}:\n{result.stderr}")

    file_path = None
    for f in os.listdir(RAW_VIDEO_DIR):
        if f.startswith(output_filename) and f.endswith(".mp4"):
            file_path = os.path.join(RAW_VIDEO_DIR, f)
            break

    if not file_path:
        raise FileNotFoundError(f"Downloaded file not found for {output_filename}")

    probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-show_format", file_path]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    meta = json.loads(probe.stdout)

    duration = float(meta["format"].get("duration", 0))
    resolution = "unknown"
    for stream in meta.get("streams", []):
        if stream.get("codec_type") == "video":
            resolution = f"{stream['width']}x{stream['height']}"
            break

    return {"file_path": file_path, "duration": round(duration, 2), "resolution": resolution}
