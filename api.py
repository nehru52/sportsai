from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, tempfile, json, sys

sys.path.insert(0, "C:/sportsai-backend/data_collection")
from pose_extractor import extract_pose
from spike_analyser import load_biomechanics_by_level, compute_thresholds, analyse_biomechanics

app = FastAPI(title="SportsAI Volleyball Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load thresholds once at startup
data = load_biomechanics_by_level()
thresholds = compute_thresholds(data)

@app.get("/")
def root():
    return {"status": "ok", "message": "SportsAI Volleyball Analysis API"}

@app.post("/analyse/spike")
async def analyse_spike(video: UploadFile = File(...)):
    # Save uploaded video to temp file
    suffix = os.path.splitext(video.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        result = extract_pose(tmp_path, "spike")
        report = analyse_biomechanics(result["biomechanics"], thresholds)

        good = sum(1 for r in report.values() if r["status"] == "GOOD")
        total = len(report)

        if good == total:
            verdict = "ELITE"
        elif good >= total * 0.6:
            verdict = "GOOD"
        else:
            verdict = "NEEDS WORK"

        return {
            "verdict": verdict,
            "score": f"{good}/{total}",
            "metrics": report,
            "frames_analysed": len(result["pose_sequence_3d"]),
            "confidence": result["average_confidence"]
        }
    finally:
        os.remove(tmp_path)
