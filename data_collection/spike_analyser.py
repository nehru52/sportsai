import os
import json
import numpy as np

SPIKE_DIR = "C:/sportsai-backend/data/pose_data/volleyball/spike"
METADATA_PATH = "C:/sportsai-backend/data/pose_data/volleyball/metadata.json"

METRICS = ["arm_cock_angle", "jump_height", "approach_speed", "contact_point", "follow_through"]

# Higher is better for these
HIGHER_IS_BETTER = {"arm_cock_angle", "jump_height", "approach_speed", "follow_through"}
# Lower is better for these
LOWER_IS_BETTER = {"contact_point"}

def load_biomechanics_by_level():
    metadata = json.load(open(METADATA_PATH))
    data = {"elite": [], "advanced": [], "intermediate": []}

    for record in metadata["processed"]:
        level = record["skill_level"]
        bio_file = os.path.join(SPIKE_DIR, record["biomechanics_file"])
        if not os.path.exists(bio_file):
            continue
        bio = json.load(open(bio_file))
        if level in data:
            data[level].append(bio)

    return data

def compute_thresholds(data: dict) -> dict:
    thresholds = {}
    for metric in METRICS:
        elite_vals = [d[metric] for d in data["elite"] if metric in d]
        inter_vals = [d[metric] for d in data["intermediate"] if metric in d]
        if elite_vals and inter_vals:
            thresholds[metric] = {
                "elite_mean": round(float(np.mean(elite_vals)), 3),
                "elite_std": round(float(np.std(elite_vals)), 3),
                "intermediate_mean": round(float(np.mean(inter_vals)), 3),
            }
    return thresholds

def analyse_biomechanics(bio: dict, thresholds: dict) -> dict:
    report = {}
    for metric in METRICS:
        if metric not in bio or metric not in thresholds:
            continue
        val = bio[metric]
        t = thresholds[metric]
        elite_mean = t["elite_mean"]
        elite_std = t["elite_std"]

        if metric in HIGHER_IS_BETTER:
            good = val >= (elite_mean - elite_std)
        else:
            good = val <= (elite_mean + elite_std)

        report[metric] = {
            "value": round(val, 3),
            "elite_mean": elite_mean,
            "status": "GOOD" if good else "NEEDS IMPROVEMENT",
        }
    return report

def print_report(report: dict):
    print("\n=== SPIKE ANALYSIS REPORT ===")
    for metric, result in report.items():
        status = result["status"]
        val = result["value"]
        elite = result["elite_mean"]
        indicator = "✓" if status == "GOOD" else "✗"
        print(f"  {indicator} {metric:<20} yours: {val:<10} elite avg: {elite:<10} [{status}]")
    good = sum(1 for r in report.values() if r["status"] == "GOOD")
    total = len(report)
    print(f"\nOverall: {good}/{total} metrics at elite level")
    if good == total:
        print("Verdict: ELITE form")
    elif good >= total * 0.6:
        print("Verdict: GOOD form, minor improvements needed")
    else:
        print("Verdict: NEEDS WORK")

if __name__ == "__main__":
    print("Loading biomechanics data...")
    data = load_biomechanics_by_level()
    print(f"Elite: {len(data['elite'])} | Advanced: {len(data['advanced'])} | Intermediate: {len(data['intermediate'])}")

    thresholds = compute_thresholds(data)
    print("\nElite thresholds computed:")
    for m, t in thresholds.items():
        print(f"  {m}: elite_mean={t['elite_mean']}, intermediate_mean={t['intermediate_mean']}")

    # Test against a sample from your own data
    sample_files = [f for f in os.listdir(SPIKE_DIR) if f.endswith("_biomechanics.json")]
    sample_bio = json.load(open(os.path.join(SPIKE_DIR, sample_files[0])))
    report = analyse_biomechanics(sample_bio, thresholds)
    print_report(report)
