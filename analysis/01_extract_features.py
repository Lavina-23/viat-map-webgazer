import os, json, glob
import numpy as np
import pandas as pd

RAW_DIR = "../data/raw_logs"
OUTPUT_FILE = "../data/processed/features.csv"
os.makedirs("../data/processed", exist_ok=True)

rows = []
for filepath in glob.glob(os.path.join(RAW_DIR, "*.json")):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    p_id = data["participant_id"]
    for trial in data["trials"]:
        if trial.get("probe_response") is None:
            continue

        target_sentence = trial["sentences"][-1]
        gaze_pts = target_sentence["gaze_points"]
        reading_time = target_sentence["reading_time_sec"]

        if len(gaze_pts) < 5:
            continue

        xs = np.array([p["x"] for p in gaze_pts])
        ys = np.array([p["y"] for p in gaze_pts])
        aois = [p["aoi"] for p in gaze_pts]

        num_samples = len(gaze_pts)
        num_unique = len(set(zip(np.round(xs / 20), np.round(ys / 20))))
        mx, my = np.mean(xs), np.mean(ys)
        dispersion = float(np.sqrt(np.mean((xs - mx)**2 + (ys - my)**2)))

        opt1 = sum(1 for a in aois if a == "opt1")
        opt2 = sum(1 for a in aois if a == "opt2")
        opt3 = sum(1 for a in aois if a == "opt3")
        all_opts = opt1 + opt2 + opt3
        sentence = sum(1 for a in aois if a == "sentence_center")

        rows.append({
            "participant_id": p_id,
            "story_id": trial["story_id"],
            "reading_time": reading_time,
            "num_samples": num_samples,
            "num_unique": num_unique,
            "dispersion": dispersion,
            "opt1": opt1, "opt2": opt2, "opt3": opt3,
            "all_opts": all_opts, "sentence": sentence,
            "label_tut": trial["probe_response"],
            "label_correct": trial.get("is_correct", 1)
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Sukses! {len(df)} data berhasil diekstrak ke {OUTPUT_FILE}")