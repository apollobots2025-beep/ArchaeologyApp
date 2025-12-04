import os
import datetime
import pandas as pd
from transformers import pipeline
from PIL import Image
import torch

# -----------------------------
# CONFIG
# -----------------------------
CLS_MODEL   = "microsoft/resnet-50"
DEVICE = 0 if torch.cuda.is_available() else -1
EXCEL_FILE = "artifacts_results.xlsx"

# -----------------------------
# PIPELINES
# -----------------------------
cls_pipe = pipeline("image-classification", model=CLS_MODEL, device=DEVICE)

# -----------------------------
# CLASSIFY
# -----------------------------
def classify_image(image):
    results = cls_pipe(image)
    labels = [item["label"] for item in results]
    scores = [item["score"] for item in results]
    return labels, scores

# -----------------------------
# ANALYSIS (FAST)
# -----------------------------
def quick_analysis(labels, scores):
    # Just return a concise string instead of calling a heavy model
    pairs = [f"{labels[i]} ({scores[i]:.2f})" for i in range(min(len(labels), len(scores)))]
    return "Top identifications: " + "; ".join(pairs)

# -----------------------------
# SAVE TO EXCEL
# -----------------------------
def save_to_excel(results):
    df = pd.DataFrame(results)

    if os.path.exists(EXCEL_FILE):
        old = pd.read_excel(EXCEL_FILE)
        df = pd.concat([old, df], ignore_index=True)

    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False)

    print(f"📊 Results saved to {EXCEL_FILE}")

# -----------------------------
# PROCESS FILE
# -----------------------------
def process_file(file, folder_path, gps=None):
    full_path = os.path.join(folder_path, file)
    print(f"🔍 Processing {file}...")

    try:
        image = Image.open(full_path).convert("RGB")
        labels, scores = classify_image(image)
        analysis = quick_analysis(labels[:3], scores[:3])

        gps_lat, gps_lon = (None, None)
        if gps and isinstance(gps, (tuple, list)) and len(gps) == 2:
            gps_lat, gps_lon = gps

        result_obj = {
            "file": file,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "labels": ", ".join(labels[:3]),
            "scores": ", ".join([f"{s:.2f}" for s in scores[:3]]),
            "analysis": analysis
        }

        print(f"✅ Done: {file}")
        return result_obj

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
        return None

# -----------------------------
# PROCESS FOLDER
# -----------------------------
def process_folder(folder_path, gps=None):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("No images found.")
        return []

    results = []
    for f in files:
        res = process_file(f, folder_path, gps)
        if res:
            results.append(res)

    if results:
        save_to_excel(results)

    return results

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    gps = (41.75, -111.83)  # Example GPS
    final = process_folder("artifacts", gps=gps)
    print("\nDONE.\n")
