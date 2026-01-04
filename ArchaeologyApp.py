import os
import sys
import datetime
import pandas as pd
from transformers import pipeline
from PIL import Image
import torch

# -----------------------------
# CONFIG
# -----------------------------
# Fully open model (no gated HF access required)
CLS_MODEL_HF = "google/vit-base-patch16-224"
BUNDLED_MODEL_DIR = os.path.join("hf_models", "vit-base")
DEVICE = 0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else -1
EXCEL_FILE = "artifacts_results.xlsx"

def get_model_source():
    """
    Returns the model path for pipeline:
    - Use PyInstaller bundled directory if available
    - Use local folder if present
    - Otherwise fall back to HF ID (open model)
    """
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base:
            bundled = os.path.join(base, BUNDLED_MODEL_DIR)
            if os.path.exists(bundled):
                return bundled

    if os.path.exists(BUNDLED_MODEL_DIR):
        return BUNDLED_MODEL_DIR

    return CLS_MODEL_HF

# -----------------------------
# PIPELINES
# -----------------------------
MODEL_SOURCE = get_model_source()
cls_pipe = pipeline("image-classification", model=MODEL_SOURCE, device=DEVICE)

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

    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="w", if_sheet_exists="replace") as writer:
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
