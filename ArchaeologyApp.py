import json, urllib.request, base64, os, time, datetime

API_TOKEN = "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"      # HuggingFace API token
WEB_URL   = "https://script.google.com/macros/s/AKfycbwnGsokM6o1MGcHdA4dpIVNAXgeAycBxE6uw2qngbWNN8gxhPZuVDcc0K7f1KMCqmyZ2w/exec"      # Google Apps Script URL

CLIP_MODEL  = "openai/clip-vit-large-patch14"
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# -----------------------------
# 1. IMAGE → CLIP IDENTIFICATION
# -----------------------------

def identify_clip(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()

    payload = {
        "model": CLIP_MODEL,
        "inputs": {
            "image": img_b64
        }
    }

    req = urllib.request.Request(
        "https://router.huggingface.co/inference",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read().decode())

    labels = data["outputs"][0]["labels"]
    scores = data["outputs"][0]["scores"]
    embedding = data["outputs"][0]["embedding"]

    return labels, scores, embedding


# -----------------------------
# 2. LLAMA → DEEP ANALYSIS
# -----------------------------

def analyze_llama(labels, scores):
    pairs = [f"{labels[i]} ({scores[i]:.2f})" for i in range(len(labels))]

    prompt = f"""
You are a world-class archaeologist.

These are the image classifier's top identifications:
{pairs}

Write an EXTREMELY detailed archaeological analysis including:

- What the object most likely is
- Probable cultural origin
- Estimated time period
- Materials and crafting techniques
- Decoration meaning (symbols, engravings, motifs)
- Historical context
- Comparison to similar real artifacts
- Preservation state and risks
- How professionals would examine/determine authenticity
- Scientific methods (XRF, carbon dating, CT scanning)
- Excavation context
- Final probability and confidence ranking

Write **12 to 18 paragraphs**, extremely detailed.
**However, do remember that you can only put 50,000 characters into your report. Make sure that you put 50,000 characters or less.**
"""

    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "temperature": 0.7
    }

    req = urllib.request.Request(
        "https://router.huggingface.co/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as r:
        resp = json.loads(r.read().decode())

    return resp["choices"][0]["message"]["content"]


# -----------------------------
# 3. SEND TO GOOGLE SHEETS
# -----------------------------

def send_to_sheets(final_data):
    req = urllib.request.Request(
        WEB_URL,
        data=json.dumps(final_data).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        print("Sheets:", r.read().decode())


# -----------------------------
# 4. PROCESS MULTIPLE IMAGES
# -----------------------------

def process_folder(folder_path, gps=None):
    results = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]

    for file in files:
        full_path = os.path.join(folder_path, file)
        print(f"\n🔍 Processing {file}...")

        # 1. CLIP
        labels, scores, embedding = identify_clip(full_path)

        # 2. LLaMA analysis
        analysis = analyze_llama(labels[:5], scores[:5])

        # 3. Collect data
        result_obj = {
            "file": file,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "gps": gps,
            "clip_labels": labels[:5],
            "clip_scores": [float(s) for s in scores[:5]],
            "clip_embedding": embedding,
            "analysis": analysis
        }

        results.append(result_obj)

    # Send everything to Sheets in one payload
    send_to_sheets({"batch": results})

    return results


# -----------------------------
# 5. RUN
# -----------------------------

# Example:
# gps = {"lat": 34.0219, "lon": -118.4814}  # optional
gps = None

final = process_folder("artifacts", gps=gps)

print("\nDONE.\n")