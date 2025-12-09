#!/usr/bin/env python3
"""
Download a Hugging Face model to a local folder so it can be bundled into the EXE.
Usage:
  python tools/prep_model.py --model microsoft/resnet-50 --out hf_models/resnet50
"""
import argparse
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification

def download_model(model_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model {model_id} to {out_dir} ...")
    model = AutoModelForImageClassification.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    download_model(args.model, Path(args.out))
