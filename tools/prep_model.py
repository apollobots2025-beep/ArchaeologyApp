# tools/prep_model.py
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# Download model weights
model = AutoModelForImageClassification.from_pretrained(args.model, cache_dir=args.out)
# Download feature extractor (preprocessor)
extractor = AutoFeatureExtractor.from_pretrained(args.model, cache_dir=args.out)

print(f"Model downloaded to {args.out}")
