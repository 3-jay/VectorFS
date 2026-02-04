#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from PIL import Image
from qwen3_vl_embedding import Qwen3VLEmbedder
from transformers import BitsAndBytesConfig


# --- CONFIG ---
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

def get_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def get_file_embedding(model, file_path):
    """Generates an embedding for a file based on its type."""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in IMG_EXTS:
            img = Image.open(file_path).convert("RGB")
            inputs = [{'image': img}]
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                # Using your character limit logic for stability
                content = f.read(25000).strip()
            inputs = [{'text': content}]
        
        # Generate and return the vector as a Float32 numpy array
        res = model.process(inputs)
        return res.float().cpu().numpy()[0]
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <file_path_1> <file_path_2>")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]


    print(f"Loading Qwen3-VL-Embedding-8B with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen3VLEmbedder(
        model_name_or_path="Qwen/Qwen3-VL-Embedding-8B",
        quantization_config=bnb_config,
        device_map="auto"
    )

    print(f"Embedding File 1: {os.path.basename(path1)}")
    vec1 = get_file_embedding(model, path1)
    
    print(f"Embedding File 2: {os.path.basename(path2)}")
    vec2 = get_file_embedding(model, path2)

    if vec1 is not None and vec2 is not None:
        score = get_cosine_similarity(vec1, vec2)
        print("\n" + "="*30)
        print(f"SIMILARITY SCORE: {score:.4f}")
        print("="*30)
        
        # Interpretation Guide
        if score > 0.85:
            print("Interpretation: Near Identical / Strong Semantic Match")
        elif score > 0.60:
            print("Interpretation: Highly Related")
        elif score > 0.30:
            print("Interpretation: Vaguely Related")
        else:
            print("Interpretation: Unrelated")
    else:
        print("Failed to generate one or both embeddings.")

if __name__ == "__main__":
    main()