import torch
from ops_mm_embedding_v1 import OpsMMEmbeddingV1
import os
import psutil

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

print(f"Start Mem: {get_mem():.2f} GB")
model = OpsMMEmbeddingV1("OpenSearch-AI/Ops-MM-embedding-v1-2B", device="cpu", attn_implementation="sdpa")
print(f"Model Loaded. Mem: {get_mem():.2f} GB")

file_path = "tests/govdocs_test_20/000/000/000478.txt"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    txt = f.read(25000).strip() # Mimic the production script limit

print(f"Read {len(txt)} chars.")
print("Running inference...")
try:
    res = model.get_text_embeddings([txt])
    print("Inference success!")
    print(res.shape)
except Exception as e:
    print(f"Caught exception: {e}")

print(f"End Mem: {get_mem():.2f} GB")
