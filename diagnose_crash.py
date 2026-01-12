import torch
from ops_mm_embedding_v1 import OpsMMEmbeddingV1
import os
import psutil
import faulthandler
import sys

# Enable fault handler to dump stack trace on segfault
faulthandler.enable()

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

print(f"Start Mem: {get_mem():.2f} GB")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

# Load model with explicit max_length to force truncation to safe limits
model = OpsMMEmbeddingV1(
    "OpenSearch-AI/Ops-MM-embedding-v1-2B", 
    device="cpu", 
    attn_implementation="sdpa",
    max_length=8192  # Force token-level truncation
)
print(f"Model Loaded. Mem: {get_mem():.2f} GB")

file_path = "tests/govdocs_test_20/000/000/000478.txt"
print(f"Testing file: {file_path}")

try:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read(25000).strip()
    
    print(f"Read {len(txt)} chars.")
    print("Preview of text start:", txt[:100])
    print("Preview of text end:", txt[-100:])
    
    print("Running tokenization only check...")
    # Isolate tokenization from inference to pinpoint crash
    tokens = model.processor.tokenizer(txt, return_tensors="pt")
    print(f"Tokenization successful. Shape: {tokens.input_ids.shape}")
    
    print("Running full inference...")
    res = model.get_text_embeddings([txt])
    print("Inference success!")
    print(res.shape)

except Exception as e:
    print(f"Caught Python exception: {e}")

print(f"End Mem: {get_mem():.2f} GB")
