#!/usr/bin/env python3
import os, sys, time, argparse, math
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import Birch
import pandas as pd
import csv
from transformers import BitsAndBytesConfig
from qwen3_vl_embedding import Qwen3VLEmbedder

# --- CONFIG ---
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
SKIP_EXTS = {".zip", ".gz", ".tar", ".exe", ".so", ".dll", ".class", ".pdf", ".swf", ".jar"}

def get_mem_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)  # Returns GB
    except ImportError:
        return 0.0

def log_heartbeat(msg: str):
    with open("heartbeat.log", "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")

def is_image(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk
    except Exception:
        return True

def list_files_recursive(root: str) -> list:
    out = []
    log_heartbeat(f"Scanning directory: {root}")
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            
            # 1. Image Check
            if is_image(p):
                out.append(p)
                continue
            
            # 2. Blocklist Check
            ext = os.path.splitext(p)[1].lower()
            if ext in SKIP_EXTS:
                continue

            # 3. Binary Content Check (Safety Fallback)
            if not is_binary(p):
                out.append(p)
    return sorted(out)

def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception: return None

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(25000).strip() # Truncate to avoid excessive memory on huge files
    except Exception: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-Embedding-8B")
    ap.add_argument("--cache", default="govdocs_vectors.csv")
    ap.add_argument("--limit", type=int, default=None, help="Stop after this many files")
    args = ap.parse_args()

    # --- 1. EMBEDDING PHASE ---
    processed_files = set()
    X_list = []
    
    # Load existing data if available
    if os.path.exists(args.cache) and os.path.getsize(args.cache) > 0:
        print(f"Loading existing cache: {args.cache}")
        try:
            df = pd.read_csv(args.cache)
            processed_files = set(df.iloc[:, 0].tolist())
            print(f"Loaded {len(processed_files)} previously processed files.")
        except Exception as e:
            print(f"Warning: Could not read cache {args.cache}: {e}. Starting fresh.")
    
    # Discovery
    files = list_files_recursive(args.root)
    files_to_process = [f for f in files if os.path.basename(f) not in processed_files]
    
    if args.limit:
        remaining_slots = args.limit - len(processed_files)
        if remaining_slots <= 0:
             print(f"Limit of {args.limit} reached with existing {len(processed_files)} files.")
             files_to_process = []
        else:
             print(f"Limit applied: Processing {remaining_slots} more files.")
             files_to_process = files_to_process[:remaining_slots]

    print(f"Found {len(files)} total files. Processing {len(files_to_process)} new files. Model: {args.repo}")
    
    if files_to_process:
        # Initialize Qwen3-VL Embedder
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen3VLEmbedder(
            model_name_or_path=args.repo,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Open in APPEND mode
        mode = "a" if (os.path.exists(args.cache) and os.path.getsize(args.cache) > 0) else "w"
        with open(args.cache, mode, newline="") as f_out:
            writer = csv.writer(f_out)
            
            # Write header only if new file
            header_written = mode == "a"
            
            import gc
            
            for i, p in enumerate(files_to_process):
                filename = os.path.basename(p)
                log_heartbeat(f"Starting {filename} | Mem: {get_mem_usage():.2f}GB")
                
                # Explicit GC every 50 files
                if i > 0 and i % 50 == 0:
                    gc.collect()

                start_time = time.perf_counter()
                vec = None
                
                try:
                    inputs = []
                    if is_image(p):
                        img = load_image(p)
                        if img:
                            inputs = [{'image': img}]
                    else:
                        txt = read_text(p)
                        if txt:
                            inputs = [{'text': txt}]
                    
                    if inputs:
                        # process returns a tensor of shape (1, hidden_size)
                        res = model.process(inputs)
                        vec = res.float().cpu().numpy()[0]

                except Exception as e:
                    print(f"ERROR processing {filename}: {e}")
                    log_heartbeat(f"ERROR on {filename}: {e}")
                    continue

                if vec is not None:
                    if not header_written:
                        writer.writerow(["filename"] + [f"d{d}" for d in range(len(vec))])
                        header_written = True
                    writer.writerow([filename] + vec.tolist())
                    f_out.flush()
                
                elapsed = time.perf_counter() - start_time
                print(f"[{i+1}/{len(files_to_process)}] Processed {filename} in {elapsed:.2f}s | Mem: {get_mem_usage():.2f}GB")
    
    # Re-load for clustering
    if os.path.exists(args.cache) and os.path.getsize(args.cache) > 0:
        df = pd.read_csv(args.cache)
        X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        print("No data found.")
        return

    # --- 2. CLUSTERING & PLOTTING ---
    if len(X) < 2:
        print("Not enough vectors to cluster.")
        return

    print(f"Running t-SNE and BIRCH on {len(X)} vectors...")
    # Adjust perplexity for small datasets
    perplexity = min(30, len(X)-1)
    if perplexity < 1: perplexity = 1
    
    Z = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(X)
    y_birch = Birch(n_clusters=None, threshold=0.7).fit_predict(X)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(Z[:, 0], Z[:, 1], c=y_birch, cmap='tab20', s=40, alpha=0.7)
    plt.title(f"BIRCH Clusters (found {len(np.unique(y_birch))} clusters)")
    plt.colorbar(label='Cluster ID')
    
    plot_path = "final_birch_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Done! Plot saved to {plot_path}")

    # --- 3. CLUSTER ANALYSIS OUTPUT ---
    print("Generating cluster analysis report...")
    report_path = "cluster_analysis.txt"
    
    # Create a DataFrame for analysis
    # df is already loaded above with 'filename' as column 0
    filenames = df.iloc[:, 0].tolist()
    
    analysis_df = pd.DataFrame({
        'filename': filenames,
        'cluster_id': y_birch
    })
    
    # Sort by cluster ID then filename
    analysis_df = analysis_df.sort_values(['cluster_id', 'filename'])
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BIRCH CLUSTER ANALYSIS REPORT\n")
        f.write("="*40 + "\n\n")
        
        # Statistics
        n_clusters = len(np.unique(y_birch))
        f.write(f"Total Files: {len(filenames)}\n")
        f.write(f"Total Clusters: {n_clusters}\n")
        
        counts = analysis_df['cluster_id'].value_counts()
        f.write(f"Largest Cluster: ID {counts.idxmax()} ({counts.max()} files)\n")
        f.write(f"Smallest Cluster: ID {counts.idxmin()} ({counts.min()} files)\n")
        f.write(f"Average Files per Cluster: {counts.mean():.2f}\n\n")
        
        f.write("="*40 + "\n\n")
        f.write("DETAILED CLUSTER LISTINGS\n\n")
        
        # Group listing
        for cluster_id, group in analysis_df.groupby('cluster_id'):
            files_in_cluster = group['filename'].tolist()
            f.write(f"Cluster {cluster_id} ({len(files_in_cluster)} files):\n")
            for fn in files_in_cluster:
                f.write(f"  - {fn}\n")
            f.write("\n")
            
    print(f"Analysis saved to {report_path}")

if __name__ == "__main__":
    main()
