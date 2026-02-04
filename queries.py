import torch
from transformers import BitsAndBytesConfig
from qwen3_vl_embedding import Qwen3VLEmbedder

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Initialize directly with 4-bit config to match your environment
model = Qwen3VLEmbedder(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Dummy input
inputs = [{'text': "Hello world"}]
print("Generating embedding...")
vec = model.process(inputs)

print("\n" + "="*30)
print(f"Vector Shape: {vec.shape}")
print(f"Vector Dimension: {vec.shape[1]}")
print("="*30)