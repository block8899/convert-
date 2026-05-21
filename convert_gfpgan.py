import torch
import pnnx
import os
import sys
import gc

print("1. Loading GFPGAN model...")

# GFPGAN architecture
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

model = GFPGANv1Clean(
    out_size=512,
    num_style_feat=512,
    channel_multiplier=2,
    decoder_load_path=None,
    fix_decoder=False,
    num_mlp=8,
    input_is_latent=False,
    different_w=True,
    narrow=1,
    sft_half=True
)

ckpt = torch.load("GFPGANv1.4.pth", map_location="cpu", weights_only=False)
if 'params_ema' in ckpt:
    model.load_state_dict(ckpt['params_ema'], strict=False)
elif 'params' in ckpt:
    model.load_state_dict(ckpt['params'], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)

model.eval()
print("Model loaded OK")

# Disable gradient
torch.set_grad_enabled(False)

print("2. Converting to NCNN via PNNX...")

# GFPGAN input: [1, 3, 512, 512] (cropped face)
dummy = torch.randn(1, 3, 512, 512)

try:
    pnnx.export(model, "gfpgan", inputs=dummy)
    print("PNNX export done!")
except Exception as e:
    print(f"PNNX failed: {e}")
    sys.exit(1)

del model, dummy
gc.collect()

print("3. Verifying...")
param_file = "gfpgan.ncnn.param"
bin_file = "gfpgan.ncnn.bin"

if os.path.exists(param_file) and os.path.exists(bin_file):
    size_param = os.path.getsize(param_file) / 1024
    size_bin = os.path.getsize(bin_file) / 1024 / 1024
    print(f"  {param_file}: {size_param:.1f} KB")
    print(f"  {bin_file}: {size_bin:.1f} MB")
    with open(param_file, "r") as f:
        lines = f.readlines()
        if len(lines) >= 2:
            print(f"  Layers: {lines[1].strip()}")
        # Find input/output names
        for line in lines:
            if line.strip().startswith("Input"):
                print(f"  Input layer: {line.strip()}")
            if "out" in line.lower() and line.strip().startswith("Convolution"):
                pass
    # Get last blob name for output
    last_line = lines[-1].strip()
    parts = last_line.split()
    if len(parts) >= 4:
        output_blob = parts[-1] if len(parts) >= 2 else "unknown"
        print(f"  Last layer output: {output_blob}")
    print("GFPGAN NCNN OK!")
else:
    print("FAILED! Files not found.")
    print(f"Current files: {os.listdir('.')}")
    sys.exit(1)
