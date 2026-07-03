"""
generate_mem_files.py
=====================
Generates all .mem files required by host.cpp for the Mamba-790M FPGA
accelerator, starting from a fresh download of the pretrained model.

Pipeline:
  1. Download Mamba-790M from HuggingFace
  2. Extract and INT8-quantize block 0 weights (per-tensor symmetric)
  3. Reload dequantized weights so the GPU forward pass mirrors FPGA precision
  4. Load WikiText-103 test split, take the first 2048 tokens
  5. Run inference, hook block 0 to capture input/output activations
  6. Save all weights, biases, A matrix, D vector, and golden tokens as
     hex .mem files in the format host.cpp expects

Output files (saved to OUTPUT_DIR):

  INT8 weight files (2 hex digits per value, signed byte):
    norm_weight.mem       (D_MODEL,)
    in_proj_weight.mem    (D_INNER * 2 * D_MODEL,)  -- x rows then z rows
    conv_weight.mem       (D_INNER * D_CONV,)
    x_proj_weight.mem     ((DT_RANK + 2*D_STATE) * D_INNER,)
    dt_proj_weight.mem    (D_INNER * DT_RANK,)
    out_proj_weight.mem   (D_MODEL * D_INNER,)

  Q4.12 files (4 hex digits per value, signed 16-bit, scale=4096):
    conv_bias.mem         (D_INNER,)
    dt_proj_bias.mem      (D_INNER,)
    logA_mat.mem          (D_INNER * D_STATE,) -- NOTE: stores actual A,
                                                  not log(A), see comment below
    D_vec.mem             (D_INNER,)
    input_5tokens.mem     (N_GOLDEN * D_MODEL,)
    golden_5tokens.mem    (N_GOLDEN * D_MODEL,)

Requirements:
    pip install torch transformers datasets numpy

Usage:
    python generate_mem_files.py

"""

import numpy as np
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG, must match mamba_types.h and host.cpp
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID     = "state-spaces/mamba-790m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
OUTPUT_DIR   = Path("mem_files")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL  = 1536
D_INNER  = 3072
D_STATE  = 16
D_CONV   = 4
DT_RANK  = 96
N_GOLDEN = 5
SEQ_LEN  = 2048   # number of tokens fed through the model (only first N_GOLDEN saved)

Q412_SCALE = 4096.0   # ap_fixed<16,4> fractional scale (2^12)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def quantize_int8_symmetric(arr):
    """
    Per-tensor symmetric INT8 quantization.
    Returns (int8_array, scale_float).
    scale is chosen so that max(|arr|) maps to 127.
    To recover float on the host: int8_value * scale
    """
    abs_max = np.max(np.abs(arr))
    scale   = (abs_max / 127.0) if abs_max > 0 else 1.0
    q       = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, float(scale)


def save_int8_mem(path, arr):
    """
    Save INT8 array as hex .mem file.
    Each value is one line of 2 hex digits (unsigned representation of signed byte).
    host.cpp loads these with load_int8_scaled().
    """
    flat = arr.flatten().astype(np.int8)
    with open(path, 'w') as f:
        for v in flat:
            f.write(f'{int(v) & 0xFF:02x}\n')
    print(f"  Saved {path}  ({len(flat)} values, INT8)")


def save_q412_mem(path, arr):
    """
    Save float32 array as Q4.12 fixed-point hex .mem file.
    Each value is one line of 4 hex digits (unsigned representation of signed 16-bit).
    ap_fixed<16,4>: scale = 2^12 = 4096, representable range [-8, +8).
    host.cpp loads these with load_q412().
    """
    flat = arr.flatten().astype(np.float32)
    clipped = 0
    with open(path, 'w') as f:
        for v in flat:
            scaled = np.round(v * Q412_SCALE)
            if scaled > 32767 or scaled < -32768:
                clipped += 1
            q = int(np.clip(scaled, -32768, 32767))
            f.write(f'{q & 0xFFFF:04x}\n')
    if clipped > 0:
        print(f"  Saved {path}  ({len(flat)} values, Q4.12) WARNING: {clipped} values clipped to Q4.12 range")
    else:
        print(f"  Saved {path}  ({len(flat)} values, Q4.12)")


def get_param(model, name):
    """Fetch a parameter tensor from block 0 as a float32 numpy array."""
    full_name = f"backbone.layers.0.{name}"
    for n, p in model.named_parameters():
        if n == full_name:
            return p.detach().cpu().float().numpy()
    raise KeyError(f"Parameter not found: {full_name}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1, load model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Mamba-790M from HuggingFace")
print("=" * 60)

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Device : {DEVICE}")
print(f"Output : {OUTPUT_DIR.resolve()}\n")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = tokenizer.eos_token

# Build config manually. HuggingFace has a known bug where it ignores
# d_model from the checkpoint config for mamba-790m, so we set it explicitly.
config = MambaConfig(
    hidden_size=1536,
    num_hidden_layers=48,
    vocab_size=50280,
    state_size=16,
    expand=2,
    conv_kernel=4,
)

model = MambaForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=False,
)
model = model.to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Loaded. Total parameters: {total_params:,}\n")

print("Block 0 weight shapes (sanity check):")
for name, param in model.named_parameters():
    if "backbone.layers.0" in name:
        short = name.replace('backbone.layers.0.', '')
        print(f"  {short:30s} {list(param.shape)}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2, extract and quantize block 0 weights
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Extracting and quantizing block 0 weights")
print("=" * 60)

scales = {}

# norm.weight (D_MODEL,)
norm_w         = get_param(model, "norm.weight")
norm_q, norm_s = quantize_int8_symmetric(norm_w)
scales["norm_weight"] = norm_s
save_int8_mem(OUTPUT_DIR / "norm_weight.mem", norm_q)

# in_proj.weight (6144, 1536). first 3072 rows are x, next 3072 are z
in_proj_w         = get_param(model, "mixer.in_proj.weight")
assert in_proj_w.shape == (D_INNER * 2, D_MODEL), \
    f"in_proj shape mismatch: {in_proj_w.shape}"
in_proj_q, in_proj_s = quantize_int8_symmetric(in_proj_w)
scales["in_proj_weight"] = in_proj_s
save_int8_mem(OUTPUT_DIR / "in_proj_weight.mem", in_proj_q)

# conv1d.weight (3072, 1, 4) flatten to (3072, 4)
conv_w         = get_param(model, "mixer.conv1d.weight")
conv_w_flat    = conv_w.reshape(D_INNER, D_CONV)
conv_q, conv_s = quantize_int8_symmetric(conv_w_flat)
scales["conv_weight"] = conv_s
save_int8_mem(OUTPUT_DIR / "conv_weight.mem", conv_q)

# conv1d.bias (3072,). Q4.12, no quantization
conv_b = get_param(model, "mixer.conv1d.bias")
save_q412_mem(OUTPUT_DIR / "conv_bias.mem", conv_b)

# x_proj.weight (128, 3072). outputs dt(96) + B(16) + C(16) = 128
x_proj_w         = get_param(model, "mixer.x_proj.weight")
assert x_proj_w.shape == (DT_RANK + 2 * D_STATE, D_INNER), \
    f"x_proj shape mismatch: {x_proj_w.shape}"
x_proj_q, x_proj_s = quantize_int8_symmetric(x_proj_w)
scales["x_proj_weight"] = x_proj_s
save_int8_mem(OUTPUT_DIR / "x_proj_weight.mem", x_proj_q)

# dt_proj.weight (3072, 96)
dt_proj_w         = get_param(model, "mixer.dt_proj.weight")
assert dt_proj_w.shape == (D_INNER, DT_RANK), \
    f"dt_proj shape mismatch: {dt_proj_w.shape}"
dt_proj_q, dt_proj_s = quantize_int8_symmetric(dt_proj_w)
scales["dt_proj_weight"] = dt_proj_s
save_int8_mem(OUTPUT_DIR / "dt_proj_weight.mem", dt_proj_q)

# dt_proj.bias (3072,). Q4.12
dt_proj_b = get_param(model, "mixer.dt_proj.bias")
save_q412_mem(OUTPUT_DIR / "dt_proj_bias.mem", dt_proj_b)

# out_proj.weight (1536, 3072)
out_proj_w         = get_param(model, "mixer.out_proj.weight")
assert out_proj_w.shape == (D_MODEL, D_INNER), \
    f"out_proj shape mismatch: {out_proj_w.shape}"
out_proj_q, out_proj_s = quantize_int8_symmetric(out_proj_w)
scales["out_proj_weight"] = out_proj_s
save_int8_mem(OUTPUT_DIR / "out_proj_weight.mem", out_proj_q)

# A matrix (3072, 16). Stored in a file called logA_mat.mem for historical
# reasons, but the kernel expects the linear A matrix, not log(A).
# The model parameter A_log is defined such that A = -exp(A_log), so we
# compute that and save it as Q4.12.
A_log = get_param(model, "mixer.A_log")
assert A_log.shape == (D_INNER, D_STATE), \
    f"A_log shape mismatch: {A_log.shape}"
A_mat = -np.exp(A_log).astype(np.float32)
print(f"  A_mat range: [{A_mat.min():.4f}, {A_mat.max():.4f}] (Q4.12 limit: [-8, +8))")
save_q412_mem(OUTPUT_DIR / "logA_mat.mem", A_mat)

# D vector (3072,). SSM skip connection, Q4.12
D_vec = get_param(model, "mixer.D")
assert D_vec.shape == (D_INNER,), \
    f"D_vec shape mismatch: {D_vec.shape}"
save_q412_mem(OUTPUT_DIR / "D_vec.mem", D_vec)

print()
print("Computed scale factors:")
for k, v in scales.items():
    print(f"  {k:20s}  {v:.18f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3, reload dequantized weights into block 0
# This makes the GPU forward pass mirror FPGA precision so the captured
# golden tokens are consistent with what the kernel will compute.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Reloading dequantized block 0 weights into the model")
print("=" * 60)

mapping = {
    "norm.weight":          (norm_q,     norm_s),
    "mixer.in_proj.weight": (in_proj_q,  in_proj_s),
    "mixer.conv1d.weight":  (conv_q,     conv_s),
    "mixer.x_proj.weight":  (x_proj_q,   x_proj_s),
    "mixer.dt_proj.weight": (dt_proj_q,  dt_proj_s),
    "mixer.out_proj.weight":(out_proj_q, out_proj_s),
}

with torch.no_grad():
    for name, param in model.named_parameters():
        if "backbone.layers.0" not in name:
            continue
        short = name.replace("backbone.layers.0.", "")
        if short in mapping:
            q_arr, s = mapping[short]
            w_float = torch.tensor(
                q_arr.astype(np.float32) * s,
                dtype=torch.float32
            ).reshape(param.shape)
            param.copy_(w_float)

print("Block 0 weights reloaded as dequantized INT8.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4, capture golden input/output tokens using WikiText-103
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Capturing golden tokens (WikiText-103 test split)")
print("=" * 60)

print("Loading WikiText-103...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
full_text = " ".join([x for x in dataset["text"] if len(x.strip()) > 0])
tokens    = tokenizer.encode(full_text)
print(f"Total WikiText tokens: {len(tokens):,}")
assert len(tokens) >= SEQ_LEN, "WikiText-103 produced fewer tokens than expected"

# Use the first SEQ_LEN tokens, matching how the original artifact was generated
input_ids = torch.tensor([tokens[:SEQ_LEN]], dtype=torch.long).to(DEVICE)
print(f"Feeding {input_ids.shape[1]} tokens through the model\n")

captured_inputs  = []
captured_outputs = []

def hook_fn(module, inp, out):
    x_in  = inp[0].detach().cpu().float().numpy()
    x_out = (out if isinstance(out, torch.Tensor) else out[0]) \
            .detach().cpu().float().numpy()
    captured_inputs.append(x_in)
    captured_outputs.append(x_out)

handle = model.backbone.layers[0].register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(input_ids)

handle.remove()

assert len(captured_inputs) == 1
block_in  = captured_inputs[0][0]   # (SEQ_LEN, D_MODEL)
block_out = captured_outputs[0][0]  # (SEQ_LEN, D_MODEL)
print(f"Captured: input {block_in.shape}, output {block_out.shape}\n")

golden_inputs  = block_in[:N_GOLDEN]
golden_outputs = block_out[:N_GOLDEN]

print(f"Input  range: [{golden_inputs.min():.4f},  {golden_inputs.max():.4f}]")
print(f"Output range: [{golden_outputs.min():.4f}, {golden_outputs.max():.4f}]")

if np.max(np.abs(golden_inputs)) >= 8.0:
    print(f"  WARNING: input exceeds Q4.12 range [-8,+8)")
if np.max(np.abs(golden_outputs)) >= 8.0:
    print(f"  WARNING: output exceeds Q4.12 range [-8,+8)")

save_q412_mem(OUTPUT_DIR / "input_5tokens.mem",  golden_inputs)
save_q412_mem(OUTPUT_DIR / "golden_5tokens.mem", golden_outputs)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "model":       MODEL_ID,
    "device":      DEVICE,
    "n_golden":    N_GOLDEN,
    "seq_len":     SEQ_LEN,
    "d_model":     D_MODEL,
    "d_inner":     D_INNER,
    "d_state":     D_STATE,
    "dt_rank":     DT_RANK,
    "d_conv":      D_CONV,
    "q412_scale":  int(Q412_SCALE),
    "scales":      scales,
    "input_range":  {"min": float(golden_inputs.min()),  "max": float(golden_inputs.max())},
    "output_range": {"min": float(golden_outputs.min()), "max": float(golden_outputs.max())},
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"""
{'=' * 60}
DONE. Files written to: {OUTPUT_DIR.resolve()}
{'=' * 60}

INT8 weight files (load_int8_scaled in host.cpp):
  norm_weight.mem
  in_proj_weight.mem
  conv_weight.mem
  x_proj_weight.mem
  dt_proj_weight.mem
  out_proj_weight.mem

Q4.12 files (load_q412 in host.cpp):
  conv_bias.mem
  dt_proj_bias.mem
  logA_mat.mem        (stores linear A = -exp(A_log), not the raw A_log)
  D_vec.mem
  input_5tokens.mem
  golden_5tokens.mem

Verify host.cpp #define SCALE_* values match the computed scales above.
Copy the .mem files to the BASE path in host.cpp.
""")