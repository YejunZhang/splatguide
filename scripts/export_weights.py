"""Export the diffusion U-Net of a Lightning checkpoint to safetensors (no optimizer state).
Run it in an environment whose NumPy major version matches the one that wrote the checkpoint.

    python scripts/export_weights.py logs/splatguide/checkpoints/epoch=012.ckpt checkpoints/splatguide.safetensors
"""
import os
import sys

from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seva.model import read_state_dict  # noqa: E402

save_file({k: v.contiguous() for k, v in read_state_dict(sys.argv[1]).items()}, sys.argv[2])
print(f"Saved {sys.argv[2]}")
