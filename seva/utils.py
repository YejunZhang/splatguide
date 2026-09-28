import torch

from seva.model import Seva, SevaParams, print_load_warning, read_state_dict


def load_seva(path: str, device: str | torch.device = "cuda") -> Seva:
    """Load SEVA or SplatGuide weights for inference in bf16. Input channels (11 = SEVA,
    15 = + rendered latents) and the token cross-attention are inferred from the weights."""
    state_dict = read_state_dict(path)
    token_keys = [k for k in state_dict if k.endswith("attn3.to_k.weight")]
    params = SevaParams(
        in_channels=state_dict["input_blocks.0.0.weight"].shape[1],
        token_dim=state_dict[token_keys[0]].shape[1] if token_keys else None,
    )
    model = Seva(params)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print_load_warning(missing, unexpected)
    print(
        f"Loaded {path}: in_channels={params.in_channels}, "
        f"token attention={'on' if params.token_dim else 'off'}"
    )
    return model.to(torch.bfloat16).to(device).eval()
