from dataclasses import dataclass, field

import torch
import torch.nn as nn
from safetensors.torch import load_file

from seva.modules.layers import (
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    timestep_embedding,
)
from seva.modules.transformer import MultiviewTransformer


@dataclass
class SevaParams(object):
    # 11 = 4 latent + 1 mask + 6 plucker; SplatGuide adds 4 rendered-latent channels (15).
    in_channels: int = 11
    model_channels: int = 320
    out_channels: int = 4
    num_frames: int = 21
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 2, 1])
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_head_channels: int = 64
    transformer_depth: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    context_dim: int = 1024  # CLIP
    # Reconstruction-token dim (WorldMirror camera + register tokens); None disables attn3.
    token_dim: int | None = None
    dense_in_channels: int = 6
    dropout: float = 0.0
    unflatten_names: list[str] = field(
        default_factory=lambda: ["middle_ds8", "output_ds4", "output_ds2"]
    )
    ckpt_path: str | None = None
    freeze_time_embed: bool = True
    use_checkpoint: bool = True

    def __post_init__(self):
        assert len(self.channel_mult) == len(self.transformer_depth)


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    if len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def read_state_dict(path: str, prefix: str = "model.diffusion_model.") -> dict:
    """Seva weights from a `.safetensors` file (SEVA or exported SplatGuide) or from the U-Net
    part of a SplatGuide Lightning `.ckpt`."""
    if not path.endswith(".ckpt"):
        return load_file(path)
    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    return {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


class Seva(nn.Module):
    def __init__(self, params: SevaParams) -> None:
        super().__init__()
        self.params = params

        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.time_embed.requires_grad_(not params.freeze_time_embed)

        def transformer(ch, name, depth):
            return MultiviewTransformer(
                ch,
                ch // params.num_head_channels,
                params.num_head_channels,
                name=name,
                depth=depth,
                context_dim=params.context_dim,
                token_dim=params.token_dim,
                unflatten_names=params.unflatten_names,
                use_checkpoint=params.use_checkpoint,
            )

        def resblock(ch, out_ch):
            return ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=out_ch,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    resblock(ch, mult * params.model_channels)
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    input_layers.append(
                        transformer(ch, f"input_ds{ds}", params.transformer_depth[level])
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=ch))
                )
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            resblock(ch, None),
            transformer(ch, f"middle_ds{ds}", params.transformer_depth[-1]),
            resblock(ch, None),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    resblock(ch + ich, params.model_channels * mult)
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    output_layers.append(
                        transformer(ch, f"output_ds{ds}", params.transformer_depth[level])
                    )
                if level and i == params.num_res_blocks:
                    ds //= 2
                    output_layers.append(Upsample(ch, ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )

        if params.ckpt_path is not None:
            self.load_pretrained(params.ckpt_path)

    def load_pretrained(self, path: str):
        """Initialise from SEVA or SplatGuide weights. Starting from SEVA, the extra input
        channels (rendered latents) are zero-initialised and attn3/norm4 are created from
        attn2/norm2 with fresh key/value projections, so training starts from SEVA's behaviour."""
        print(f"Loading checkpoint from {path}...")
        state_dict = read_state_dict(path)
        self._adapt_seva_state_dict(state_dict)
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=True)
        print_load_warning(missing, unexpected)

    def _adapt_seva_state_dict(self, state_dict: dict):
        key = "input_blocks.0.0.weight"
        old_weight = state_dict[key]
        if old_weight.shape[1] != self.params.in_channels:
            print(
                f"Extending input channels from {old_weight.shape[1]} to "
                f"{self.params.in_channels} with zero initialization"
            )
            new_weight = old_weight.new_zeros(
                old_weight.shape[0], self.params.in_channels, *old_weight.shape[2:]
            )
            new_weight[:, : old_weight.shape[1]] = old_weight
            state_dict[key] = new_weight

        if self.params.token_dim is None or any("attn3." in k for k in state_dict):
            return
        for attn2_key in [k for k in state_dict if "attn2." in k]:
            attn3_key = attn2_key.replace("attn2.", "attn3.")
            attn2_weight = state_dict[attn2_key]
            if "to_k.weight" in attn3_key or "to_v.weight" in attn3_key:
                new_weight = attn2_weight.new_zeros(
                    attn2_weight.shape[0], self.params.token_dim
                )
                torch.nn.init.xavier_uniform_(new_weight, gain=0.02)
                state_dict[attn3_key] = new_weight
            else:
                state_dict[attn3_key] = attn2_weight.clone()
        for norm2_key in [
            k
            for k in state_dict
            if "norm2." in k and ("transformer_blocks" in k or "time_mix_blocks" in k)
        ]:
            state_dict[norm2_key.replace("norm2.", "norm4.")] = state_dict[norm2_key].clone()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
        token_y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        kwargs = dict(
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            token_context=token_y,
        )

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, **kwargs)
            hs.append(h)
        h = self.middle_block(h, **kwargs)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, **kwargs)
        h = h.type(x.dtype)
        return self.out(h)


class SGMWrapper(nn.Module):
    """Maps the condition dict onto Seva's inputs. Frames are flattened into the batch dim
    and `c` holds per-frame conditions (training and `seva.inference.do_sample`).
    The attribute name gives the `model.diffusion_model.` checkpoint prefix."""

    def __init__(self, diffusion_model: Seva):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c["concat"]), dim=1)
        # Tokens come as [B, V*5, D]; repeat them for every frame of each sample.
        tokens = c.get("tokens")
        if tokens is not None:
            num_frames = c["crossattn"].shape[0] // tokens.shape[0]
            tokens = tokens.unsqueeze(1).expand(-1, num_frames, -1, -1)
            tokens = tokens.reshape(-1, *tokens.shape[2:])
        return self.diffusion_model(
            x,
            t=t,
            y=c["crossattn"],
            dense_y=c["dense_vector"],
            token_y=tokens,
            **kwargs,
        )
