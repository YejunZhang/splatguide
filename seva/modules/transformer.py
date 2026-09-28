from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    """SEVA spatial block. With `token_dim` set, an extra cross-attention
    (attn3/norm4) over reconstruction tokens runs in parallel to the CLIP one."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        token_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.attn3 = None
        self.norm4 = None
        if token_dim is not None:
            self.attn3 = Attention(
                query_dim=dim,
                context_dim=token_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
            )
            self.norm4 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        token_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        h = self.attn2(self.norm2(x), context=context) + x
        if token_context is not None:
            h = h + self.attn3(self.norm4(x), context=token_context)
        x = self.ff(self.norm3(h)) + h
        return x


class TransformerBlockTimeMix(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        token_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout)
        self.attn1 = Attention(
            query_dim=inner_dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=inner_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.attn3 = None
        self.norm4 = None
        if token_dim is not None:
            self.attn3 = Attention(
                query_dim=inner_dim,
                context_dim=token_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
            )
            self.norm4 = nn.LayerNorm(inner_dim)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_frames: int,
        token_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, s, _ = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=num_frames)
        x = self.ff_in(self.norm_in(x)) + x
        x = self.attn1(self.norm1(x), context=None) + x
        h = self.attn2(self.norm2(x), context=context) + x
        if token_context is not None:
            # token_context is [b, L, D], the same for every spatial position: attend once per
            # sample instead of repeating the tokens h*w times.
            q = rearrange(self.norm4(x), "(b s) t c -> b (s t) c", b=len(token_context))
            out = self.attn3(q, context=token_context)
            h = h + rearrange(out, "b (s t) c -> (b s) t c", t=num_frames)
        x = self.ff(self.norm3(h))
        x = rearrange(x, "(b s) t c -> (b t) s c", s=s)
        return x


def _call(fn, *args):
    return fn(*args)


class MultiviewTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        name: str,
        unflatten_names: list[str] = [],
        depth: int = 1,
        context_dim: int = 1024,
        token_dim: int | None = None,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        # Unflattened blocks attend across all frames jointly.
        self.unflatten = name in unflatten_names
        self.use_checkpoint = use_checkpoint

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    token_dim=token_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.time_mix_blocks = nn.ModuleList(
            [
                TransformerBlockTimeMix(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    token_dim=token_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_frames: int,
        token_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert context.ndim == 3
        _, _, h, w = x.shape
        x_in = x

        time_context = repeat(context[::num_frames], "b ... -> (b n) ...", n=h * w)
        token_time_context = None if token_context is None else token_context[::num_frames]
        if self.unflatten:
            context = context[::num_frames]
            token_context = token_time_context

        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj_in(x)

        call = partial(checkpoint, use_reentrant=False) if self.use_checkpoint and self.training else _call
        for block, mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
            if self.unflatten:
                x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w)
            x = call(block, x, context, token_context)
            if self.unflatten:
                x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w)
            x = x + call(mix_block, x, time_context, num_frames, token_time_context)

        x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + x_in
