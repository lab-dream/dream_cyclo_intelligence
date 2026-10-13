# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Chunk-update decider network.

Every environment step the frozen base policy proposes a *new* action chunk. This
module decides, from two action sequences, whether to KEEP executing the current
chunk or UPDATE to the freshly predicted one, and -- if it updates -- WHICH index
of the new chunk to start executing from.

Inputs
------
new_chunk : [B, L, d_a]
    Newly predicted action chunk (fixed length ``L``).
ctx : [B, H+L, d_a]
    Concatenation of ``[past actions (fixed horizon H) ; remaining part of the
    current chunk (left-aligned, executed actions removed)]``. If the remaining
    part is shorter than ``L`` it arrives right-zero-padded to length ``L``.
ctx_mask : [B, H+L] bool
    True for valid positions, False for padding.

Alignment convention: both the new chunk and the remaining chunk start at the
current time ``t``, so the correct alignment diagonal sits at offset 0 -- row ``i``
of the new chunk aligns with column ``H+i`` of ``ctx``. We deliberately do NOT add
positional/time embeddings to Q/K: positional information is carried by the
spatial structure of the 2D map (and by the row embedding in the index head).

Two map variants (selected by ``config.map_type``) share every other component:
    * "attention" : masked-softmax scaled dot-product; padded columns become 0.
    * "cosine"    : learnable-temperature cosine similarity; padded columns set 0.
"""

from dataclasses import dataclass
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChunkUpdateConfig:
    """Configuration for :class:`ChunkUpdateDecider`.

    Attributes:
        map_type: "attention" or "cosine".
        L: length of the new action chunk (and the padded remaining chunk).
        H: fixed horizon of past actions.
        d_a: action dimension.
        d_model: projection dimension for Q/K.
        index_decay: log-probability drop from index 0 to index ``L-1``.
            The initial index distribution is proportional to
            ``exp(-index_decay * i / (L - 1))``.
    """

    map_type: str = "attention"
    L: int = 50
    H: int = 20
    d_a: int = 14
    d_model: int|None = 64
    index_decay: float = 4.0

    @property
    def ctx_len(self) -> int:
        """Total context length ``H + L``."""
        return self.H + self.L


class ChunkUpdateDecider(nn.Module):
    """Decides whether to keep/update the executed chunk and where to resume.

    The two map variants ("attention" / "cosine") differ ONLY in how the
    ``[B, L, H+L]`` similarity map is built; the mask channel, conv trunk and both
    output heads are strictly identical.
    """

    def __init__(self, config: ChunkUpdateConfig) -> None:
        super().__init__()
        assert config.map_type in ("attention", "cosine"), (
            f"map_type must be 'attention' or 'cosine', got {config.map_type!r}"
        )
        assert config.index_decay >= 0.0, "index_decay must be non-negative"
        self.config = config
        L, d_a, d_model = config.L, config.d_a, config.d_model
        ctx_len = config.ctx_len  # H + L

        # --- 1. projection (shared by both variants) ---
        if d_model is not None:
            self.proj_q = nn.Linear(d_a, d_model)   # [B, L, d]   from new_chunk
            self.proj_k = nn.Linear(d_a, d_model)   # [B, H+L, d] from ctx
        else:
            print("Model without projection")
            self.proj_q = nn.Identity()             # [B, L, d]   from new_chunk
            self.proj_k = nn.Identity()             # [B, H+L, d] from ctx

        # --- 2. cosine-variant learnable temperature ---
        if config.map_type == "cosine":
            # init at 1.0; clamped to >= 0.01 in forward for numerical safety.
            self.tau = nn.Parameter(torch.ones(()))
        else:
            self.tau = None

        # --- 4. shared conv trunk (input has 2 channels: map + mask) ---
        # receptive field ~9 along the diagonal for break-point detection.
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(2, 8),
            nn.GELU(),
        )  # -> [B, 8, L, H+L]

        # --- 5. index head (start-index selection over the new chunk) ---
        # learnable per-row embedding: the aligned column differs per row
        # (row i <-> column H+i), so a purely shared linear head could not express
        # row-dependent column selection.
        self.row_embed = nn.Parameter(torch.randn(L, 16) * 0.5)
        index_in = 8 * ctx_len + 16
        self.index_mlp = nn.Sequential(
            nn.LayerNorm(index_in),
            nn.Linear(index_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
        )  # applied per row -> [B, L, 1] -> squeeze -> [B, L]
        # A linear ramp in logit space is an exponential categorical prior.
        # Keep it non-persistent so old checkpoints remain state-dict compatible;
        # the decay value itself is saved as part of ChunkUpdateConfig.
        self.register_buffer(
            "index_prior_logits",
            torch.linspace(0.0, -config.index_decay, L),
            persistent=False,
        )
        # Start with only a tiny input-dependent residual around the prior while
        # retaining non-zero gradients through the complete index head.
        nn.init.normal_(self.index_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.index_mlp[-1].bias)

        # --- 6. update head (switch decision) ---
        self.update_conv = nn.Conv2d(8, 2, kernel_size=1)   # -> [B, 2, L, H+L]
        update_in = 2 * L * ctx_len
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(update_in),
            nn.Linear(update_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )  # -> [B, 1]
        # Initialize update head so initial switching probability is exactly 1/8 (0.125):
        # sigmoid(b) = 1/8  =>  b = -ln(7) ≈ -1.9459
        nn.init.normal_(self.update_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.update_mlp[-1].bias, -math.log(7.0))

        # NOTE: this is the actor only. Under the asymmetric actor-critic setup the
        # value function is a separate, sim-only PrivilegedCritic (see critic.py).

    # ------------------------------------------------------------------ #
    # map computation
    # ------------------------------------------------------------------ #
    def _compute_map(
        self, q: torch.Tensor, k: torch.Tensor, ctx_mask: torch.Tensor
    ) -> torch.Tensor:
        """Build the ``[B, L, H+L]`` similarity map with padded columns zeroed.

        Args:
            q: [B, L, d_model] projected new chunk.
            k: [B, H+L, d_model] projected context.
            ctx_mask: [B, H+L] bool, True for valid columns.

        Returns:
            [B, L, H+L] map; padded columns are exactly 0 for both variants.
        """
        # broadcast the column mask over query rows: [B, 1, H+L].
        col_valid = ctx_mask.unsqueeze(1)

        if self.config.map_type == "attention":
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.config.d_model ** 0.5)
            scores = scores.masked_fill(~col_valid, float("-inf"))
            attn = torch.softmax(scores, dim=-1)   # padded columns -> exactly 0
            return attn

        # cosine variant.
        qn = F.normalize(q, dim=-1, eps=1e-6)
        kn = F.normalize(k, dim=-1, eps=1e-6)
        tau = self.tau.clamp(min=0.01)
        m = torch.matmul(qn, kn.transpose(-1, -2)) / tau
        # explicit masked_fill is required: proj_k bias makes padded keys non-zero
        # and normalization would rescale them to unit norm, so the eps alone is
        # not enough to neutralize padding.
        m = m.masked_fill(~col_valid, 0.0)
        return m

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        new_chunk: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the decider.

        Args:
            new_chunk: [B, L, d_a] newly predicted action chunk.
            ctx: [B, H+L, d_a] ``[past ; remaining current chunk (padded)]``.
            ctx_mask: [B, H+L] bool, True for valid positions.

        Returns:
            dict with:
                index_logits: [B, L] raw logits over the new-chunk start index.
                update_logit: [B, 1] raw logit for the keep/update decision.
                map:          [B, L, H+L] similarity map (for viz/debugging).
        """
        # numerical safety: an all-padding row would make softmax produce NaN.
        assert torch.all(ctx_mask.any(dim=-1)), "every ctx_mask row needs >=1 valid position"

        B = new_chunk.shape[0]
        L, ctx_len = self.config.L, self.config.ctx_len

        # --- 1. projection ---
        q = self.proj_q(new_chunk)   # [B, L, d]
        k = self.proj_k(ctx)         # [B, H+L, d]

        # --- 2. map ---
        attn_map = self._compute_map(q, k, ctx_mask)   # [B, L, H+L]

        # --- 3. conv input: stack map + float mask channel -> [B, 2, L, H+L] ---
        mask_ch = ctx_mask.to(attn_map.dtype)                       # [B, H+L]
        mask_ch = mask_ch.view(B, 1, 1, ctx_len).expand(B, 1, L, ctx_len)
        conv_in = torch.cat([attn_map.unsqueeze(1), mask_ch], dim=1)  # [B, 2, L, H+L]

        # --- 4. shared conv trunk ---
        feat = self.conv(conv_in)   # [B, 8, L, H+L]

        # --- 5. index head ---
        # per-row features: [B, 8, L, H+L] -> [B, L, 8*(H+L)].
        row_feat = feat.permute(0, 2, 1, 3).reshape(B, L, 8 * ctx_len)
        row_emb = self.row_embed.unsqueeze(0).expand(B, L, 16)
        row_in = torch.cat([row_feat, row_emb], dim=-1)             # [B, L, 8*(H+L)+16]
        index_logits = self.index_mlp(row_in).squeeze(-1)          # [B, L]
        index_logits = index_logits + self.index_prior_logits      # exponential initial prior

        # --- 6. update head ---
        upd = self.update_conv(feat)                               # [B, 2, L, H+L]
        update_logit = self.update_mlp(upd.reshape(B, -1))         # [B, 1]

        return {
            "index_logits": index_logits,
            "update_logit": update_logit,
            "map": attn_map,
        }
