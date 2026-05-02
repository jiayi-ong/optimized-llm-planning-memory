"""
compressor/tree_gat.py
======================
Lightweight path-set attention module for the MCTSGraphAttentionDistiller.

Design: Dense multi-head attention over a small node set
---------------------------------------------------------
MCTS trees projected by ``MCTSTreeRepresentation`` contain at most 3–5 paths
(1 best path + up to K alternative paths). For this small cardinality,
sparse graph operations (PyTorch Geometric, DGL) add dependency overhead
with no practical benefit. Instead we use standard dense multi-head attention
where every node can attend to every other node.

The "bidirectional" property of the original GAT design maps cleanly here:
  - Layer 1 (bottom-up proxy): nodes attend to each other freely
  - Layer 2 (top-down proxy): nodes attend to each other, with the
    best-path node boosted as a "root anchor" via additive bias

This captures:
  - What the best path tells us about the current planning state
  - What alternatives tell us that the best path lacks
  - Which alternatives are most constraint-relevant

No PyTorch Geometric is required — only standard PyTorch.

Usage
-----
    encoder = PathSetEncoder(in_dim=512, num_heads=4, dropout=0.1)
    out = encoder(node_feats, best_path_mask)  # [N, 512]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _MultiHeadAttentionLayer(nn.Module):
    """
    Single multi-head attention layer over N node embeddings.

    Each node attends to every other node (dense, fully-connected).
    Residual connection + LayerNorm applied after attention.

    Architecture note
    -----------------
    We implement this from scratch rather than using ``nn.MultiheadAttention``
    to allow per-node structural biases (e.g., best-path anchoring in Layer 2)
    without subclassing or monkey-patching the PyTorch built-in.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._scale = math.sqrt(self._head_dim)

        # QKV projections (fused for efficiency)
        self._qkv = nn.Linear(dim, 3 * dim, bias=False)
        self._out_proj = nn.Linear(dim, dim, bias=False)
        self._dropout = nn.Dropout(dropout)
        self._norm = nn.LayerNorm(dim)
        self._ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self._ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,                         # [N, dim]
        attn_bias: Tensor | None = None,   # [N, N] additive bias on attention logits
    ) -> Tensor:                           # [N, dim]
        N, dim = x.shape

        # Compute Q, K, V — shape before split: [N, 3*dim]
        qkv = self._qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)    # each [N, dim]

        # Reshape for multi-head: [N, H, head_dim]
        q = q.view(N, self._num_heads, self._head_dim)
        k = k.view(N, self._num_heads, self._head_dim)
        v = v.view(N, self._num_heads, self._head_dim)

        # Scaled dot-product attention: [H, N, N]
        q = q.permute(1, 0, 2)  # [H, N, head_dim]
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        scores = torch.bmm(q, k.transpose(1, 2)) / self._scale  # [H, N, N]

        if attn_bias is not None:
            # Broadcast bias across heads: attn_bias [N, N] → [1, N, N]
            scores = scores + attn_bias.unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)                 # [H, N, N]
        attn_weights = self._dropout(attn_weights)

        attn_out = torch.bmm(attn_weights, v)                    # [H, N, head_dim]
        attn_out = attn_out.permute(1, 0, 2).reshape(N, dim)     # [N, dim]
        attn_out = self._out_proj(attn_out)

        # Residual + LayerNorm
        x = self._norm(x + attn_out)

        # Feed-forward sub-layer
        ff_out = self._ff(x)
        x = self._ff_norm(x + ff_out)

        return x


class PathSetEncoder(nn.Module):
    """
    Two-layer bidirectional path-set attention for MCTS tree distillation.

    Encodes a small set of path embeddings (best path + alternative paths)
    by running 2 rounds of multi-head self-attention.

    Layer 1 — All-to-all attention
        Every path attends to every other. Captures:
          - What does path A know that path B doesn't?
          - Where do the paths agree?

    Layer 2 — Root-anchored attention
        Same as Layer 1 but adds a learnable bias toward the best-path node,
        simulating the "top-down" direction in a tree where the root (best path)
        provides global context to all branch nodes.

    Parameters
    ----------
    dim       : Feature dimension (must match T5-small hidden = 512).
    num_heads : Number of attention heads.
    dropout   : Dropout probability on attention weights.

    Usage
    -----
        encoder = PathSetEncoder(dim=512, num_heads=4)
        best_mask = torch.zeros(N).bool(); best_mask[0] = True  # first node is best path
        out = encoder(node_feats, best_path_idx=0)  # [N, 512]
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._layer1 = _MultiHeadAttentionLayer(dim, num_heads, dropout)
        self._layer2 = _MultiHeadAttentionLayer(dim, num_heads, dropout)

        # Learnable bias scalar that boosts best-path attention in Layer 2.
        # One value per head to allow head-specific root-attention strength.
        self._root_anchor_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        node_feats: Tensor,   # [N, dim] — best-path node MUST be at index 0
        n_nodes: int,         # actual number of nodes (may be padded to fixed size)
    ) -> Tensor:              # [N, dim]
        """
        Parameters
        ----------
        node_feats : [N, dim] tensor of path embeddings. Best-path node at index 0.
        n_nodes    : Number of actual nodes (rest are padding). Used to mask attention.

        Returns
        -------
        [N, dim] tensor of contextualised path embeddings.
        """
        N = node_feats.size(0)

        # Padding mask: prevent padded nodes from participating in attention
        # (only matters if N > n_nodes, which happens when batch is padded)
        if n_nodes < N:
            pad_mask = torch.zeros(N, N, device=node_feats.device)
            pad_mask[:, n_nodes:] = -1e9   # mask out padded columns (keys/values)
        else:
            pad_mask = None

        # Layer 1: all-to-all (bottom-up proxy)
        out = self._layer1(node_feats, attn_bias=pad_mask)

        # Layer 2: root-anchored (top-down proxy)
        # Add learnable bias on the column corresponding to the best-path node (index 0)
        # so all nodes give extra attention to the root-like best-path representation.
        root_bias = torch.zeros(N, N, device=node_feats.device)
        root_bias[:, 0] = self._root_anchor_bias  # boost column 0 (best-path node)
        if pad_mask is not None:
            root_bias = root_bias + pad_mask

        out = self._layer2(out, attn_bias=root_bias)

        return out
