# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
WhaleKV: Navigating the Deep Context Ocean via Topological KV Cache Compression

Three core components:
  ASG  — Atomic Super-node Graph
  QSP  — Query-driven Semantic Propagation
  GTA  — Global Topological Anchoring (multi-turn)
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from whalekv.presses.scorer_press import ScorerPress
from whalekv.presses.snapkv_press import SnapKVPress


@dataclass
class WhaleKVPress(ScorerPress):
    """
    WhaleKV (Single-Turn): Atomic Super-node Graph + Query-driven Semantic Propagation.

    Addresses semantic blindness of query-aware point-wise KV cache compression.

    Pipeline
    --------
    ASG — Atomic Super-node Graph Construction:
        Tokens are grouped into fixed-size micro-chunks (size C). Each chunk forms
        a super-node with mean-pooled key features (semantic representation) and
        max-pooled attention scores (importance signal). This reduces graph complexity
        from O(L²) to O((L/C)²) and prevents token tearing of multi-token entities.

    QSP — Query-driven Semantic Propagation:
        A cosine similarity graph is built over super-node embeddings and sparsified
        via KNN (top-k neighbors). Attention energy is propagated over this graph
        with one-step random walk diffusion, recovering semantically related tokens
        that receive insufficient direct query attention.

    Score Fusion (eq. 7 in paper):
        Score_super = A_super + A_prop
        where A_super is the base atomic attention score and A_prop is the propagated
        structural signal. Super-node scores are broadcast to constituent tokens,
        treating each chunk as an atomic all-or-nothing retention unit.

    Parameters
    ----------
    compression_ratio : float, default=0.0
    window_size : int, default=64
    kernel_size : int, default=5
    micro_chunk_size : int, default=4
        Chunk size C for atomic super-node construction.
    top_k_edges : int or None, default=16
        Number of KNN neighbors per super-node for graph sparsification.
    similarity_threshold : float or None, default=None
        Optional lower bound for edge weights after KNN sparsification.
    alpha : float, default=1.0
        Weight for the QSP propagation term (A_prop).
    n_sink : int, default=4
        Number of initial sink tokens always retained.
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5
    micro_chunk_size: int = 4
    top_k_edges: Optional[int] = 16
    similarity_threshold: Optional[float] = None
    alpha: float = 1.0
    n_sink: int = 4

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_kv_groups = module.config.num_attention_heads // num_kv_heads
        k_context = k_len - self.window_size

        assert k_len > self.window_size, (
            f"Sequence length {k_len} must be greater than window_size={self.window_size}"
        )

        # ── Attention Assessment and Initialization (ASG §4.1) ───────────────────
        # Compute S_init from the observation window, then apply 1D avg-pooling
        if attentions is not None:
            attn_weights = attentions[..., -self.window_size:, :-self.window_size]
        else:
            attn_weights = SnapKVPress.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        attn_scores = attn_weights.mean(dim=-2)
        attn_scores = F.avg_pool1d(attn_scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        attn_scores = attn_scores.view(bsz, num_kv_heads, num_kv_groups, k_context).mean(dim=2)

        K_ctx = keys[:, :, :k_context, :]

        # ── Super-node Construction (ASG §4.1) ───────────────────────────────────
        # Pad to a multiple of C, form M = L_pad / C super-nodes
        C = self.micro_chunk_size
        remainder = k_context % C
        pad_len = (C - remainder) if remainder > 0 else 0
        padded_len = k_context + pad_len
        M = padded_len // C

        if pad_len > 0:
            K_ctx_padded = F.pad(K_ctx, (0, 0, 0, pad_len))
            attn_padded = F.pad(attn_scores, (0, pad_len))
        else:
            K_ctx_padded = K_ctx
            attn_padded = attn_scores

        # Mean-pooling for semantic representation K_super (eq. 2)
        K_view = K_ctx_padded.view(bsz, num_kv_heads, M, C, head_dim)
        super_K = K_view.mean(dim=3)

        # Max-pooling for attention score A_super (eq. 3)
        A_view = attn_padded.view(bsz, num_kv_heads, M, C)
        super_A = A_view.max(dim=3)[0]

        # ── Similarity Graph Construction (QSP §4.2) ─────────────────────────────
        # Cosine similarity matrix S (eq. 4), self-loops masked
        super_K_norm = F.normalize(super_K, p=2, dim=-1)
        macro_sim = torch.matmul(super_K_norm, super_K_norm.transpose(-1, -2)).clamp(min=0)

        diag = torch.eye(M, device=keys.device, dtype=torch.bool)
        macro_sim = macro_sim.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)

        # KNN sparsification: keep only top-k strongest edges per node
        if self.top_k_edges is not None and self.top_k_edges > 0 and self.top_k_edges < M:
            topk_vals, _ = torch.topk(macro_sim, k=self.top_k_edges, dim=-1)
            kth_thresholds = topk_vals[..., -1:]
            macro_sim = macro_sim.masked_fill(macro_sim < kth_thresholds, 0.0)

        if self.similarity_threshold is not None:
            macro_sim = macro_sim.masked_fill(macro_sim < self.similarity_threshold, 0.0)

        # ── Attention Graph Propagation (QSP §4.2) ───────────────────────────────
        # Row-normalized transition matrix W (eq. 5)
        row_sum = macro_sim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        W = macro_sim / row_sum

        # One-step diffusion: A_prop = W · A_super (eq. 6)
        super_propagated = torch.bmm(
            W.view(bsz * num_kv_heads, M, M),
            super_A.view(bsz * num_kv_heads, M, 1),
        ).squeeze(-1).view(bsz, num_kv_heads, M)

        # ── Score Fusion and Broadcast (eq. 7) ───────────────────────────────────
        # Score_super = A_super + A_prop; broadcast to token level (atomic retention)
        def _bc(x):
            return x.unsqueeze(3).expand(-1, -1, -1, C).reshape(
                bsz, num_kv_heads, padded_len)[:, :, :k_context]

        context_scores = _bc(super_A) + self.alpha * _bc(super_propagated)

        # Protect observation window and sink tokens
        scores = F.pad(context_scores, (0, self.window_size), value=float('inf'))
        if self.n_sink > 0:
            scores[:, :, :self.n_sink] = float('inf')

        return scores


@dataclass
class WhaleKVMultiTurnPress(ScorerPress):
    """
    WhaleKV (Multi-Turn, Fixed-β): ASG + QSP + Global Topological Anchoring.

    Extends WhaleKVPress with Global Topological Anchoring (GTA) for multi-turn
    robustness. The centrality weight β is a fixed hyperparameter.

    GTA computes query-agnostic degree centrality C_norm ∈ [0, 1] for each super-node
    from the full (pre-KNN) similarity graph and scales it to the per-head attention
    range to prevent centrality from overwhelming query-driven signals.

    Score Fusion:
        Score_super = A_super + α·A_prop + β·C_aligned
        where C_aligned = C_norm × max(A_super) ensures the centrality bonus
        never exceeds β × max_attn.

    Parameters
    ----------
    compression_ratio : float, default=0.0
    window_size : int, default=64
    kernel_size : int, default=5
    micro_chunk_size : int, default=4
    top_k_edges : int or None, default=16
    similarity_threshold : float or None, default=None
    alpha : float, default=1.0
        QSP propagation weight.
    beta : float, default=0.3
        GTA centrality weight. After dimension alignment, beta is a true relative
        fraction of max attention (beta=0.5 → centrality bonus ≤ 50% × max_attn).
    n_sink : int, default=4
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5
    micro_chunk_size: int = 4
    top_k_edges: Optional[int] = 16
    similarity_threshold: Optional[float] = None
    alpha: float = 1.0
    beta: float = 0.3
    n_sink: int = 4

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_kv_groups = module.config.num_attention_heads // num_kv_heads
        k_context = k_len - self.window_size
        assert k_len > self.window_size

        # ── Attention Assessment and Initialization ───────────────────────────────
        if attentions is not None:
            attn_weights = attentions[..., -self.window_size:, :-self.window_size]
        else:
            attn_weights = SnapKVPress.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )
        attn_scores = attn_weights.mean(dim=-2)
        attn_scores = F.avg_pool1d(attn_scores, kernel_size=self.kernel_size,
                                   padding=self.kernel_size // 2, stride=1)
        attn_scores = attn_scores.view(bsz, num_kv_heads, num_kv_groups, k_context).mean(dim=2)

        K_ctx = keys[:, :, :k_context, :]
        C = self.micro_chunk_size
        remainder = k_context % C
        pad_len = (C - remainder) if remainder > 0 else 0
        padded_len = k_context + pad_len
        M = padded_len // C

        K_ctx_padded = F.pad(K_ctx, (0, 0, 0, pad_len)) if pad_len > 0 else K_ctx
        attn_padded  = F.pad(attn_scores, (0, pad_len))  if pad_len > 0 else attn_scores

        # ── ASG: Super-node Construction ──────────────────────────────────────────
        super_K = K_ctx_padded.view(bsz, num_kv_heads, M, C, head_dim).mean(dim=3)
        super_A = attn_padded.view(bsz, num_kv_heads, M, C).max(dim=3)[0]

        # ── QSP: Similarity Graph ─────────────────────────────────────────────────
        super_K_norm = F.normalize(super_K, p=2, dim=-1)
        macro_sim = torch.matmul(super_K_norm, super_K_norm.transpose(-1, -2)).clamp(min=0)
        diag = torch.eye(M, device=keys.device, dtype=torch.bool)
        macro_sim = macro_sim.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)

        # ── GTA: Degree centrality from the full graph (pre-KNN) (eq. 8-9) ────────
        # C_raw = sum of similarity weights; C_norm = C_raw / max(C_raw)
        global_centrality_raw = macro_sim.sum(dim=-1)
        centrality_max = global_centrality_raw.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        C_norm = global_centrality_raw / centrality_max  # [0, 1]

        # Dimension alignment: scale C_norm to [0, max_attn] so beta is meaningful
        attn_scale = super_A.max(dim=-1, keepdim=True)[0].clamp(min=1e-9)
        C_aligned = C_norm * attn_scale

        # ── QSP: KNN sparsification + propagation ────────────────────────────────
        if self.top_k_edges is not None and 0 < self.top_k_edges < M:
            topk_vals, _ = torch.topk(macro_sim, k=self.top_k_edges, dim=-1)
            macro_sim = macro_sim.masked_fill(macro_sim < topk_vals[..., -1:], 0.0)
        if self.similarity_threshold is not None:
            macro_sim = macro_sim.masked_fill(macro_sim < self.similarity_threshold, 0.0)

        row_sum = macro_sim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        W = macro_sim / row_sum
        super_propagated = torch.bmm(
            W.view(bsz * num_kv_heads, M, M),
            super_A.view(bsz * num_kv_heads, M, 1),
        ).squeeze(-1).view(bsz, num_kv_heads, M)

        def _bc(x):
            return x.unsqueeze(3).expand(-1, -1, -1, C).reshape(
                bsz, num_kv_heads, padded_len)[:, :, :k_context]

        # Score_super = A_super + α·A_prop + β·C_aligned
        context_scores = _bc(super_A) + self.alpha * _bc(super_propagated) + self.beta * _bc(C_aligned)

        scores = F.pad(context_scores, (0, self.window_size), value=float('inf'))
        if self.n_sink > 0:
            scores[:, :, :self.n_sink] = float('inf')
        return scores


@dataclass
class WhaleKVAdaptivePress(ScorerPress):
    """
    WhaleKV (Multi-Turn, Adaptive): ASG + QSP + Scenario-Aware Adaptive GTA.

    Implements the full WhaleKV eq. 12 with entropy-driven adaptive calibration
    as described in Section 4.3 of the paper.

    The key innovation over WhaleKVMultiTurnPress is that the GTA weight is NOT
    a fixed hyperparameter β but is automatically computed from the Shannon entropy
    of the current query's attention distribution:

        H  = -Σ A_super[j] · log(A_super[j])         (eq. 10)
        Φ  = H / log(M)                               (eq. 11, normalized flatness)
        Score_super = A_super + A_prop + Φ · C_norm   (eq. 12)

    Physical interpretation:
    - Low entropy (Φ → 0): concentrated attention → retrieval task → GTA deactivates,
      preserving localized precision without introducing structural noise.
    - High entropy (Φ → 1): diffuse attention → cross-domain multi-turn shift →
      GTA fully activates, anchoring global semantic topology against attention rot.

    This makes WhaleKVAdaptivePress a zero-hyperparameter multi-turn method:
    the GTA weight self-calibrates to the underlying reasoning regime automatically.

    Parameters
    ----------
    compression_ratio : float, default=0.0
    window_size : int, default=64
    kernel_size : int, default=5
    micro_chunk_size : int, default=4
    top_k_edges : int or None, default=16
    similarity_threshold : float or None, default=None
    alpha : float, default=1.0
        QSP propagation weight (fixed; the GTA weight adapts via Φ).
    n_sink : int, default=4
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5
    micro_chunk_size: int = 4
    top_k_edges: Optional[int] = 16
    similarity_threshold: Optional[float] = None
    alpha: float = 1.0
    n_sink: int = 4

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_kv_groups = module.config.num_attention_heads // num_kv_heads
        k_context = k_len - self.window_size
        assert k_len > self.window_size

        # ── Attention Assessment and Initialization ───────────────────────────────
        if attentions is not None:
            attn_weights = attentions[..., -self.window_size:, :-self.window_size]
        else:
            attn_weights = SnapKVPress.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )
        attn_scores = attn_weights.mean(dim=-2)
        attn_scores = F.avg_pool1d(attn_scores, kernel_size=self.kernel_size,
                                   padding=self.kernel_size // 2, stride=1)
        attn_scores = attn_scores.view(bsz, num_kv_heads, num_kv_groups, k_context).mean(dim=2)

        K_ctx = keys[:, :, :k_context, :]
        C = self.micro_chunk_size
        remainder = k_context % C
        pad_len = (C - remainder) if remainder > 0 else 0
        padded_len = k_context + pad_len
        M = padded_len // C

        K_ctx_padded = F.pad(K_ctx, (0, 0, 0, pad_len)) if pad_len > 0 else K_ctx
        attn_padded  = F.pad(attn_scores, (0, pad_len))  if pad_len > 0 else attn_scores

        # ── ASG: Super-node Construction ──────────────────────────────────────────
        super_K = K_ctx_padded.view(bsz, num_kv_heads, M, C, head_dim).mean(dim=3)
        super_A = attn_padded.view(bsz, num_kv_heads, M, C).max(dim=3)[0]

        # ── QSP: Similarity Graph ─────────────────────────────────────────────────
        super_K_norm = F.normalize(super_K, p=2, dim=-1)
        macro_sim = torch.matmul(super_K_norm, super_K_norm.transpose(-1, -2)).clamp(min=0)
        diag = torch.eye(M, device=keys.device, dtype=torch.bool)
        macro_sim = macro_sim.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)

        # ── GTA: Degree centrality C_norm from the full graph (eq. 8-9) ──────────
        global_centrality_raw = macro_sim.sum(dim=-1)   # C_raw
        centrality_max = global_centrality_raw.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        C_norm = global_centrality_raw / centrality_max  # C_norm ∈ [0, 1]

        # ── GTA: Entropy-driven Scenario Flatness Coefficient Φ (eq. 10-11) ──────
        # Treat A_super as a probability distribution over M super-nodes
        A_prob = super_A / super_A.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        H = -(A_prob * (A_prob + 1e-9).log()).sum(dim=-1, keepdim=True)   # [bsz, heads, 1]
        H_max = torch.tensor(M, dtype=H.dtype, device=H.device).log()
        Phi = (H / H_max).clamp(0.0, 1.0)   # Φ ∈ [0, 1]

        # ── QSP: KNN sparsification + propagation ────────────────────────────────
        if self.top_k_edges is not None and 0 < self.top_k_edges < M:
            topk_vals, _ = torch.topk(macro_sim, k=self.top_k_edges, dim=-1)
            macro_sim = macro_sim.masked_fill(macro_sim < topk_vals[..., -1:], 0.0)
        if self.similarity_threshold is not None:
            macro_sim = macro_sim.masked_fill(macro_sim < self.similarity_threshold, 0.0)

        row_sum = macro_sim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        W = macro_sim / row_sum
        super_propagated = torch.bmm(
            W.view(bsz * num_kv_heads, M, M),
            super_A.view(bsz * num_kv_heads, M, 1),
        ).squeeze(-1).view(bsz, num_kv_heads, M)

        def _bc(x):
            return x.unsqueeze(3).expand(-1, -1, -1, C).reshape(
                bsz, num_kv_heads, padded_len)[:, :, :k_context]

        # ── Score Fusion (eq. 12): adaptive GTA weight Φ ─────────────────────────
        # Score_super = A_super + A_prop + Φ · C_norm
        # Φ is per-head [bsz, heads, 1], broadcast automatically
        context_scores = _bc(super_A) + self.alpha * _bc(super_propagated) + Phi * _bc(C_norm)

        scores = F.pad(context_scores, (0, self.window_size), value=float('inf'))
        if self.n_sink > 0:
            scores[:, :, :self.n_sink] = float('inf')
        return scores
