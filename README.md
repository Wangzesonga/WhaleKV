# WhaleKV: Navigating the Deep Context Ocean via Topological KV Cache Compression

A training-free KV cache compression framework that replaces local scoring heuristics
with global topological modeling. WhaleKV addresses two fundamental failure modes of
existing methods:

- **Semantic Blindness** — query-aware point-wise methods miss implicitly linked evidence.
- **Temporal Amnesia** — query drift in multi-turn sessions evicts early critical context.

## Architecture

WhaleKV consists of three tightly integrated components:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Single-Turn                                                               │
│  ASG ──→ QSP ──→ Score_super = A_super + A_prop                            │
│                                                                            │
│  Multi-Turn (Adaptive)                                                     │
│  ASG ──→ QSP ──→ GTA(Φ) ──→ Score_super = A_super + A_prop  + Φ · C_norm   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

| Component | Full Name | Role |
|-----------|-----------|------|
| **ASG** | Atomic Super-node Graph | Groups sub-word tokens into atomic chunks; reduces complexity O(L²) → O((L/C)²) |
| **QSP** | Query-driven Semantic Propagation | One-step graph diffusion recovers implicitly linked evidence |
| **GTA** | Global Topological Anchoring | Entropy-driven adaptive centrality bonus prevents temporal amnesia |

## Installation

```bash
cd WhaleKV
pip install -e .
```

## Quick Start

```python
import whalekv
from whalekv import WhaleKVPress, WhaleKVMultiTurnPress, WhaleKVAdaptivePress
```

### Single-Turn: WhaleKVPress (ASG + QSP)

```python
from whalekv import WhaleKVPress
from transformers import pipeline

press = WhaleKVPress(compression_ratio=0.5)

pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="cuda",
)

with press(pipe.model):
    output = pipe("Summarize the following document: ...")
```

### Multi-Turn (Fixed-β): WhaleKVMultiTurnPress (ASG + QSP + GTA, fixed β)

```python
from whalekv import WhaleKVMultiTurnPress

press = WhaleKVMultiTurnPress(
    compression_ratio=0.5,
    beta=0.3,   # fixed GTA weight
)
```

### Multi-Turn (Adaptive Φ): WhaleKVAdaptivePress (ASG + QSP + Adaptive GTA)

Implements eq. 12 of the paper with entropy-driven scenario calibration.
No β hyperparameter — the GTA weight is computed automatically.

```python
from whalekv import WhaleKVAdaptivePress

press = WhaleKVAdaptivePress(compression_ratio=0.5)
# Φ self-calibrates: low-entropy (retrieval) → 0; high-entropy (drift) → 1
```

## Method Details

### 1. Atomic Super-node Graph (ASG)

Tokens are partitioned into non-overlapping micro-chunks of size `C` (default 4).
Each chunk becomes a *super-node* with:
- **Mean-pooled** key features — semantic representation (eq. 2)
- **Max-pooled** attention score — importance signal (eq. 3)

Treating chunks as atomic units prevents *Token Tearing* (inconsistent retention of
multi-token entities such as UUIDs, proper nouns, or code snippets).

### 2. Query-driven Semantic Propagation (QSP)

A cosine similarity graph is constructed over super-node embeddings and sparsified
via KNN (top-k edges, default k=16). Attention energy propagates via one-step
random-walk diffusion:

```
A_prop = W · A_super                          (eq. 6)
Score_super = A_super + A_prop                (eq. 7)
```

This allows structurally connected but weakly-attended tokens to recover retention
mass, overcoming the semantic blindness of direct query matching.

### 3. Global Topological Anchoring (GTA)

Degree centrality is computed from the full (pre-KNN) similarity graph and
max-normalized to [0, 1]:

```
C_raw[i]  = Σ_j S[i,j]                       (eq. 8)
C_norm[i] = C_raw[i] / max(C_raw)            (eq. 9)
```

#### Fixed-β variant (WhaleKVMultiTurnPress)

```
Score_super = A_super + α·A_prop + β·C_aligned
```

`C_aligned` scales `C_norm` to the per-head attention range, ensuring β is a true
relative fraction of the maximum attention score. Suitable when a stable task-specific
β can be tuned.

#### Adaptive-Φ variant (WhaleKVAdaptivePress) — eq. 10–12

Shannon entropy of `A_super` measures attention flatness:

```
H  = -Σ_j A_super[j] · log(A_super[j])      (eq. 10)
Φ  = H / log(M)                              (eq. 11)  ∈ [0, 1]
Score_super = A_super + A_prop + Φ · C_norm  (eq. 12)
```

- **Φ → 0** (concentrated / retrieval query): GTA deactivates → precise localized
  retention.
- **Φ → 1** (diffuse / cross-domain drift): GTA fully activates → global structural
  preservation against temporal amnesia.

This makes WhaleKVAdaptivePress a **zero-hyperparameter** multi-turn method.

## Bundled Baselines

For fair comparison, WhaleKV ships with implementations of the baselines evaluated
in the paper:

| Class | Paper |
|-------|-------|
| `SnapKVPress` | SnapKV  |
| `ChunkKVPress` | ChunkKV  |
| `PyramidKVPress` | PyramidKV  |
| `ExpectedAttentionPress` | Expected Attention  |

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compression_ratio` | 0.0 | Fraction of tokens to evict (0 = no compression, 0.5 = keep 50%) |
| `window_size` | 64 | Observation window size for attention computation |
| `kernel_size` | 5 | 1D avg-pool kernel for attention smoothing |
| `micro_chunk_size` | 4 | Atomic chunk size C |
| `top_k_edges` | 16 | KNN neighbors per super-node |
| `alpha` | 1.0 | QSP propagation weight |
| `beta` | 0.3 | GTA weight (WhaleKVMultiTurnPress only; adaptive Φ replaces β in WhaleKVAdaptivePress) |
| `n_sink` | 4 | Number of initial tokens always retained |

## Citation

```
WhaleKV: Navigating the Deep Context Ocean via Topological KV Cache Compression.
```
