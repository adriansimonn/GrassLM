<p align="center">
  <img src="images/icon.png" alt="GrassLM" width="128" />
</p>

<h1 align="center">GrassLM</h1>

<p align="center">
  An attention-free language model built on Grassmann flows
  <br />
  <a href="https://arxiv.org/abs/2512.19428">Paper</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="https://github.com/adriansimonn/GrassLM/releases/download/v1.0.0/GrassLM-1.0.dmg">Download (MacOS)</a>
</p>

---

GrassLM is a language model that replaces self-attention entirely with a geometric mixing mechanism based on **Grassmann manifolds**. Instead of computing dense L×L attention matrices, it encodes local token relationships as points on Gr(2, r) via Plücker coordinates — achieving competitive perplexity with **linear scaling** in sequence length.

Based on *"Attention Is Not What You Need: Grassmann Flows as an Attention-Free Alternative for Sequence Modeling"* ([Zhang Chong, 2025](https://arxiv.org/abs/2512.19428)).

#### Note: The models implemented in the application are very small and produce outputs much worse than foundational LLMs. This project is a research implementation.

## Architecture

Standard transformers use self-attention as a form of **tensor lifting**: each token is lifted into a high-dimensional space of pairwise interactions via an L×L compatibility matrix. This is expressive but analytically opaque — the degrees of freedom grow so large that the model's behavior becomes nearly impossible to describe with explicit invariants.

GrassLM takes a different approach. Instead of lifting to an attention tensor, each layer applies a **Causal Grassmann mixing block**:

```
                    ┌───────────────────────────────────────────────────┐
                    │               Causal Grassmann Layer              │
                    │                                                   │
  h_t ∈ R^d ──────► │  1. Linear reduction  →  z_t ∈ R^r     (r ≪ d)    │
                    │  2. Local pairing     →  (z_t, z_{t+Δ}) for Δ ∈ W │
                    │  3. Plücker encoding  →  p_t ∈ R^(r choose 2).    │
                    │  4. Projection        →  g_t ∈ R^d                │
                    │  5. Gated fusion      →  α ⊙ h_t + (1-α) ⊙ g_t    │
                    │  6. Feed-forward      →  h'_t ∈ R^d               │
                    │                                                   │
                    └───────────────────────────────────────────────────┘
```

**How it works:**

1. **Linear reduction.** Each hidden state h_t is projected from R^d to a low-dimensional R^r (e.g. r = 32).

2. **Multi-scale local pairing.** For a set of causal window offsets W = {1, 2, 4, 8, 12, 16}, each token is paired with future neighbors at multiple scales: (z_t, z_{t+Δ}).

3. **Grassmann / Plücker encoding.** Each pair spans a 2D subspace on the Grassmann manifold Gr(2, r). This subspace is embedded into projective space via the Plücker map — computing all pairwise determinants p_ij = z_{t,i} · z_{t+Δ,j} − z_{t,j} · z_{t+Δ,i} to produce a coordinate vector in R^(r choose 2).

4. **Projection back to model space.** The Plücker features are linearly projected back to R^d and aggregated across offsets.

5. **Gated fusion.** The original hidden state and the Grassmann features are fused via a learned sigmoid gate, followed by layer normalization.

6. **Feed-forward block.** A standard position-wise FFN (GELU activation, 4× expansion) with residual connection.

**Key properties:**

| | Self-Attention | Grassmann Mixing |
|---|---|---|
| **Complexity** | O(L²d) | O(Ld²) — linear in L |
| **Core operation** | Dense L×L weight matrix | Plücker coordinates on Gr(2, r) |
| **Interpretability** | Opaque tensor lifting | Finite-dimensional manifold with explicit invariants |
| **Information flow** | Global pairwise weights | Multi-scale local windows across layers |

The mixing mechanism scales **linearly** in sequence length for fixed rank r and window count, compared to the **quadratic** cost of full self-attention.

## Models

| Model | Params | Layers | d_model | d_reduce | Context | Dataset |
|---|---|---|---|---|---|---|
| **GrassLM-10M** | 12.6M | 6 | 256 | 32 | 128 | WikiText-2 |
| **GrassLM-100M** | 101.7M | 12 | 640 | 80 | 256 | WikiText-103 |

## Project Structure

```
python/          PyTorch training, evaluation, and export
cpp/             C++ inference engine (Eigen + Apple Accelerate)
app/             SwiftUI macOS desktop application
models/          Model registry with configs and checkpoints
docs/            Research paper
```

## Getting Started

### Training (Python)

```bash
pip install -r python/requirements.txt
python python/train.py --model GrassLM-10M
```

### Inference (C++)

```bash
cd cpp && mkdir build && cd build
cmake .. && make
./grasslm_cli
```

### macOS App

Open `app/GrassLM.xcodeproj` in Xcode and build.