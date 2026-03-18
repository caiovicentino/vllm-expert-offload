# MoE Expert Weight Caching

vLLM can run MoE models that exceed available GPU memory by keeping all expert
weights in CPU pinned memory and caching only the most-recently/frequently-used
experts in a fixed-size GPU scratch buffer.

This is controlled by two options:

| Option | Default | Description |
| --- | --- | --- |
| `--moe-expert-cache-size N` | `0` (disabled) | Number of expert slots to allocate in the GPU buffer per layer |
| `--moe-expert-cache-policy POLICY` | `lru` | Eviction policy: `lru`, `lfu`, `fifo`, or `slru` |

!!! note
    Expert caching requires `--enforce-eager`. CUDA graph capture is
    incompatible with the dynamic Python bookkeeping in `prepare()`.

!!! note
    Expert caching is not compatible with expert parallelism (EP > 1),
    data parallelism, or sequence parallelism.

## Quick start

```bash
# OLMoE-1B-7B: 64 experts, fits on 8 GB GPU with 16 cached per layer
vllm serve allenai/OLMoE-1B-7B-0924 \
    --moe-expert-cache-size 16 \
    --moe-expert-cache-policy lru \
    --enforce-eager
```

### Python API

`moe_expert_cache_size` is exposed as a direct `LLM` constructor parameter:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="allenai/OLMoE-1B-7B-0924",
    moe_expert_cache_size=16,
    enforce_eager=True,
)
```

To set the eviction policy via the Python API, pass it through `EngineArgs`:

```python
from vllm.engine.arg_utils import EngineArgs

args = EngineArgs(
    model="allenai/OLMoE-1B-7B-0924",
    moe_expert_cache_size=16,
    moe_expert_cache_policy="slru",
    enforce_eager=True,
)
llm = LLM.from_engine_args(args)
```

## How it works

```text
Decode (unique experts ≤ capacity) — GPU fast path:
  topk_ids → prepare():
    hit  → refresh policy state   (O(1))
    miss → evict, H2D copy, update mapping[expert] = slot
  → fused_experts(buf_w13, buf_w2, remapped_ids)

Prefill (unique experts > capacity) — CPU fallback:
  → _moe_forward_cpu(): F.linear() on CPU pinned weights
  → correct but slower (~ms per forward); decode resumes GPU path
```

A persistent `_mapping` tensor (`int32`, GPU) holds the `expert_id → slot`
mapping. It is updated in-place for misses and used for a vectorized remap —
no CPU tensor build or H2D transfer on the hot path.

## Eviction policies

All policies implement the `ExpertCachePolicy` ABC
(`vllm/model_executor/layers/fused_moe/cache_policy.py`).
LRU, LFU, and FIFO are thin wrappers around
[cachetools](https://cachetools.readthedocs.io/).
SLRU is a pure-Python two-tier implementation.

| Policy | Eviction rule | O(1)? | Best for |
| --- | --- | --- | --- |
| `lru` (default) | least recently used | ✅ | decode-heavy, temporal locality |
| `lfu` | least frequently used | ✅ (heap) | highly skewed routing — same experts always hot |
| `fifo` | insertion order | ✅ | uniform routing, deterministic eviction |
| `slru` | two-tier: probationary → protected | ✅ | mixed prefill+decode; protects hot experts from burst eviction |

### SLRU details

New experts enter the *probationary* tier (20 % of capacity). A second
access promotes them to the *protected* tier (80 %). Eviction always
targets the probationary tier first, so experts that are accessed more
than once are shielded from one-off prefill bursts. This typically
improves hit rates by 10–30 % over pure LRU on real-world serving
workloads.

## Observability

### DEBUG-level hit/miss log

Set `VLLM_LOGGING_LEVEL=DEBUG` to get a per-layer hit/miss report every
60 seconds:

```text
DEBUG vllm...lru_cache: Expert cache: 1234 hits, 56 misses (95.7% hit rate)
```

### INFO-level per-layer report

Pass `--enable-logging-iteration-details` (or set
`observability_config.enable_logging_iteration_details=True`) to get a
per-layer INFO log every 300 seconds:

```text
INFO vllm...layer: model.layers.0.mlp expert cache: 1234 hits, 56 misses (95.7% hit rate)
```

## Sizing guidance

Set `--moe-expert-cache-size` to the number of experts that must fit on
GPU simultaneously per layer. For a model with `E` experts and `top_k`
routing:

- **Minimum useful**: `top_k` (one slot per active expert per token, no
  eviction during decode)
- **Typical decode**: `2 × top_k` – `4 × top_k` gives headroom for
  locality without wasting VRAM
- **Maximum** (no-op): `E` (all experts on GPU, equivalent to normal mode)

During prefill, `unique(topk_ids)` often exceeds capacity — this triggers
the CPU fallback automatically.

## GPU memory note

Expert weights in CPU pinned memory are invisible to the `--gpu-memory-utilization`
profiler. The profiler will underestimate available KV cache headroom by the
expert weight footprint (a safe margin, not a hazard), but exact
`gpu-memory-utilization`-based sizing will be off.

## Tests

```bash
# Unit tests: ExpertLRUCache (18 tests)
pytest tests/kernels/moe/test_expert_lru_cache.py -v

# Unit tests: cache policies (LRU/LFU/FIFO/SLRU)
pytest tests/model_executor/layers/fused_moe/test_cache_policy.py -v

# End-to-end correctness (compare_two_settings with/without cache)
pytest tests/basic_correctness/test_moe_expert_cache.py -v -s
```
