# How Vector Databases Find Similarities

Here’s a clear, developer-oriented walkthrough of how a vector database finds “similar” items—and the knobs you can tune to get great results.

## What’s compared?

1. **Embeddings**
   Raw data (text, images, audio) → fixed-size numeric vectors via an embedding model. Similar meaning ⇒ nearby vectors.

2. **Similarity metrics**

   * **Cosine similarity** (angle between vectors):
     [
     \mathrm{cos_sim}(a,b)=\frac{a\cdot b}{|a||b|}
     ]
   * **Inner product / dot product**: captures alignment *and* magnitude.
   * **Euclidean (L2) distance**: straight-line distance.
     Tip: L2-normalize vectors ⇒ **cosine ≡ inner product**.

## How search works (fast path)

Brute-force is exact but O(N·d). Vector DBs use **Approximate Nearest Neighbor (ANN)** indexes for sub-linear latency with high recall.

### Indexing (offline/async)

* **HNSW** (Hierarchical Navigable Small World): graph-based; great recall/latency.
  Tunables: `M` (degree), `efConstruction`.
* **IVF** (Inverted File Lists): k-means partitions into `nlist` clusters; query probes only the closest (`nprobe`).
* **PQ/OPQ** (Product Quantization): compress vectors; big RAM win, small accuracy hit.
* Others: **ScaNN**, **Annoy**, **FLAT** (exact).

### Query phase

1. Embed the query with the same model.
2. **Coarse search**: pick promising regions (HNSW neighbors / IVF centroids).
3. **Fine search**: compute distances on candidates; optionally **rerank** top-K with exact distances or a cross-encoder.
4. Return top-K ids + scores.

## Hybrid & filtered search

* **Metadata filters**: e.g., `lang='pl' AND created_at > now()-30d`. Applied pre/during ANN via filter-aware indexes.
* **Hybrid lexical + vector**: combine BM25 with vector scores (weighted sum or learning-to-rank).
* **Reranking**: multi-stage pipelines where vectors shortlist; heavier models (cross-encoders, MMR) reorder for relevance/diversity.

## Quality, latency, and key knobs

* **Recall@K**: true-neighbor fraction vs exact search.
  Tune `efSearch` (HNSW) or `nprobe` (IVF) to hit SLA.
* **Dimensionality & model**: 384–1024d is common; consider PCA/OPQ for RAM.
* **Normalization**: L2-normalize for cosine stability.
* **Batching**: amortize overhead; vector DBs love throughput.
* **Warm vs cold**: caches help steady-state latency.

## Pitfalls

* **Embedding drift**: mixing models/versions hurts similarity. Store `model_id`; re-embed or shard per model.
* **Hubness**: some vectors become “universal neighbors” in high-D. Mitigate with normalization and MMR diversification.
* **Boilerplate dominance**: prompts with lots of template text collapse similarity—clean inputs or use hybrid scoring.
* **Score comparability**: cosine scores aren’t calibrated across corpora/models—prefer ranks over global thresholds.

## Vector + Graph (when relationships matter)

Blend semantics with structure:

* Vectors retrieve **candidate nodes** fast.
* Graph edges (citations, entities, concepts) expand/validate context.
* Example blended score:
  [
  \alpha\cdot \text{vector_sim} + \beta\cdot \text{graph_proximity} + \gamma\cdot \text{metadata_match}
  ]

## Quick reference (cheat sheet)

| Technique        | Strengths               | Key knobs                         | Notes                               |
| ---------------- | ----------------------- | --------------------------------- | ----------------------------------- |
| **HNSW**         | Top recall, low latency | `M`, `efConstruction`, `efSearch` | Great default; memory-heavier       |
| **IVF**          | Fast, filter-friendly   | `nlist`, `nprobe`                 | Good for very large corpora         |
| **IVF-PQ / OPQ** | RAM-efficient           | codebook size, sub-vectors        | Small accuracy tax, huge memory win |
| **FLAT**         | Exact                   | —                                 | Best quality, slowest at scale      |

## Minimal mental model

* **Model defines geometry**: choose domain-fit embeddings → good neighborhoods.
* **Index is an accelerator**: trade tiny recall for big wins in speed/memory.
* **Pipelines win**: ANN shortlist → filters → hybrid/rerank to balance quality and latency.
