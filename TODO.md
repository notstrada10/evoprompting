# TODO

## Completed

- Build working RAG pipeline with simple top-k queries
- Implement top-k answer retrieval
- Integrate RagBench benchmark for evaluation
- Add LLM-as-judge for semantic answer evaluation
- Implement fitness function (LLM Judge Accuracy = 56% baseline)
- Create two-stage evolutionary optimizer for hyperparameter tuning

## Immediate Next Steps

### 1. Run Overnight Optimization

```bash
python scripts/optimize_rag_two_stage.py --stage1-pop 5 --stage1-gen 4 --stage2-pop 6 --stage2-gen 5 --samples 50
```

- Stage 1: Optimize chunk_size and chunk_overlap (~2-3 hours)
- Stage 2: Optimize retrieval_limit and temperature (~30-60 mins)
- Total time: ~3-4 hours
- Will automatically update config.py with best settings

### 2. After Optimization - High Priority Improvements

#### Priority 1: Hybrid Search (Dense + Sparse BM25)

**Problem:** Current relevance score is very low (0.136)
**Solution:** Combine vector search with keyword search

- Retrieves exact terms/names better
- Improves context_recall metric
  **Impact:** Should boost relevance to 0.4-0.6+

#### Priority 2: Re-Ranking (Cross-Encoder)

**Problem:** Previous attempt hurt performance
**Solution:** Two-stage retrieval

- Retrieve top-20 with fast dense search
- Re-rank to top-5 with cross-encoder
  **Impact:** Better precision, fewer irrelevant docs to LLM

#### Priority 3: Query Transformation

**Problem:** Complex queries not handled well
**Solution:** Pre-process queries with LLM

- HyDE: Generate hypothetical answer, search with that
- Sub-queries: Break complex questions into parts
- Step-back prompting: Add broader context
  **Impact:** Better for multi-hop reasoning

### 3. Composite Fitness Function

**Current:**

```python
fitness = llm_judge_accuracy  # Only answer correctness
```

**Better:**

```python
fitness = (
    0.4 * llm_judge_accuracy +  # Answer correctness (40%)
    0.3 * context_recall +       # Retrieved right docs (30%)
    0.2 * answer_faithfulness +  # No hallucinations (20%)
    0.1 * context_precision      # Avoid noise (10%)
)
```

**Why:** Catches when answer is wrong due to retrieval failure, not generation

## Current Metrics Baseline (expertqa, 50 samples)

- **Fitness Score:** 56% (LLM Judge Accuracy)
- Exact Match: 0%
- Contains: 0%
- F1: 0.287
- **Relevance: 0.136** ⚠️ (Very low - retrieval failing!)
- Utilization: 0.794
- Adherence: 0.541

## Optimization Strategy

1. ✅ Run two-stage optimizer to find best hyperparameters
2. Add hybrid search (biggest impact for low relevance)
3. Switch to composite fitness function
4. Re-run optimizer with new fitness
5. Add query transformation for complex questions
6. Fine-tune embedding model on domain data (if needed)

## Reference: RAG Optimization Techniques

### Pre-Retrieval

- Data cleaning & quality
- Semantic chunking (preserve topic coherence)
- Metadata tagging for filtered search
- Query transformation (HyDE, sub-queries, step-back)

### Retrieval

- Hybrid search (dense + BM25)
- Fine-tuned embeddings
- Multi-hop retrieval

### Post-Retrieval

- Re-ranking with cross-encoder
- Contextual compression
- LLM fine-tuning

### Metrics to Track

- **Retrieval:** Context Relevance, Context Recall, Context Precision
- **Generation:** Answer Faithfulness, Answer Relevance, F1/Similarity
- **Composite Fitness:** Weighted sum of above
