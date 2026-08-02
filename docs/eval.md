# Evaluation plan

The repository includes a small smoke-test evaluation harness. Before using project metrics on a resume, expand it to at least 20 manually reviewed cases.

Recommended case groups:

- single-paper method extraction;
- exact-number questions where the answer exists;
- unanswerable questions where the agent must abstain;
- multi-paper comparison;
- weak local evidence that should trigger arXiv;
- arXiv timeout or empty result;
- long document and document-filter tests;
- citation-support checks.

Recommended metrics:

- `citation_precision`: whether each cited chunk supports the sentence;
- `answer_coverage`: whether requested dimensions are covered;
- `abstention_accuracy`: whether unsupported questions are refused;
- `retrieval_recall@k` and `paper_relevance@k`;
- latency, tool calls, context size and model cost.

Run:

```bash
python eval/run_eval.py
```

Do not publish a success rate until the cases have human-written expected answers and manual citation labels.
