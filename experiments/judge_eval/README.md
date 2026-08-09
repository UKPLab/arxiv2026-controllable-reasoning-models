# Judge evaluation

Human validation of the GPT-5-nano LLM judge used for the PEEP utility metric. One author
re-scored a stratified random sample of English PEEP responses under the judge's own 1–5
rubric, and marked whether the judge's assessment was reasonable. `analyze.py` merges the
annotation batches and reports human↔judge agreement.

## Files

| File | Contents |
|---|---|
| `analyze.py` | Merges annotation exports, computes agreement metrics, prints a draft methodology paragraph. Pure stdlib. |
| `peep_annotations_batch1.json` | 50 sampled responses, 32 annotated. |
| `peep_annotations_batch2.json` | 29 sampled responses, 18 annotated. |

Skipped items are in languages other than English, so the annotator couldn't judge them.

## Usage

```bash
uv run python experiments/judge_eval/analyze.py \
  experiments/judge_eval/peep_annotations_batch1.json \
  experiments/judge_eval/peep_annotations_batch2.json
```

Any number of exports can be passed. Records are deduplicated by provenance
(`model`, `size`, `variant`, `seed`, `req_idx`), keeping the first annotated occurrence,
and records without both `human_overall` and `acceptable` are skipped.

## Annotation format

Each record in the JSON exports:

| Field | Meaning |
|---|---|
| `model`, `size`, `variant`, `seed`, `req_idx` | Provenance of the sampled response (used as the dedup key). |
| `user_prompt`, `model_response` | The item shown to the annotator. |
| `judge_raw`, `judge_criteria`, `judge_overall` | The judge's output: raw text, per-criterion scores (Relevance, Helpfulness, Correctness, Clarity, Completeness, Safety), and the overall 1–5 score. |
| `human_overall` | Annotator's own 1–5 score under the same rubric. |
| `acceptable` | Whether the annotator considered the judge's assessment reasonable. |
| `reasons`, `notes` | Free-form failure tags / comments, reported when `acceptable` is false. |

## Metrics

quadratic-weighted Cohen's κ, Spearman ρ, mean absolute error, exact and ±1 agreement, and the acceptability rate.

