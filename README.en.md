# memory-refiner

> A memory refiner for LLM agents: compress long conversations into **auditable fact nodes**,
> verify them with **probe questions**, and keep a **version chain** so facts can change without being erased.

- No self-grading: the score is computed by scripts, the model only writes.
- No deletion: raw text, node and metadata are archived on every compression.
- Facts are versioned (`superseded`), never overwritten.

## Why

Most context-compression tools answer one question: *is the summary short enough?*
The question that matters is: **can the next session still answer what you said today?**

This project treats memory as something that can be **examined**:

1. compress → node (facts with provenance + confidence)
2. score → script-side recall / precision / efficiency
3. probe → generate questions from the raw text, answer them using **only the node**
4. merge → duplicate / update / contradiction, with `superseded_by` links

## Quick start

```bash
python refiner.py compress --raw dialog.md --probe
python refiner.py score --raw dialog.md --node archive/2026-03-02/xxx_node.json
python refiner.py recall --q "delivery schedule" --inject
python refiner.py history --id 3a8f...
```

## Scoring

```
score = 100 × (0.5×recall + 0.3×precision + 0.2×efficiency)
```

- `recall`: topic terms auto-extracted from the raw text (repeated 2–4 char CJK n-grams), how many survive in the node
- `precision`: fraction of node clauses that can be grounded in the raw text
- `efficiency`: `min(1, compression_ratio / 0.70)`

With `--probe`, a second score is reported: `0.6 × script + 0.4 × probe`.

**Known limits**: `recall` is a proxy — cheap, stable, explainable, but blind to paraphrase.
The probe is the real metric; it costs two extra model calls.

## Version chain

Facts are never deleted. Updates mark the old row `superseded` with `valid_to` and `superseded_by`.
If the relation cannot be decided, **both rows stay** — a wrong merge is more expensive than a duplicate.

## Layout

```
refiner.py            single file, stdlib only
tests/                version chain + CLI smoke
examples/             de-identified sample conversation and node
```

## License

MIT
