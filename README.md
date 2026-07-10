# Medical text extraction

Two small, independent inference pipelines:

1. **Structured extraction:** clinical text → schema-validated JSONL through an
   OpenAI-compatible vLLM server.
2. **Dense embeddings:** clinical text → NumPy vectors through a Hugging Face
   encoder or base transformer model.

This repository deliberately contains no Ollama integration, prediction models,
XGBoost, regression, EHR fusion, clustering, or report-specific batch classes.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[embeddings]'
```

Run a vLLM server separately. For example:

```bash
vllm serve openai/gpt-oss-20b --port 8000 --max-model-len 32768
```

## 1. Structured JSON extraction

Input is JSONL. Each line must contain a text field and may contain any metadata
you want copied to the output.

```json
{"note_id":"n1","text":"Discharge summary: ..."}
```

```bash
extract-structured \
  --input notes.jsonl \
  --output outputs/discharge.jsonl \
  --schema schemas/discharge_renal.json \
  --model openai/gpt-oss-20b \
  --workers 8
```

The default server is `http://127.0.0.1:8000/v1`. Override it with
`--base-url`. Each successful output line contains the input metadata plus an
`extraction` object. Failed records contain an `error` and can be retried by
rerunning with `--resume` (the default).

To add another report type, create a standard JSON Schema. No Python changes
are needed. `schemas/nephrology_renal.json` is a second example.

## 2. Dense embedding extraction

```bash
extract-embeddings \
  --input notes.jsonl \
  --output outputs/embeddings.npz \
  --model NeuML/bioclinical-modernbert-base-embeddings \
  --batch-size 8
```

The `.npz` contains:

- `embeddings`: float32 matrix shaped `(documents, dimensions)`
- `ids`: document identifiers

The default is attention-mask-aware mean pooling. Use `--pooling cls` if the
selected model was trained for CLS pooling. Long documents can be represented
by averaging overlapping chunk vectors with `--chunk-tokens` and
`--chunk-overlap`.

Any text model supported by `transformers.AutoModel` can be selected, including
GATORTron, ModernBERT, or the base transformer behind a MedGemma checkpoint.
For semantic retrieval or similarity, prefer a model specifically trained to
produce embeddings; a generative checkpoint's hidden states are vectors but are
not automatically good semantic embeddings.

## Input and privacy

Both commands read local JSONL and write local artifacts. They do not send data
anywhere except the vLLM endpoint you configure. Do not commit patient text,
outputs, credentials, or model-cache files.

## Lightweight checks

```bash
python -m compileall -q src tests
pytest -q
```

