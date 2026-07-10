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
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` installs both pipeline runtimes and vLLM. If vLLM is
managed in a separate environment, install this package with
`pip install -e '.[embeddings]'` in the client environment instead.

## Start vLLM on one GPU

Run the server in its own terminal:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
  --served-model-name openai/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80
```

Confirm it is ready:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

On HiPerGator, an error mentioning `CXXABI_1.3.15` means the server selected
the system C++ runtime. Scope the active conda environment's runtime to the
vLLM command:

```bash
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \
  vllm serve openai/gpt-oss-20b \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80
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

The pipeline sends the JSON Schema to vLLM as a strict structured-output
constraint and validates every response again with `jsonschema`. Schema-level
descriptions carry the clinical extraction rules, so a new task should include
clear instructions for negation, timing, missing values, and allowed inference.

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
- `metadata`: JSON containing model ID, pooling, normalization, and chunk settings

The default is attention-mask-aware mean pooling. Use `--pooling cls` if the
selected model was trained for CLS pooling. Long documents can be represented
by averaging overlapping chunk vectors with `--chunk-tokens` and
`--chunk-overlap`. `--chunk-tokens` must fit inside `--max-length` after the
model's special tokens are added.

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

With a local vLLM server running, execute the opt-in semantic checks:

```bash
RUN_VLLM_INTEGRATION=1 pytest -q tests/test_vllm_integration.py
```

## Verified configuration

The structured pipeline has been exercised end to end with:

- one NVIDIA B200 GPU
- `vllm==0.19.0`
- `openai/gpt-oss-20b`
- strict JSON-schema output for both included schemas
- concurrent requests (`--workers 2` and `--workers 3`)
- dense extraction with cached `emilyalsentzer/Bio_ClinicalBERT` (three notes,
  768 dimensions, finite unit-normalized float32 vectors)

This verifies server startup, OpenAI-compatible requests, constrained decoding,
JSON parsing, schema validation, concurrent output writing, and both clinical
task schemas. Model outputs still require task-specific evaluation on a labeled,
de-identified validation set before clinical or research use.
