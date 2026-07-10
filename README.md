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

Input can be JSONL or CSV. JSONL lines must contain a text field and may contain
any metadata you want copied to the output.

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

### Run the existing discharge and nephrology reports

Four convenience scripts point to the current `/orange` report files and write
artifacts under the Git-ignored `outputs/` directory:

```bash
# vLLM server must already be running
scripts/run_discharge_structured.sh
scripts/run_nephrology_structured.sh

# Dense vectors on GPU 0
scripts/run_discharge_embeddings.sh
scripts/run_nephrology_embeddings.sh
```

Use `--limit` for a smoke test before a full run:

```bash
scripts/run_discharge_structured.sh --limit 2 --no-resume
scripts/run_nephrology_embeddings.sh --limit 2
```

Configuration is available through environment variables without editing a
script. For example:

```bash
WORKERS=16 OUTPUT=/orange/path/to/discharge.jsonl \
  scripts/run_discharge_structured.sh

MODEL=UFNLP/gatortron-base BATCH_SIZE=16 CUDA_VISIBLE_DEVICES=0 \
  scripts/run_nephrology_embeddings.sh
```

The defaults are:

| Script group | Input | Output |
|---|---|---|
| Discharge structured | `/orange/pinaki.sarder/singletarya/mmai/text_fe/discharge_summaries.csv` | `outputs/discharge_structured.jsonl` |
| Nephrology structured | `/orange/pinaki.sarder/kirill.luka/mmai/text_fe/nephrology_consults.csv` | `outputs/nephrology_structured.jsonl` |
| Discharge embeddings | discharge CSV above | `outputs/discharge_embeddings.npz` |
| Nephrology embeddings | nephrology CSV above | `outputs/nephrology_embeddings.npz` |

The CSV reader streams rows, and structured requests are kept in a bounded
queue, so the 5 GB discharge CSV is not loaded into memory. The embedding
pipeline also streams text batches; only the resulting vector matrix is retained
until the final compressed `.npz` is written.

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
