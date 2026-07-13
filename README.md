# Medical text extraction

Two independent inference pipelines:

1. **Structured extraction:** clinical text → schema-validated JSONL through an
   OpenAI-compatible vLLM server.
2. **Dense embeddings:** clinical text → NumPy vectors through a Hugging Face
   encoder or base transformer model.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` installs both pipeline runtimes and vLLM. If vLLM is
managed in a separate environment, install only the components needed by the
current environment:

```bash
pip install -e .                 # structured-extraction client
pip install -e '.[server]'       # client plus local vLLM server
pip install -e '.[embeddings]'   # client plus embedding runtime
pip install -e '.[dev]'          # unit-test dependencies
```

## Model access

Structured extraction defaults to `google/medgemma-27b-text-it`. Before the
first run, log in to Hugging Face and accept Google's Health AI Developer
Foundations terms on the model page:

<https://huggingface.co/google/medgemma-27b-text-it>

Model weights are downloaded to the Hugging Face cache, never to this
repository.

## Start MedGemma on one GPU

Run the server in its own terminal:

```bash
GPU=0 PORT=8000 scripts/serve_medgemma.sh
```

The launcher uses one GPU, a 32K context window, deterministic vLLM generation
defaults, and XGrammar JSON constraints with unrestricted whitespace disabled.
The last setting is required to prevent constrained-generation whitespace loops.

Confirm it is ready:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

On HiPerGator, activate the intended environment before launching. If an error
mentions `CXXABI_1.3.15`, provide its Python and C++ runtime explicitly:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" GPU=0 PORT=8000 \
  scripts/serve_medgemma.sh
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
  --model google/medgemma-27b-text-it \
  --workers 8
```

The default server is `http://127.0.0.1:8000/v1`. Override it with
`--base-url`. To load balance across independent vLLM replicas, provide a
comma-separated list such as
`--base-url http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1`. Requests are
assigned round-robin while one process remains responsible for safe output
writing and resume handling. Each successful output line contains the input
metadata plus an `extraction` object. Failed records contain an `error` and can
be retried by rerunning with `--resume` (the default).

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

For both structured datasets on two GPUs, one command starts a MedGemma server
on each GPU, waits for both health checks, and distributes each dataset across
both replicas with 16 total workers. Datasets run sequentially so neither GPU
becomes idle when one source finishes earlier. Both servers stop when finished:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" scripts/run_all_structured_two_gpus.sh
```

The durable results are:

- `outputs/discharge_structured.jsonl`
- `outputs/nephrology_structured.jsonl`
- `outputs/logs/medgemma_gpu0.log`
- `outputs/logs/medgemma_gpu1.log`

All are excluded by `.gitignore`. Rerunning resumes successful records and
retries failed records. Pass `--no-resume` only when intentionally replacing an
existing extraction.

Generated results and logs exist only on the machine that ran extraction. They
are not included when this repository is cloned or shared through GitHub.

Use `--limit` for a smoke test before a full run:

```bash
scripts/run_discharge_structured.sh --limit 2 --no-resume
scripts/run_nephrology_embeddings.sh --limit 2

# Two-GPU structured smoke run (two records from each source)
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  scripts/run_all_structured_two_gpus.sh --limit 2 --no-resume
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

## Repository layout

```text
schemas/                  JSON Schemas and clinical extraction rules
scripts/                  Reproducible dataset and vLLM launchers
src/text_extraction/      Structured-extraction and embedding packages
tests/                    Unit and opt-in vLLM integration tests
outputs/                  Local results and logs (generated, Git-ignored)
```

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

- one- and two-replica NVIDIA B200 configurations
- `vllm==0.19.0`
- `google/medgemma-27b-text-it`
- strict JSON-schema output for both included schemas
- concurrent requests (`--workers 8` per replica)
- dense extraction with cached `emilyalsentzer/Bio_ClinicalBERT` (three notes,
  768 dimensions, finite unit-normalized float32 vectors)

On 25 discharge and 25 nephrology reports, MedGemma returned 50/50 valid
schema-conforming records. Extraction took 7.02 seconds for discharge and
14.86 seconds for nephrology after server startup. The comparison model,
`google/gemma-4-31B-it`, also returned 50/50 valid records and took 8.39 and
17.61 seconds respectively on the same one-B200 setup.

This verifies server startup, OpenAI-compatible requests, constrained decoding,
JSON parsing, schema validation, concurrent output writing, and both clinical
task schemas. Model outputs still require task-specific evaluation on a labeled,
de-identified validation set before clinical or research use.
