# Separate 3080 Ti Benchmark Worker

## What can be distributed

A GPU in another computer cannot be pooled with the local 4060 Ti and 3070 Ti for one model-generation process. It can run an independent benchmark worker. Assign whole `(suite, arm, seed)` generation jobs to one machine and merge only completed output files.

## Required parity

1. Check out the same Git commit as the primary machine.
2. Create the same `research` conda environment from the frozen dependency file.
3. Copy or mount the frozen model snapshot, all six QLoRA adapter directories, and the benchmark snapshots.
4. Set the same absolute data layout, or export the data-root overrides below.
5. Record the commit, model snapshot revision, adapter artifact checksums, prompt template, decoding parameters, and benchmark revisions in the worker log.

## Worker data roots

The official-suite generator reads these paths without network access:

```powershell
$env:LIVECODEBENCH_DATA_ROOT = 'D:\UNLV-Research\hf_cache\hub\datasets--livecodebench--code_generation_lite\snapshots\0fe84c3912ea0c4d4a78037083943e8f0c4dd505'
$env:BIGCODEBENCH_DATA_ROOT = 'D:\UNLV-Research\hf_cache\hub\datasets--bigcode--bigcodebench\snapshots\b74c0d0bf70d2c0bc459be537895cca163007f1a'
$env:HF_DATASETS_CACHE = 'D:\UNLV-Research\hf_datasets_cache'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

Do not make two computers append to the same JSONL over a network share. Each worker writes locally and copies a completed file to the primary result directory after verifying the line count and SHA-256 checksum.

## Recommended ownership

| Machine | Work ownership |
| --- | --- |
| Local 4060 Ti | EvalPlus generation/scoring and small suites |
| Local 3070 Ti | DS-1000 generation or evaluator jobs |
| Separate 3080 Ti | LiveCodeBench and BigCodeBench generation, then OJBench |

Within a suite, one worker should own a complete arm at a time. For example, assign the 3080 Ti the seven base/Raw/Curated artifacts for BigCodeBench; do not split one artifact across machines.

## Google Colab worker

The VS Code Colab worker follows the same ownership rule. It is a generation-only
worker; all official scoring remains on the primary Windows machine. The frozen
contract is `protocols/colab_benchmark_worker_v1.json`, and the executable notebook
is `notebooks/colab_benchmark_worker_v1.ipynb`.

Open `UNLV-Colab.code-workspace`, connect the notebook with `Select Kernel > Colab
> New Colab Server`, then upload the source bundle and adapter-parts folder from
`Colab Transfer`:

```text
colab_worker_source_v1.zip
adapter_parts_4m/
```

The adapter archive is split into 379 parts capped at 4 MiB. Runtime evidence
showed that the Colab Contents API rejected every 24 MiB part with HTTP 400/500,
while the smaller transfer contract stays below the base64 request limit. The
notebook reconstructs the archive on Drive and checks its complete SHA-256 before
extraction.

The notebook copies the archives to Google Drive, verifies source, adapter, and
benchmark SHA-256 values, downloads the exact Qwen snapshot, and checks all remote
paths before loading the model. Its initial queue owns only DS-1000 generation for
Raw seeds 101 and 202. This does not overlap with the local BigCodeBench workers.

Do not compare an arm produced on Colab with local arms until the manifest and
preflight cells pass. Record the allocated GPU in the notebook output. A Colab
runtime may disappear, so completed outputs are written under
`MyDrive/UNLV-Colab-Worker-v1/results`; incomplete `.tmp` files are not scoreable.

## Preflight and run

Run this before a worker starts generation:

```powershell
conda run --no-capture-output -n research python -c "from external_evaluation.official_suite_generator import preflight_suite; [preflight_suite(s) for s in ('livecodebench', 'bigcodebench', 'ds1000', 'ojbench')]; print('preflight: pass')"
```

Example worker job on its own GPU 0:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
conda run --no-capture-output -n research python external_evaluation\official_suite_generator.py --suite bigcodebench --arm raw_safe_natural --seed 11
```

Before merging, check that BigCodeBench has 1,140 JSONL records, OJBench has 464 records, and each output is non-empty. LiveCodeBench writes one JSON array and must be validated by parsing the JSON and counting its records.
