# MAGER

## 1. Install

```bash
cd MAGER-GitHub
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download data

Download `MAGER-dataset` from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1gNF5Wl-nrCiDETUtRYKP-A?pwd=8aku
Code: 8aku
```

After downloading, put the dataset folder under the project root as `data`:

```text
MAGER-GitHub/
├── data/
│   ├── politifact/
│   ├── gossipcop/
│   └── mcfend/
├── src/
├── requirements.txt
└── README.md
```

For example, if the downloaded folder is named `MAGER-dataset`, run:

```bash
cd MAGER-GitHub
mv /path/to/MAGER-dataset ./data
```

## 3. Configure LLM API

```bash
export AGENT_A_BASE_URL=http://127.0.0.1:8000/v1
export AGENT_G_BASE_URL=http://127.0.0.1:8000/v1
export AGENT_D_BASE_URL=http://127.0.0.1:8000/v1
export AGENT_E_BASE_URL=http://127.0.0.1:8000/v1

export AGENT_A_API_KEY=EMPTY
export AGENT_G_API_KEY=EMPTY
export AGENT_D_API_KEY=EMPTY
export AGENT_E_API_KEY=EMPTY

export AGENT_A_MODEL=qwen3-8b
export AGENT_G_MODEL=qwen3-8b
export AGENT_D_MODEL=qwen3-8b
export AGENT_E_MODEL=qwen3-8b
```

## 4. Check data

```bash
PYTHONPATH=src/mager python src/mager/check_data.py --dataset politifact
PYTHONPATH=src/mager python src/mager/check_data.py --dataset gossipcop
PYTHONPATH=src/mager python src/mager/check_data.py --dataset mcfend
```

## 5. Generate enhanced data if needed

If `data/<dataset>/enhanced_dataset_<dataset>_qwen3-8b.jsonl` is missing, run:

```bash
PYTHONPATH=src/mager python src/mager/run.py preprocess --dataset politifact --limit 400
```

Change `politifact` to `gossipcop` or `mcfend` for other datasets.

## 6. Evaluate

```bash
PYTHONPATH=src/mager python src/mager/run.py evaluate --dataset politifact --limit 50
```

## 7. Evolve meta-paths

```bash
PYTHONPATH=src/mager python src/mager/run.py evolve \
  --dataset politifact \
  --evolve_limit 100 \
  --test_limit 200 \
  --max_iterations 100 \
  --population_size 20
```

## Common parameters

- `--dataset`: dataset name. Choose from `politifact`, `gossipcop`, `mcfend`.
- `--limit`: number of samples used in evaluation, or number of fake/real samples used separately in preprocessing.
- `--evolve_limit`: number of samples used for meta-path evolution.
- `--test_limit`: number of samples used for final evaluation after evolution.
- `--max_iterations`: number of genetic evolution iterations.
- `--population_size`: number of meta-path candidates kept in the population.
- `--icl_pool_size`: size of the candidate pool for in-context example retrieval.
- `--top_k`: number of retrieved few-shot examples for each query.
- `--meta_path_indices`: manually select initial meta-paths by index, for example `--meta_path_indices 0 1 2`.
- `--meta_path_json`: load selected/evolved meta-paths from a JSON file.
- `--no_save`: run evaluation without saving result files.
