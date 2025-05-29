# VivaBench: Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in LLMs

This repository is the official implementation of *VivaBench—“Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models.”*

VivaBench is a multi-turn benchmark of 1,152 physician-curated clinical vignettes that simulates a viva voce (oral) exam: agents must iteratively gather H&P findings and order investigations to arrive at a diagnosis.

## 📋 Requirements

API-keys for OpenAI/OpenRouter if you use those providers. See **Configuration** below.

## 🛠 Installation
Install the package in editable mode to expose the vivabench console script:

```
git clone 
pip install -e .
```

```
$ which vivabench
/path/to/venv/bin/vivabench
```
## ⚙️ Configuration

All pipeline parameters live in YAML:

- **configs/evaluate.yaml**
    
    - `data.input` → input CSV of vignettes
    - `data.output_dir` → where to write logs & results
    - `data.batch_size`, `data.max_workers`
    - `models.examiner`, `models.agent` blocks (provider, model, temp, API‐key/env)
    - `examination.*` → mapper/parser limits & SNOMED path
    - `logging.level`
- **configs/generate.yaml**
    
    - `pipeline.input` / `pipeline.output` / `pipeline.batch_size` / `pipeline.limit`
    - `embeddings.*`, `mappings.*`
    - `models.generator`, `models.reasoning`
    - `logging.level`

Edit the defaults, or override via CLI flags.

---

## 📚 Demo
To get an overview of the core functions within the VivaBench framework, the best entry point is `demo.ipynb`. 


## 🚀 CLI Usage

### 1. Run the Evaluation Pipeline
To reproduce experiment results outlined in our paper, simply run the evaluation pipeline

```bash
vivabench evaluate \
  --config configs/evaluate.yaml \
  [--input     /path/to/my_input.csv] \
  [--output_dir /path/to/outdir] \
  [--evaluation_id id_of_evaluation_run]
```

- Reads `data.input` or `--input` override
- Instantiates examiner & agent models via `init_chat_model`, `init_openrouter_chat_model`, or `init_ollama_chat_model`
- Executes `run_examinations_parallel(...)`
- Saves per-case logs in `output_dir/logs/` and results CSVs in `output_dir/results/`

### 2. Re-run Metrics on Existing Output
The evaluation pipeline runs metrics by default. However, if you want to re-run metrics on a specific file, you can use this command

```bash
vivabench metrics \
  --config configs/evaluate.yaml \
  --output_csv /path/to/results/full_results.csv \
  [--output_dir /path/to/metrics_out]
```

- Loads your evaluation YAML & the `--output_csv`
- Calls `EvaluationMetrics(...)` to compute accuracy, precision/recall, confidence scores
- Writes `metrics.csv` under the same output directory

### 3. Run the Generation Pipeline
If you want to generate more cases from clinical vignettes, you can use this command
```bash
vivabench generate \
  --config configs/generate.yaml \
  [--input  /path/to/seed_vignettes.csv] \
  [--output /path/to/generated.csv]
```

- Builds and runs `PipelineConfig(...)` → `run_pipeline(...)`
- Produces a structured clinical case dataset in the specified `pipeline.output`

## 🎓 Citation
If you use VivaBench in your work, please cite:


```
@article{vivabench2025,
  title   = {Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models},
  author  = {Anonymous Author(s)},
  journal = {},
  year    = {2025},
}
```
## 📝 License & Contributing
This project is released under the CC-NA License. Contributions welcome—please open an issue or pull request.