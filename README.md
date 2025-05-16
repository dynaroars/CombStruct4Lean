# CombStruct4Lean

<p align="center">
    <a href="https://huggingface.co/datasets/10n9/CombStruct4Lean"><img src="https://img.shields.io/badge/🤗-huggingface-FFD21E"></a>
</p>

CombStruct4Lean is a benchmark of 383 formalized combinatorics problems in the Lean 4 proof assistant, sourced from high-school olympiad-level math competitions. Unlike existing datasets, CombStruct4Lean emphasizes the creation and use of problem-specific combinatorial structures, making it particularly challenging for both autoformalization and automated theorem proving tasks. The benchmark is built using an iterative LLM-guided pipeline with semantic and human verification, providing a realistic setting for evaluating formal reasoning in combinatorics.

## Requirements

This project requires `python >= 3.12` and `lean-toolchain==4.9.0`

Install dependencies

```bash
pip install -r requirements.txt
```

Build Mathlib4 library

```bash
cd mathlib4
lake build
```

Setup DeepseekProver dependency
```bash
rmdir dsprover/mathlib4 
ln -sf $(realpath mathlib4) dsprover
```

Setup API keys

```bash
export OPENAI_API_KEY=<your-api-key>
export ANTHROPIC_API_KEY=<your-api-key>
```

## Benchmark
Our benchmark is in `data/CombStruct4Lean.jsonl` and in this [HuggingFace dataset](https://huggingface.co/datasets/10n9/CombStruct4Lean).

The code to generate our benchmark is in `scripts/pipeline_{formalize, informalize,semantic_check}.py`. 

Before running the code, you need to embed the Mathlib4 by running this command:

```bash
python scripts/encode_mathlib.py \
  mathlib4 \ 
  <output_path> \
  --model_name nomic-ai/CodeRankEmbed
```

To perform benchmark creation process (Sec. 3), change the configuration files `configs/{formalization, informalization,semantic_check}.yaml` and run

```bash
# Formalization process (Sec. 3.2)
python scripts/pipeline_formalize.py

# Informalization (Sec. 3.3)
python scripts/pipeline_informalization.py

# Semantic check (Sec 3.3)
python scripts/pipeline_semantic_check.py
```

## Evaluation
### Autoformalization (Sec. 4.1)
Run experiments with [`Goedel-Formalizer` models](https://huggingface.co/Goedel-LM)

```bash
python scripts/exp_formalize_vllm.py \
    --input_path data/CombStruct4Lean.jsonl \
    --output_path <output_path> \
    --model_path <HuggingFace_path> \
    --num_samples {1, 16}
```

Run ablation Study:

```bash
# No Premise setting
python scripts/exp_ablation.py --no-search

# No Guided Feedback setting
python scripts/exp_ablation.py --no-search --no-feedback
```

Evaluate with `Ground-truth Alignment`

```bash
python scripts/eval_autoformalization.py \
  /path/to/experiment/output \
  data/CombStruct4Lean.jsonl

python scripts/summarize_autoformalization.py \
  /path/to/experiment/output/1 \
  /path/to/experiment/output/2 \
  ...
```

### Automated Theorem Proving (Sec. 4.2)

Run experiments with [`Deepseek-Prover`](https://arxiv.org/abs/2408.08152) and [`Goedel-Prover`](https://arxiv.org/abs/2502.07640)

```bash
cp -t dsprover/configs configs/sampling_deepseek.py configs/sampling_goedel.py
cp data/benchmark_dsprover.jsonl dsprover/datasets

python -m prover.launch \
  --config=configs/sampling_{deepseek,goedel}.py \
  --log_dir=logs/sampling_{deepseek,goedel}

python -m prover.summarize \
  --config=configs/sampling_{deepseek,goedel}.py \
  --log_dir=logs/sampling_{deepseek,goedel}
```

Run experiments with `Claude-3.5-Sonnet` and `o4-mini`

```bash
python scripts/exp_atp.py \
  --config configs/theorem_proving_{sonnet,o4mini}.yaml \
  --num-responses {1,5,10}

python scripts/proof_compile_check.py /path/to/experiment/output
```