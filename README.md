# TRACESTEER

TRACESTEER is a research project for mitigating hallucinations in diffusion language models (DLLMs). Its core implementation, **Feedback Flow**, uses previous predictions to produce trainable residual corrections to Transformer hidden states, evaluated on TruthfulQA and multiple-choice benchmarks.

The project includes training, feedback-based generation, and evaluation entry points designed for LLaDA and Dream, covering TruthfulQA, COPA, StoryCloze, MMLU, Winogrande, and BoolQ.

## Method

Feedback Flow operates as follows:

1. Freeze the base language model and train only the added Feedback Flow module.
2. During training, disable feedback and obtain an initial prediction for masked answer tokens. During generation, use predictions from the previous denoising iteration.
3. Feed the predicted tokens back into the base model and extract hidden states from a designated layer as feedback features.
4. Refine the current hidden states through cross-attention, self-attention, and a feed-forward network, sharing weights across Flow refinement steps.
5. Compute a gate from the denoising timestep and the normalized entropy of the previous prediction, then apply residual corrections at response positions.

The training objective combines cross-entropy over masked answer tokens with a hidden-state stability regularizer:

```text
loss = generation_loss + stability_weight × stability_loss
```

The stability term measures the mean squared difference between the original and corrected hidden states. Training selects the best Feedback Flow parameters by validation loss, saves them, and automatically evaluates the held-out test split. Use `--skip-test` to skip evaluation after training.

## Tasks and Metrics

| Task | Dataset argument / entry point | Evaluation |
| --- | --- | --- |
| TruthfulQA | `our_model.train_feedback_flow` | MC0, MC1, MC2, MC3; optional GPT truthfulness and informativeness metrics |
| COPA | `--dataset copa` | Multiple-choice accuracy |
| StoryCloze | `--dataset storycloze` | Multiple-choice accuracy |
| MMLU | `--dataset mmlu` | Multiple-choice accuracy |
| Winogrande | `--dataset wino` | Multiple-choice accuracy |
| BoolQ | `--dataset boolq` | Multiple-choice accuracy |

Multiple-choice evaluation compares candidate answers using masked-token log probabilities. For TruthfulQA, `--test-metrics all` also generates answers and scores them with an evaluator model, reporting truthfulness, informativeness, their joint metric, and answer diagnostics.

## Installation

The code uses Python 3.10+ syntax and primarily depends on PyTorch, Transformers, NumPy, Pandas, and tqdm. Exact package versions from the experiment environment are recorded in `experiment.txt`, including:

| Dependency | Version |
| --- | --- |
| PyTorch | 2.5.1 |
| Transformers | 4.51.0 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| tqdm | 4.67.3 |

Activate your experiment environment and install dependencies from the repository root:

```bash
pip install -r experiment.txt
```

The dependency file targets Linux with NVIDIA CUDA and includes CUDA 12 runtime packages and Triton. GPU training requires a working NVIDIA driver and CUDA environment. This file is a package-version snapshot: it does not record the Python version, driver version, or package index, and installation in a fresh environment has not yet been verified.

### Implementation Status

The repository currently lacks `model/backbone.py`, which is required by the training and evaluation entry points. This adapter must be supplied in addition to the Python dependencies. It handles model loading, device selection, LLaDA/Dream backbone resolution, prediction-position alignment, and generation, exposing the following interfaces:

```python
DEFAULT_MODEL_DIR
backbone_name
backbone_output_path
forward_model
generate
load_backbone
prediction_mask
resolve_backbone
resolve_backbone_args
token_logits
```

Dream feedback generation also requires a compatible `diffusion_generate` interface on the base model. The examples below assume that the backbone adapter, model checkpoints, and datasets are available.

## Models and Data

Model weights and datasets must be obtained separately; they are not included in this repository.

- **Base model:** A LLaDA or Dream checkpoint containing the configuration, tokenizer, and files required for loading. Specify its location with `--model-dir`.
- **TruthfulQA:** `TruthfulQA_v1.csv` with fields including `Question`, `Best Answer`, `Correct Answers`, and `Incorrect Answers`. Multiple reference answers are separated by semicolons.
- **Other datasets:** Data files compatible with the loaders in `evaluate_COPA.py`. Specify their location with `--data-path`.
- **Generated-answer evaluation:** TruthfulQA GPT metrics use a Qwen3 8B evaluator by default. Configure `DEFAULT_EVAL_MODEL_DIR` in `model/steering_DLLM_.py` before training, or pass `--gpt-eval-model` for standalone evaluation. `--test-gpt-device cpu` controls only the evaluator device.

Some modules contain preset paths. When reproducing experiments, explicitly specify model, data, and output paths, and check the evaluator configuration.

## Training and Evaluation

Run the following commands from the repository root and replace `/path/to/...` with your actual paths.

### Training on TruthfulQA

```bash
python -m our_model.train_feedback_flow \
  --backbone llada \
  --model-dir /path/to/llada-Instruct \
  --data-path /path/to/TruthfulQA_v1.csv \
  --output-path ./logs/feedback_flow_llada.pt \
  --device cuda \
  --epochs 5 \
  --batch-size 1 \
  --layer-idx 16 \
  --feedback-layer-idx 16 \
  --flow-steps 1 \
  --timestep-gate-hidden 32 \
  --stability-weight 0.25 \
  --steps 128 \
  --test-gen-length 128 \
  --test-steps 128 \
  --test-block-length 4 \
  --test-metrics mc0 mc1 mc2 mc3
```

Layer indices must be within the model's layer range. To include GPT metrics, set `--test-metrics all` and prepare the evaluator model. Add `--skip-test` to run training only.

### Training on Other Datasets

```bash
CUDA_VISIBLE_DEVICES=0 python -m our_model.our_model_otherdata.train_feedback_flow_otherdata \
  --dataset boolq \
  --backbone llada \
  --model-dir /path/to/llada-Instruct \
  --data-path /path/to/boolq.jsonl \
  --output-path ./logs/boolq_feedback_flow.pt \
  --device cuda \
  --num-samples 817 \
  --seed 42 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --epochs 3 \
  --batch-size 1 \
  --steps 128 \
  --flow-steps 3 \
  --num-heads 8 \
  --stability-weight 0.01 \
  --flow-scale 1.0
```

Replace `--dataset` with `copa`, `storycloze`, `mmlu`, or `wino` and provide the corresponding data file. To use Dream, set `--backbone dream` and supply a Dream checkpoint. Dream generation requires a CFG scale of 0.

### Evaluating a Trained Module

```bash
python -m our_model.eval_truthfulqa_feedback_flow \
  --backbone llada \
  --model-dir /path/to/llada-Instruct \
  --data-path /path/to/TruthfulQA_v1.csv \
  --flow-path ./logs/feedback_flow_llada.pt \
  --split-manifest-path ./logs/feedback_flow_llada.splits.json \
  --output-path ./logs/feedback_flow_llada.test.json \
  --metrics mc0 mc1 mc2 mc3
```

For other datasets, use the corresponding evaluation entry point:

```bash
python -m our_model.our_model_otherdata.eval_feedback_flow_otherdata \
  --dataset boolq \
  --backbone llada \
  --model-dir /path/to/llada-Instruct \
  --data-path /path/to/boolq.jsonl \
  --flow-path ./logs/boolq_feedback_flow.pt \
  --split-manifest-path ./logs/boolq_feedback_flow.splits.json \
  --output-path ./logs/boolq_feedback_flow.test.json \
  --device cuda
```

## Parameters and Outputs

| Argument | Description |
| --- | --- |
| `--layer-idx` | Transformer layer where Feedback Flow corrections are applied |
| `--feedback-layer-idx` | Layer used to extract feedback hidden states |
| `--flow-steps` | Number of recurrent FlowBlock refinement steps with shared parameters |
| `--steps` | Number of denoising timesteps used for training timestep sampling; distinct from Flow refinement steps |
| `--stability-weight` | Weight of the hidden-state stability regularizer |
| `--mask-prob` | Masking probability for answer tokens during training |
| `--skip-test` | Skip automatic evaluation after training; training dependencies are still required |

The default split ratios are 70% training, 15% validation, and approximately 15% testing, with a random seed of 42. Saved split manifests allow subsequent evaluations to reuse the same partitions.

For `--output-path ./logs/feedback_flow_llada.pt`, the default outputs are:

- `feedback_flow_llada.pt`: Feedback Flow parameters, module configuration, training history, and metadata. It does not contain the full base-model weights.
- `feedback_flow_llada.splits.json`: Split indices and records of the samples used.
- `feedback_flow_llada.test.json`: Evaluation report produced after training.
- `feedback_flow_llada.test.answers.json`: Generated TruthfulQA answers when GPT metrics are enabled.

## Repository Structure

```text
TRACESTEER/
├── our_model/
│   ├── feedback_flow.py             # FlowBlock, gating, model hooks, and parameter I/O
│   ├── generate_feedback.py         # Generation using prediction feedback
│   ├── losses.py                    # Generation and hidden-state stability losses
│   ├── train_feedback_flow.py       # TruthfulQA training entry point
│   ├── eval_truthfulqa_feedback_flow.py
│   ├── test_feedback_flow.py        # Feedback Flow unit tests
│   └── our_model_otherdata/
│       ├── train_feedback_flow_otherdata.py
│       ├── eval_feedback_flow_otherdata.py
│       ├── run_all_feedback_flow_otherdata.sh
│       └── test_backbone_switching.py
├── model/
│   ├── steering.py                  # Hidden-state steering and model structure utilities
│   ├── truthfulqa_steering.py        # Prompts, data splits, and feature extraction
│   ├── steering_DLLM_.py            # TruthfulQA evaluation, aggregation, and evaluator loading
│   └── steering_otherdata/common.py # Dataset loading, prompts, and splits
├── utils/
│   ├── load_data.py                 # TruthfulQA CSV loading
│   └── metric_indector.py           # Evaluation metrics
├── evaluate_COPA.py                 # Loaders and prompt formatters for five datasets
├── experiment.txt                   # Experiment dependency versions
├── result.md                        # Experiment commands and historical results
└── README.md
```

## Testing

After installing dependencies and supplying the backbone adapter, inspect the CLI arguments and run the existing unit tests:

```bash
python -m our_model.train_feedback_flow --help
python -m our_model.our_model_otherdata.train_feedback_flow_otherdata --help
python -m unittest our_model.test_feedback_flow our_model.our_model_otherdata.test_backbone_switching
```

Unit tests cover the feedback module, losses, and selected backbone-switching behavior. Full reproduction also requires validating training, generation, and evaluation with actual models and datasets.

## Experiment Records

`result.md` contains historical experiment commands and evaluation results. Before reusing these commands, replace model and data paths and correct any unquoted spaces in paths. The historical results have not yet been reproduced end to end with the current repository.
