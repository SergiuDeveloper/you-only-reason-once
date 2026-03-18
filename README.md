# YORO - You-Only-Reason-Once

Designing an architecture that performs full reasoning only once, instead of re-running the entire model for every generated token.

---

## Motivation

Standard autoregressive LLMs redo the full forward pass, including all the expensive middle "reasoning" layers, for every single generated token. The prompt is re-analyzed, the reasoning stack re-runs, and only then does the model emit the next token. Most of the per-token cost comes from this repeated reasoning, not from selecting the token itself.

**YORO's core idea:** run the heavy reasoning block exactly once (on the prompt), cache its output, and reuse it for all subsequent tokens. A set of small trainable subnets compensates for the missing reasoning passes so the model can still generate coherent, high-quality continuations.

---

## Architecture

The architecture splits a pretrained LLM (currently TinyLlama-1.1B-Chat-v1.0) into three frozen blocks plus three small trainable subnets.

### Frozen blocks (from the base model)

| Block                | Contents                                             | Role                                                |
| -------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| **Embedding subnet** | First N transformer layers + token embeddings + RoPE | Converts tokens into contextualized representations |
| **Reasoning subnet** | Middle M transformer layers                          | Deep reasoning - run **only once**, on the prompt   |
| **Coherence subnet** | Final K transformer layers + LayerNorm + LM head     | Converts hidden states to output logits             |

### Trainable subnets (learned from scratch)

| Subnet                   | Type                                                            | Role                                                                                             |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Adaptation subnet**    | MLP (Linear → ReLU stacked)                                     | Transforms the cached reasoning output so it can be reused at later positions                    |
| **Compensation subnet**  | Newly-initialized transformer layers (same class as base model) | Processes the current embedding-level representation to compensate for the absent reasoning pass |
| **Concatenation subnet** | MLP (Linear → ReLU stacked)                                     | Merges adaptation and compensation outputs before the coherence block                            |

### Key property

The reasoning subnet never runs again after the first token. For a sequence of T generated tokens, reasoning cost is O(1) rather than O(T), and all the adaptation/compensation/concatenation subnets are tiny relative to the full model.

---

## Two forward modes

### Autoregressive mode (inference)

The model generates one token at a time, maintaining a cache of the reasoning output across steps.

1. **First token:** embed → embedding subnet → reasoning subnet → cache output → coherence subnet → logit
2. **All later tokens:** embed → embedding subnet → compensation subnet (on current step) + adaptation subnet (on cached reasoning) → concatenation subnet → coherence subnet → logit

The cache is never updated after the first token.

### Teacher forcing mode (training)

To train efficiently, the entire prompt + response sequence is processed **in a single parallel forward pass** rather than one token at a time:

1. Embed the full sequence.
2. Run the embedding subnet on the full sequence.
3. Run the reasoning subnet **only on the prompt portion** `[0, prompt_length)` and cache the result.
4. Compute logits for prompt positions using only the cached reasoning output → coherence subnet.
5. Pad the cached reasoning tensor to the full sequence length.
6. Run adaptation and compensation subnets over the full sequence in parallel.
7. Merge through the concatenation subnet → coherence subnet to get logits for the generated positions `[prompt_length, seq_length)`.
8. Concatenate all logits and return them.

This makes training orders of magnitude faster than running the autoregressive path step-by-step, while being equivalent in what the model is trained to do. The test suite (`tst/test_teacher_forcing.py`) verifies that both modes produce the same predictions.

---

## Repository structure

```
you-only-reason-once/
├── src/
│   └── subnet_model.py          # Core model definition
├── tst/
│   └── test_teacher_forcing.py  # Correctness and speed tests
├── notebooks/
│   ├── generate_dataset.ipynb   # Build training dataset with teacher logits
│   ├── training.ipynb           # Knowledge-distillation training loop
│   ├── inference.ipynb          # Load checkpoint and generate text
│   ├── eval_api.ipynb           # Evaluate trained model via lm-eval harness
│   ├── eval_base_transformers.ipynb  # Baseline TinyLlama evaluation
│   └── eval_investigate.ipynb   # Per-task debugging / qualitative analysis
└── README.md
```

---

## Source code

### `src/subnet_model.py`

All model classes live here.

**`BaseModel`** - thin `nn.Module` base that adds a `num_parameters()` helper.

**`SubnetLLM`** - the main class. Constructor arguments:

| Argument               | Meaning                                                                           |
| ---------------------- | --------------------------------------------------------------------------------- |
| `model_name`           | HuggingFace model ID (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)                  |
| `cache_dir`            | Local cache directory for the base model weights                                  |
| `embedding_layers`     | How many of the base model's transformer layers to assign to the embedding subnet |
| `coherence_layers`     | How many to assign to the coherence subnet                                        |
| `compensation_layers`  | How many newly-initialized transformer layers to use in the compensation subnet   |
| `adaptation_layers`    | How many MLP layers for the adaptation subnet                                     |
| `concatenation_layers` | How many MLP layers for the concatenation subnet                                  |
| `device`               | `'cuda'` or `'cpu'`                                                               |
| `dtype`                | `torch.bfloat16` (default)                                                        |

The remaining base-model layers (everything between the embedding and coherence blocks) become the reasoning subnet. All frozen subnet parameters have `requires_grad=False`. Only adaptation, compensation, and concatenation subnets are trained.

**`TransformerSubnet`** - wraps a slice of frozen transformer layers for the embedding and reasoning subnets. Passes hidden states through layers sequentially with rotary position embeddings and a causal mask.

**`CompensationSubnet`** - same interface as `TransformerSubnet` but layers are freshly initialized (random weights) using the same layer class as the base model. These weights are learned during training.

**`MLPSubnet`** - stacked `Linear → ReLU` layers, all with Xavier-normal weight initialization. Used for both the adaptation and concatenation subnets.

**`CoherenceSubnet`** - wraps the frozen final transformer layers, layer norm, and LM head. Always frozen.

---

## Tests

### `tst/test_teacher_forcing.py`

Three test functions:

**`test_batch_equivalence()`**
Checks that teacher forcing mode and autoregressive mode produce identical predictions. Three test sequences are created; for each the prompt (first 5 tokens) is split from the generated portion. The autoregressive path runs token-by-token collecting logits; teacher forcing processes the full sequence at once. The test asserts that predicted tokens match at every position for every sequence. Logit differences (max and mean) are also logged for debugging.

**`test_speed_comparison()`**
Benchmarks teacher forcing (parallel) vs autoregressive (sequential). Runs trials for 1 and 10 generation tokens, reports the wall-clock speedup factor and percentage improvement.

**`test_speed_comparison_vs_original()`**
Benchmarks `SubnetLLM` against the original unmodified TinyLlama model. Tests token counts of 1, 10, 50, 100, and 200 tokens, and reports the speedup (or slowdown) of SubnetLLM relative to the base model. This is the primary measure of real-world inference gain.

---

## Notebooks

### `notebooks/generate_dataset.ipynb` - build the training dataset

Creates the `SergiuNistor/yoro-train` dataset used for distillation.

1. Loads five public instruction-following datasets:
   - `vicgalle/alpaca-gpt4`
   - `databricks/databricks-dolly-15k`
   - `WizardLMTeam/WizardLM_evol_instruct_V2_196k`
   - `Open-Orca/SlimOrca`
   - `teknium/OpenHermes-2.5`
2. Parses each into `(system_prompt, user_message)` pairs and applies TinyLlama's chat template.
3. Runs the **teacher model** (TinyLlama via vLLM) to generate up to 256 response tokens per prompt, collecting the top-10 log-probabilities at every position.
4. Applies temperature scaling (`SOFT_TEMPERATURE = 3.0`) to soften the teacher distribution: `p_soft = exp(logprob / temperature)`, then normalizes.
5. Saves three arrays per example:
   - `input_token_ids` - prompt token IDs
   - `logprob_token_ids` - top-10 token IDs at each generated position
   - `logprob_values` - corresponding (softened, normalized) probabilities

### `notebooks/training.ipynb` - Stage 1 distillation training

The current stage: fine-tuning SubnetLLM by distilling from TinyLlama.

**Key components:**

- **`collate_batch()`** - left-pads variable-length sequences to the batch maximum; records prompt lengths and creates validity masks for loss computation.
- **`compute_distillation_loss()`** - for each non-padded output position, reconstructs the full teacher distribution (sparse top-10 → dense vocab), then computes KL divergence `KL(teacher || student)`. Only valid positions contribute to the loss.
- **`train_epoch()` / `validate()`** - standard training loop with mixed-precision (`bfloat16`), gradient accumulation (`GRADIENT_ACCUMULATION_STEPS = 2`), gradient clipping (`MAX_GRAD_NORM = 1.0`), and checkpoint saving every 10,000 steps.
- **`save_training_checkpoint()` / `load_training_checkpoint()`** - serializes full training state (model weights, optimizer, GradScaler, RNG states) for mid-epoch resumption.

Default hyperparameters:

| Parameter                     | Value |
| ----------------------------- | ----- |
| `BATCH_SIZE`                  | 32    |
| `LEARNING_RATE`               | 3e-4  |
| `NUM_EPOCHS`                  | 10    |
| `WEIGHT_DECAY`                | 0.01  |
| `GRADIENT_ACCUMULATION_STEPS` | 2     |
| `MAX_GRAD_NORM`               | 1.0   |

### `notebooks/inference.ipynb` - generate text from a checkpoint

Loads a saved SubnetLLM checkpoint, reconstructs the model config, and provides two generation helpers:

- **`generate(prompt, max_tokens)`** - greedy token generation from a plain text string.
- **`generate_chat(messages, max_tokens)`** - applies the TinyLlama chat template and generates a response.

Includes example prompts in all five dataset formats (Alpaca, Dolly, WizardLM, SlimOrca, OpenHermes) so you can quickly verify generation quality after training.

### `notebooks/eval_api.ipynb` - evaluate trained model

Starts a vLLM server exposing the trained SubnetLLM on a local port, then runs the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) `leaderboard` task suite (BBH, MATH, IFEVAL, etc.) with batch size 128. Results are directly comparable to public leaderboard numbers.

### `notebooks/eval_base_transformers.ipynb` - baseline evaluation

Runs the same `lm-evaluation-harness` leaderboard tasks on the unmodified pretrained TinyLlama model (batch size 64, bfloat16). Provides the baseline numbers that SubnetLLM results should be compared against.

### `notebooks/eval_investigate.ipynb` - per-task qualitative analysis

Runs a single benchmark task (`leaderboard_bbh_boolean_expressions`) limited to 5 examples through the local API server. Used for debugging and understanding individual model failures.

---

## Training stages

### Stage 1 - Knowledge distillation fine-tuning (current, in progress)

The goal of Stage 1 is to avoid pretraining from scratch. Instead, it starts from a strong pretrained model (TinyLlama-1.1B-Chat-v1.0), freezes most of it, and trains only the three small subnets (adaptation, compensation, concatenation) to compensate for the missing reasoning passes.

The distillation setup is critical here. Because the student must learn to match the teacher's output distribution without the reasoning subnet running on generated tokens, naive next-token prediction loss is insufficient. The student never sees ground-truth reasoning states. Instead:

- The **teacher** is the original TinyLlama running normally (all layers, every token).
- The student receives teacher soft-label distributions (top-10 logprobs, temperature-scaled) rather than hard token labels.
- The **teacher forcing masking mechanism** allows the full prompt+response sequence to be processed in a single parallel pass, with the reasoning subnet running only on the prompt portion and the adaptation/compensation/concatenation path covering the rest, faithfully simulating inference while enabling efficient batched training.

This masking is the key technical contribution that makes Stage 1 tractable: without it, you would have to run the autoregressive loop token-by-token during training, which is far too slow.

### Stage 2 - Full pretraining (current, in progress, https://github.com/SergiuDeveloper/yoro-full-pretraining)

Stage 2 abandons fine-tuning in favor of training SubnetLLM from scratch on a large corpus. Rather than distilling from a teacher, the model will be pretrained end-to-end with the YORO architecture from random initialization of the three trainable subnets (the frozen blocks will still be initialized from a pretrained model, at least initially, or may themselves be trained).

The goal is to determine whether a model that has never seen "full reasoning on every token" during training can learn to produce high-quality outputs through the adaptation+compensation pathway alone, and whether this yields a fundamentally more efficient architecture class at scale.

---

## Dependencies

- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm) (dataset generation and evaluation serving)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (benchmarking)
- [Datasets](https://github.com/huggingface/datasets)
- [Hugging Face Hub](https://huggingface.co/) (model and dataset hosting)
