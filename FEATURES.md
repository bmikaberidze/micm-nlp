# Features

*ready-made functionality*

> What can I do here that the HuggingFace stack does not already do?

Every entry says what HuggingFace does on its own, and links to the module that implements it.

## Data

- **One `DATASET` class over every source** — local CSV / TXT / JSON, the Hub, and
  `save_to_disk` directories behind a single config block, instead of a different
  loader call per source.
  → [`datasets.dataset`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/datasets/dataset/index.html)
- **Concatenation across directories** — `get_concatenated_dataset` assembles one
  training set from a per-language (or per-domain) path template. This is the
  mechanism behind multilingual source groups; `datasets` gives you `concatenate_datasets`
  but no path-template layer above it.
  → [`DATASET.get_concatenated_dataset`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/datasets/dataset/index.html)
- **Tokenization in three configurable stages** — `tokenize.pre_rules` (sentence
  splitting, EOS appending, text-to-text reframing), `tokenize.rules` (handed to the
  tokenizer verbatim), and `tokenize.post_rules` (EOS-aware truncation, sample
  concatenation, length sorting). HuggingFace gives you the tokenizer call in the
  middle; everything on either side of it is normally yours to write.
  → [`DATASET.preprocess`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/datasets/dataset/index.html)
- **Subsetting and splitting as configuration** — use a fixed fraction of a corpus
  (`preproc_rules.subset`), carve train / validation / test by ratio
  (`preproc_rules.split`), or separate a split by token length
  (`preproc_rules.split_by_tokens_len`) — all seeded, so the same config gives the same
  partition.
- **Column standardisation** — `ds.input.standardize_key` renames a corpus's own
  column names onto the canonical `inputs` / `labels` / `task_ids`, so downstream code
  never learns any corpus's vocabulary.
- **Splits for datasets that have none** — `ds.splits` handles corpora that
  ship splits under non-standard names, or ship none at all.
- **Length statistics** — `analyze_lengths` reports the real token-length distribution
  of a split, so `max_length` is chosen from evidence rather than guessed. Truncating
  33 of 205 languages because 164 looked like a safe default is the kind of thing this
  catches.
  → [`DATASET.analyze_lengths`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/datasets/dataset/index.html)

## PEFT, including a method of our own

- **The Cross-Prompt Encoder (XPE)** — the method from *Cross-Prompt Encoder for
  Low-Performing Languages* (Findings of IJCNLP–AACL 2025). Not a wrapper around
  someone else's work: a published contribution that ships here.
  → [`models.xpe`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/models/xpe/index.html)
- **XPE, SPT and DUAL are one class**, separated by `encoder_ratio`: `0` is plain soft
  prompt tuning, `1` is pure XPE, anything between is DUAL. Comparing the three is a
  config change, not a code path.
  → [`models.xpe.encoder`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/models/xpe/encoder/index.html)
- **Three reparameterisation heads** — MLP, bidirectional LSTM, and a lightweight
  self-attention head.
  → [`models.xpe.heads`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/models/xpe/heads/index.html)
- **XPE-aware state-dict save/load** — stock `get_peft_model_state_dict` reaches for
  `prompt_encoder.embedding.weight` unconditionally, which does not exist for pure XPE,
  so saving a pure-XPE adapter with upstream PEFT raises.
  → [`models.xpe.save_load`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/models/xpe/save_load/index.html)
- **One dispatch point** — the `peft` config block routes to stock PEFT (LoRA, prefix
  tuning, P-tuning) or the XPE path without changing anything else.
  → [`models.peft`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/models/peft/index.html)
- **`CustomT5ForConditionalGeneration`** — T5 with optional FlashAttention, pretrained
  q/k/v copied across.
  → [`models.architectures`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/models/architectures/index.html)

## Training

- **Token-budget batching** — batch by token count rather than row count, with an
  `'auto'` mode that probes the GPU for the largest budget that does not run out of
  memory. HuggingFace batches by row count only, so a batch of long sequences and a
  batch of short ones cost wildly different amounts of memory.
  → [`training.batching`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/training/batching/index.html)
- **Early stopping decoupled from model selection** — HuggingFace ties them together:
  the metric that stops the run is the metric that picks the best checkpoint. Here they
  are separate, plus an `early_stopping_after` floor so a slow start is not cut short.
  → [`training.callbacks`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/training/callbacks/index.html)
- **Per-parameter-group optimizer settings from YAML** — give prompt embeddings their
  own learning rate and weight decay while the rest of the model uses another, without
  writing an optimizer.
  → [`custom_training_args.optimizer_grouped_parameters`](https://micm-nlp.readthedocs.io/en/latest/config.html)
- **`ParamNormLogger`** — logs parameter norm and per-step update norm, a stability
  probe that catches divergence before the loss shows it.
- **Collators HuggingFace lacks** — permutation language modelling *with padding*,
  seq2seq labels shifted to account for virtual tokens, and a task-id decorator that
  wraps any other collator.
  → [`training.data_collators`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/training/data_collators/index.html)
- **Closed-set generation** — restrict a generative model to a fixed set of allowed
  strings at generation time, turning it into a classifier without touching the model.
  → [`training.logits_processors`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/training/logits_processors/index.html)
- **Random task exclusion per batch**, for multi-task training.
  → [`training.trainers`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/training/trainers/index.html)

## Evaluation

- **Label-restricted likelihood** — score only the candidate label tokens instead of
  the whole vocabulary, the way lm-eval-harness does multiple choice. Requires
  `ds.label.names`.
  → [`evals.eval`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/evals/eval/index.html)
- **Length-normalised log-likelihood accuracy** — without normalisation the shortest
  option wins by construction, so an unnormalised multiple-choice score measures
  length as much as it measures the model.
  → [`evals.metrics.log_likelihood`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/evals/metrics/log_likelihood/index.html)
- **A declarative post-processing chain** between predictions and metrics: flatten,
  drop padding, decode, strip and lower, map names to ids or floats — as
  `task.preproc_rules`, not as code.
- **Per-task metric grouping**, so one run scores several tasks separately and together.
- **Confusion matrices** written per evaluation.
  → [`evals.plot`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/evals/plot/index.html)

## Tokenizers

- **A factory that knows the architectures** — applies the right special tokens and
  post-processor per architecture, rather than leaving it to the caller.
  → [`tokenizers.tokenizer`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/tokenizers/tokenizer/index.html)
- **Tokenizer training** — native SentencePiece, HuggingFace WordPiece, and byte-level
  BPE, behind one `TokenizerTrainer`.
- **`BertByT5Tokenizer`** — a byte-level vocabulary wearing BERT's special tokens.
- **`CustomXlmRoberta`** — XLM-R's multilingual vocabulary re-dressed for another
  architecture.
  → [`tokenizers.architectures`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/tokenizers/architectures/index.html)
- **`KaSenTok`** — a Georgian sentence splitter from *A Comparison of Different
  Tokenization Methods for the Georgian Language* (ICNLSP 2024), carrying 885
  abbreviations extracted from Georgian Wikipedia.
  → [`tokenizers.ka_sen_tok`](https://micm-nlp.readthedocs.io/en/latest/autoapi/micm_nlp/tokenizers/ka_sen_tok/index.html)

## What is not ours

<!-- start:features-boundary -->
`Trainer`, model loading, the `datasets` library, `evaluate`, and PEFT's base method
implementations are HuggingFace's work. This package is a layer on top of them. The
list above is what the layer adds; everything underneath it belongs to the people who
wrote it.
<!-- end:features-boundary -->
