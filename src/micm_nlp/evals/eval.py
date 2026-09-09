"""Metrics: what is computed after each evaluation, and on what.

``get_compute_metrics`` builds the ``compute_metrics`` callable handed to the HF
Trainer. Between raw model output and a number there are several configurable steps,
all driven by ``task.preproc_rules``: decode or flatten predictions, drop padded
positions, convert label ids to names or back, verify predictions and labels line up,
group by task, and finally compute whatever ``task.metric_groups`` names.

``get_preprocess_logits_for_metrics`` returns the companion hook that runs *before*
logits leave the GPU. Taking the argmax there is what keeps an evaluation from
accumulating full logits over the whole split. It is also where
``preproc_rules.label_restricted_likelihood`` restricts the answer-slot argmax to the
candidate label tokens in ``ds.label.names``, lm-eval-harness ``multiple_choice``
style, while returning the same prediction shape as the ordinary causal-LM path — so
nothing downstream needs to branch.

``ds_split`` may be passed as a callable thunk instead of a Dataset; it is resolved
once at call time, which is how the split gets reordered to match a batch sampler
that does not yield in dataset order.
"""

import evaluate
import numpy as np
import torch

from micm_nlp.enums import TaskCatSE
from micm_nlp.evals.metrics.log_likelihood import compute_log_likelihood_accurac
from micm_nlp.evals.metrics.multirc import compute_multirc
from micm_nlp.evals.plot import calc_confusion_matrix
from micm_nlp.tokenizers.decoding import batch_decode

labels_k = 'references'
predictions_k = 'predictions'


def get_compute_metrics(config, label_pad_id, metric_prefix, eval_path, tokenizer, ds_split):
    """Build the ``compute_metrics`` callable the HuggingFace ``Trainer`` expects.

    A closure rather than a method, because ``Trainer`` calls it with only
    ``(predictions, labels)`` -- everything else it needs has to be captured here.

    :param config: the run config; supplies the metric groups and preprocessing rules.
    :param label_pad_id: label padding id, excluded from scoring.
    :param metric_prefix: prefix for the returned metric names (``eval_``, ``test_``).
    :param eval_path: directory for artefacts such as the confusion matrix.
    :param tokenizer: needed when the rules ask for decoding.
    :param ds_split: the split being scored. May be a **thunk**, so ordering
        decisions can be deferred until the dataloader exists -- a length-sorted
        batch sampler yields rows in a different order than the dataset. It is
        resolved once, so verification and preprocessing see one snapshot.
    :returns: the ``compute_metrics`` function.
    """
    print('Get compute_metrics function...')
    print(' metric_prefix:', metric_prefix)
    print(' label_pad_id:', label_pad_id)
    print(' tokenizer:', type(tokenizer).__name__)
    print(' ds_split:', ds_split)

    def compute_metrics(eval_pred):
        """
        This function evaluates the performance of a model
        by comparing its predictions against the true labels.
        --
        WARNING: If the preprocess_logits_for_metrics function is used,
        we get "predictions" instead of "logits" in the eval_pred tuple.
        """
        predictions, labels = eval_pred

        # if any of the predictions or labels is None, raise an error
        if predictions is None or labels is None:
            raise ValueError('predictions or labels are None')

        # ds_split may be a thunk that defers ordering decisions until after
        # the dataloader has been built (e.g. to align with a length-sorted
        # batch sampler). Resolve once here so verify/preproc see one snapshot.
        resolved_ds_split = ds_split() if callable(ds_split) else ds_split

        # Verify that the labels match the predictions
        verify_labels_match(resolved_ds_split, labels, config) if getattr(
            config.task.preproc_rules, 'verify_labels_match', False
        ) else None

        # Calculate the confusion matrix
        calc_confusion_matrix(predictions, labels, config, eval_path) if getattr(
            config.task.preproc_rules, 'calc_confusion_matrix', False
        ) else None

        # Preprocess predictions and labels if necessary
        predictions, labels = preproc_preds_labels(predictions, labels, config, label_pad_id, tokenizer, resolved_ds_split)

        # DEBUG: show first N preds/labels (decoded if they look like token ids)
        try:
            n = 20
            head_p, head_l = predictions[:n], labels[:n]
            print(f'\n[debug compute_metrics] n_total={len(predictions)} (showing first {n})')
            print(f'  preds  : {list(head_p)}')
            print(f'  labels : {list(head_l)}')
            if hasattr(head_p, 'dtype') and getattr(head_p, 'dtype', None) is not None and 'int' in str(head_p.dtype):
                print(f'  preds  decoded: {[repr(tokenizer.decode([int(x)])) for x in head_p]}')
                print(f'  labels decoded: {[repr(tokenizer.decode([int(x)])) for x in head_l]}')
            print(
                f'  match  : {(head_p == head_l).tolist() if hasattr(head_p, "tolist") else [a == b for a, b in zip(head_p, head_l, strict=True)]}'
            )
        except Exception as e:
            print(f'[debug compute_metrics] print failed: {e}')

        # Compute the metrics
        results = _compute_metrics(predictions, labels, config, ds_split)

        # Postprocess the computed metrics
        results = postproc_metrics(results, config, metric_prefix)

        # print(results); exit()
        return results

    return compute_metrics


def verify_labels_match(ds_split, labels, config):
    """
    Verify that the labels match the dataset split labels
    """
    print('\nVerify compute_metrics labels match with ds_split labels...')

    # Ensure ds_split has 'labels' and it's the same length as labels
    if 'labels' not in ds_split[0]:
        raise ValueError("ds_split does not contain 'labels' key")

    ds_labels = [sample['labels'] for sample in ds_split]
    ds_inputs = [sample['input_ids'] for sample in ds_split]

    if len(ds_labels) != len(labels):
        raise ValueError(f'Mismatch in number of labels: ds_split has {len(ds_labels)}, eval_pred has {len(labels)}')

    for i, (ds_label, ds_input, eval_label) in enumerate(zip(ds_labels, ds_inputs, labels, strict=True)):
        eval_label_list = eval_label.tolist()[: len(ds_label)]

        if not (ds_label == eval_label_list):
            sole_ds_label = [t for t in ds_label if t != -100]
            sole_eval_label = [t for t in eval_label.tolist() if t != -100]

            if sole_ds_label == sole_eval_label:
                raise ValueError(
                    f'Sample index {i} - Labels match but are shifted: ds_label = {ds_label}, eval_label = {eval_label}'
                )
            else:
                raise ValueError(
                    f'Sample index {i} - Label mismatch: sole_ds_label = {sole_ds_label}, sole_eval_label = {sole_eval_label}'
                )

        # Verify answer tokens in labels match corresponding positions in input_ids
        if config.task.category == TaskCatSE.TEXT_GENERATION:
            for j, (lbl, inp) in enumerate(zip(ds_label, ds_input, strict=True)):
                if lbl != -100 and lbl != inp:
                    raise ValueError(f'Sample index {i}, position {j} - Label/input mismatch: label={lbl}, input={inp}')
            # print('Labels match with Input!')


def preproc_preds_labels(predictions, labels, config, label_pad_id, tokenizer, ds_split):
    """Run predictions and labels through ``task.preproc_rules`` before scoring.

    The steps are applied in a fixed order: flatten, drop padded positions, decode
    ids to text, strip and lowercase, then convert label names to floats or ids.
    Each is off unless the rules turn it on, so a task that predicts ids directly
    passes through untouched.

    :returns: the processed ``(predictions, labels)``, grouped by task when
        ``per_task`` is set.
    """
    # Preprocess predictions and labels
    preproc_rules = config.task.preproc_rules
    flatten = preproc_rules.flatten
    filter_padded = preproc_rules.filter_padded
    label_id_to_name = preproc_rules.label_id_to_name
    eval_per_task = getattr(preproc_rules, 'per_task', None)
    label_name_to_id = getattr(preproc_rules, 'label_name_to_id', False)
    label_name_to_float = getattr(preproc_rules, 'label_name_to_float', False)
    label_name_strip_lower = getattr(preproc_rules, 'label_name_strip_lower', False)
    label_pad_id = label_pad_id if label_pad_id is not None else -100
    decode = preproc_rules.decode

    def preproc_1d_preds_labels(predictions, labels, config, label_pad_id, tokenizer):
        # print(predictions.shape, labels.shape, '\n', predictions, labels)

        if decode:
            # Decode
            predictions = batch_decode(predictions, tokenizer, label_pad_id)
            labels = batch_decode(labels, tokenizer, label_pad_id)

            # Strip and lower
            if label_name_strip_lower:
                predictions = [prediction.strip().lower() for prediction in predictions]
                labels = [label.strip().lower() for label in labels]
            # utils.p(predictions, labels)
            # exit()

            # Convert label names to floating numbers or to class IDs
            if label_name_to_float:
                predictions, labels = convert_label_names_to_floats(predictions, labels)

            elif label_name_to_id:
                predictions, labels = convert_label_names_to_ids(predictions, labels)

            # print(predictions.shape, labels.shape, '\n', predictions, '\n', labels)
            # exit()

        else:
            if flatten:
                predictions = predictions.flatten()
                labels = labels.flatten()

            if filter_padded:
                mask = labels != label_pad_id
                labels = labels[mask]
                predictions = predictions[mask]

            if label_id_to_name:
                label_names = np.array(config.ds.label.names)
                predictions = label_names[predictions]
                labels = label_names[labels]

        return predictions, labels

    if eval_per_task:
        # Group predictions and labels by task ID
        predictions, labels = group_preds_labels(predictions, labels, ds_split, eval_per_task)
        # Preprocess predictions and labels for each task
        for task_id in labels.keys():
            # Adjust preprocess config for each task
            label_name_to_id = False if task_id in [4, 5, 'all'] else True
            # Run 1D preprocess for current tasks predictions and labels
            task_preds, task_labels = predictions[task_id], labels[task_id]
            task_preds, task_labels = preproc_1d_preds_labels(task_preds, task_labels, config, label_pad_id, tokenizer)
            predictions[task_id], labels[task_id] = task_preds, task_labels

    elif config.task.category == TaskCatSE.TOKEN_CLASSIFICATION:
        prep_predictions, prep_labels = [], []
        # Run over predictions and labels sentence by sentence
        for sent_preds, sent_labels in zip(predictions, labels, strict=True):
            sent_preds, sent_labels = preproc_1d_preds_labels(sent_preds, sent_labels, config, label_pad_id, tokenizer)
            prep_predictions.append(sent_preds)
            prep_labels.append(sent_labels)
        predictions, labels = prep_predictions, prep_labels

    else:
        predictions, labels = preproc_1d_preds_labels(predictions, labels, config, label_pad_id, tokenizer)

    return predictions, labels


def group_preds_labels(predictions, labels, ds_split, group_by):
    """Split predictions and labels into per-group arrays for multi-task scoring.

    :param group_by: dataset column holding the group id (usually the task id).
    :returns: ``(grouped_preds, grouped_labels)``, each keyed by group id plus an
        ``'all'`` key holding the ungrouped arrays, so overall and per-task metrics
        come from one pass.
    """
    grouped_preds = {}
    grouped_labels = {}
    for p, l, s in zip(predictions, labels, ds_split, strict=True):
        group_id = s[group_by]
        if group_id not in grouped_preds:
            grouped_preds[group_id] = []
            grouped_labels[group_id] = []
        grouped_preds[group_id].append(p)
        grouped_labels[group_id].append(l)
    # Convert lists to np.arrays
    for group_id in grouped_preds:
        grouped_preds[group_id] = np.array(grouped_preds[group_id])
        grouped_labels[group_id] = np.array(grouped_labels[group_id])
    grouped_preds['all'] = np.array(predictions)
    grouped_labels['all'] = np.array(labels)
    return grouped_preds, grouped_labels


def convert_label_names_to_floats(predictions, labels):
    """Parse decoded label strings as floats, for regression-style tasks.

    Note the nested ``string_to_float`` helper is defined but not used: the
    conversion below calls ``float()`` directly, so a prediction that does not
    parse raises rather than falling back to ``-1.0``.
    """
    def string_to_float(string, default=-1.0):
        """Converts string to float, using default when conversion not possible."""
        try:
            return float(string)
        except ValueError:
            return default

    predictions = np.array([float(prediction) for prediction in predictions])
    labels = np.array([float(label) for label in labels])
    return predictions, labels


def convert_label_names_to_ids(predictions, labels):
    """Map decoded label strings to class ids.

    The class list is derived from the *labels*, sorted -- so the id space comes
    from the gold data, and a prediction outside it maps to ``-1`` rather than
    inventing a class. How often that happens is printed, since a high unknown rate
    means the model is not producing label-shaped text at all.
    """
    def name_to_id(string_label, label_classes, default=-1):
        """Returns index of string_label in label_classes or default if not found."""
        if string_label in label_classes:
            return label_classes.index(string_label)
        return default

    # Count and print unknown label predictions
    label_classes = sorted(set(labels))
    count_decoded_unknown_label_predictions(predictions, label_classes)
    predictions = np.array([name_to_id(p, label_classes) for p in predictions])
    labels = np.array([name_to_id(l, label_classes) for l in labels])
    return predictions, labels


def _compute_metrics(predictions, labels, config, ds_split):
    results = {}
    first_metric = config.task.metric_groups[0].metrics[0]
    if first_metric == 'multirc':
        results = compute_multirc(predictions, labels, ds_split)
    if first_metric == 'log_likelihood_accuracy':
        results = compute_log_likelihood_accurac(predictions, labels, config, ds_split)
    else:
        results = compute_metrics_by_metric_groups(predictions, labels, config)
    return results


def compute_metrics_by_metric_groups(predictions, labels, config):
    """Compute every metric group named in ``task.metric_groups``.

    Each group names its metrics by string and they are loaded through
    ``evaluate.combine``; with ``eval.per_task`` set, a group scores only its own
    task's rows and its metric names are prefixed with the task name. Groups whose
    rows are absent are skipped with a message.

    :raises ValueError: if no group produced anything -- an empty metric dict is
        almost always a misconfiguration rather than a real result.
    """
    eval_per_task = getattr(config.eval, 'per_task', None)
    results = {}

    def get_metric_args(metric_group):
        metric_args = {}
        group_preds, group_labels = predictions, labels
        if eval_per_task:
            task_id = metric_group.task.id
            group_preds = predictions[task_id] if task_id in predictions else []
            group_labels = labels[task_id] if task_id in labels else []
        if not (len(group_preds) and len(group_labels)):
            return None
        labels_key = getattr(metric_group, 'labels_key', labels_k)
        predictions_key = getattr(metric_group, 'predictions_key', predictions_k)
        metric_args[predictions_key] = group_preds
        metric_args[labels_key] = group_labels
        metric_args.update({k: v for sn in metric_group.args for k, v in dict(sn).items()}) if hasattr(
            metric_group, 'args'
        ) else None
        # print('metric_group:', metric_group)
        # print('metric_args:', metric_args)
        # exit()
        return metric_args

    for metric_group in config.task.metric_groups:
        metric_args = get_metric_args(metric_group)
        if metric_args:
            metrics = evaluate.combine(metric_group.metrics)
            group_results = metrics.compute(**metric_args)
            if eval_per_task:
                group_results = add_prefix_to_metrics(group_results, f'{metric_group.task.name}/')
            results.update(group_results)
        else:
            print(f'Group {metric_group.task.id} not found in predictions or labels')

    if not results:
        raise ValueError('No metrics computed')

    return results


def postproc_metrics(results, config, add_prefix):
    """Make a metrics dict JSON-safe and prefix its keys.

    numpy arrays become lists and numpy scalars become Python scalars, so the
    result survives being written to disk and logged.
    """
    # cast np.ndarray to list and np.generic to item
    def cast_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.generic):
            return value.item()
        else:
            return value

    results = {key: cast_value(value) for key, value in results.items()}

    # filter the metrics by the prefixes
    filter_by_prefixes = config.task.preproc_rules.filter_by_prefixes
    filter_by_prefixes = tuple(filter_by_prefixes) if filter_by_prefixes else None
    if filter_by_prefixes:
        results = {key: value for key, value in results.items() if key.startswith(filter_by_prefixes)}
        # remove the prefix from the metric names
        for prefix in filter_by_prefixes:
            results = {key.replace(prefix, ''): value for key, value in results.items()}

    # prefixe the metric names with the config name
    if add_prefix:
        results = add_prefix_to_metrics(results, add_prefix)

    return results


def add_prefix_to_metrics(results, prefix):
    """Prefix every metric name, e.g. ``accuracy`` -> ``eval_accuracy``."""
    return {f'{prefix}{name}': value for name, value in results.items()}


def count_decoded_unknown_label_predictions(predictions, label_classes):
    """
    In case of decoded predictions and labels
    """
    total_preds = len(predictions)
    pred_unk_labels = [pred for pred in predictions if pred not in label_classes]
    pred_unk_percentage = (len(pred_unk_labels) / total_preds) * 100 if total_preds > 0 else 0
    print(f'\nLabel Classes: {label_classes}')
    print(f'Predictions > \n Total: {total_preds} \n Unknown: {len(pred_unk_labels)} ({pred_unk_percentage:.4f})%')


# =============================================



def _label_candidate_token_ids(config, tokenizer):
    """Resolve the candidate answer-label token ids from ``config.ds.label.names``.

    ``config.ds.label.names`` is the single source of truth for the label set
    (e.g. ``[A, B, C, D]``); a missing/empty list is an error. Each name must
    tokenize to exactly one token — the bare-letter token the FTP data stores
    as the gold answer (template ends ``Answer：A`` with no leading space, so we
    encode without a space prefix). Returns ``(names, candidate_ids)``.
    """
    label_cfg = getattr(config.ds, 'label', None)
    names = getattr(label_cfg, 'names', None) if label_cfg is not None else None
    if not names:
        raise ValueError('label_restricted_likelihood requires config.ds.label.names to be set')

    candidate_ids = []
    for name in names:
        enc = tokenizer.encode(name, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f'label name {name!r} must encode to a single token, got {enc}')
        candidate_ids.append(enc[0])
    return list(names), candidate_ids


def get_preprocess_logits_for_metrics(config, num_virtual_tokens=None, tokenizer=None):
    """Build the HF Trainer ``preprocess_logits_for_metrics`` hook.

    Reduces raw logits to the prediction shape ``get_compute_metrics`` consumes,
    BEFORE they are cached for the eval loop. The two form a pair: whatever shape
    this emits is exactly what compute_metrics expects to receive.
    """
    print('Get preprocess_logits_for_metrics function...')
    pred_axis = config.task.preproc_rules.prediction_axis
    is_log_likelihood = 'log_likelihood' in config.task.metric_groups[0].metrics[0]
    is_causal_lm = config.task.category == TaskCatSE.TEXT_GENERATION
    # Opt-in (mcqa_ftp): restrict the answer-position argmax to the candidate
    # label tokens (resolved from ds.label.names). The argmax happens here so
    # the eval pipeline never sees vocab-dim or per-candidate intermediates —
    # the returned shape matches the regular causal-LM path.
    label_restricted = getattr(config.task.preproc_rules, 'label_restricted_likelihood', False)
    candidate_ids = _label_candidate_token_ids(config, tokenizer)[1] if label_restricted else None
    # Symmetric half of the runner's shift_labels_by auto-inject: when peft's
    # CausalLM/Seq2SeqLM forward prepends virtual-token labels internally, logits
    # come out at L+n while batch labels stay at L. Trim the prefix here. The
    # shape guard below keeps this a no-op for every other wiring (TokenCls with
    # a shift-labels collator, non-peft, LoRA, etc.) — no task-category branch needed.
    trim_prefix = int(num_virtual_tokens) if num_virtual_tokens else 0

    def preprocess_logits_for_metrics(logits, labels):
        # Handle models like T5 returning multiple outputs
        if isinstance(logits, tuple):
            logits = logits[0]

        if trim_prefix and labels is not None and logits.shape[-2] == labels.shape[-1] + trim_prefix:
            logits = logits[..., trim_prefix:, :]

        if torch.isnan(logits).any():
            print('⚠️ NAN detected in logits!')
        if labels is not None and torch.isnan(labels).any():
            print('⚠️ NAN detected in labels!')

        # Causal shift: logits at position t predict token t+1, so for label at
        # position t we use logits at t-1. All causal-LM branches below operate
        # on this shifted view; the non-causal fallback uses logits as-is.
        if is_causal_lm:
            shift_logits = logits[..., :-1, :]  # (batch, seq-1, vocab)
            shift_labels = labels[..., 1:]      # (batch, seq-1)
        else:
            shift_logits, shift_labels = logits, labels

        if label_restricted:
            # FTP places exactly one answer token per row; argmax over the mask
            # finds it. Restrict the answer-slot argmax to candidate label tokens
            # and return (batch, seq) aligned with the original labels so the
            # regular filter_padded path picks up (pred, label) at the answer slot.
            answer_slot = torch.argmax((shift_labels != -100).to(torch.int), dim=1)  # (batch,)
            rows = torch.arange(shift_logits.shape[0], device=shift_logits.device)
            answer_logits = shift_logits[rows, answer_slot]  # (batch, vocab)
            cand = torch.as_tensor(candidate_ids, device=answer_logits.device)
            best_cand_token = cand[torch.argmax(answer_logits[:, cand], dim=-1)]  # (batch,)
            predictions = torch.zeros_like(labels)
            predictions[rows, answer_slot + 1] = best_cand_token
            return predictions

        if is_log_likelihood:
            # Gather the gold-token log-prob at each non -100 position, then
            # aggregate to (batch, 2) = [sequence_ll, sequence_length].
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            gather_idx = shift_labels.clone()
            gather_idx[gather_idx == -100] = 0  # safe index; masked out below
            label_log_probs = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)
            mask = (shift_labels != -100).float()
            label_log_probs = label_log_probs * mask
            return torch.stack([label_log_probs.sum(dim=-1), mask.sum(dim=-1)], dim=-1)

        if is_causal_lm:
            predictions = torch.argmax(shift_logits, dim=pred_axis).to(torch.long)
            pad = torch.zeros_like(predictions[..., :1])
            return torch.cat([pad, predictions], dim=-1)

        predictions = torch.argmax(logits, dim=pred_axis)
        return predictions.to(torch.long)

    return preprocess_logits_for_metrics

