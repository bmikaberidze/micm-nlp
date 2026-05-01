import evaluate
import numpy as np

from micm_nlp.enums import TaskCatSE
from micm_nlp.evals.metrics.log_likelihood import compute_log_likelihood_accurac
from micm_nlp.evals.metrics.multirc import compute_multirc
from micm_nlp.evals.plot import calc_confusion_matrix
from micm_nlp.tokenizers.decoding import batch_decode

labels_k = 'references'
predictions_k = 'predictions'


def get_compute_metrics(config, label_pad_id, metric_prefix, eval_path, tokenizer, ds_split):

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

        # Verify that the labels match the predictions
        verify_labels_match(ds_split, labels, config) if getattr(
            config.task.preproc_rules, 'verify_labels_match', False
        ) else None

        # Calculate the confusion matrix
        calc_confusion_matrix(predictions, labels, config, eval_path) if getattr(
            config.task.preproc_rules, 'calc_confusion_matrix', False
        ) else None

        # Preprocess predictions and labels if necessary
        predictions, labels = preproc_preds_labels(predictions, labels, config, label_pad_id, tokenizer, ds_split)

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
