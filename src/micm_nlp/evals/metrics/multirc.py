"""MultiRC scoring through the official SuperGLUE scorer.

``evaluate.load('super_glue', 'multirc')`` will not accept a flat prediction array:
each prediction must be tagged with its ``(paragraph, question, answer)`` index
triple. This module rebuilds those triples from the ``idx/paragraph``,
``idx/question`` and ``idx/answer`` columns of ``ds_split``, coercing predictions and
labels to 0/1 along the way.
"""

import evaluate


def compute_multirc(predictions, labels, ds_split):
    """Score MultiRC with the official SuperGLUE metric.

    ``evaluate.load('super_glue', 'multirc')`` will not take a flat prediction
    array: MultiRC scores per question, so each prediction has to carry the
    paragraph/question/answer indices it belongs to. This rebuilds that structure
    from ``ds_split``'s ``idx/*`` columns and coerces predictions and labels to
    0/1, accepting either booleans or the strings ``'true'``/``'1'``.

    :param predictions: one prediction per answer row.
    :param labels: gold labels, same order.
    :param ds_split: the split being scored; supplies ``idx/paragraph``,
        ``idx/question`` and ``idx/answer``.
    :returns: the official scorer's output (exact match and F1a).
    :raises Exception: re-raised after printing the offending sample, if a row is
        missing its index columns.
    """
    multirc_metric = evaluate.load('super_glue', 'multirc')

    structured_preds = []
    references = []

    def to_int_bool(x):
        if isinstance(x, str):
            return int(x.lower().strip() in ['true', '1'])
        return int(bool(x))

    for pred, label, sample in zip(predictions, labels, ds_split, strict=True):
        try:
            structured_preds.append(
                {
                    'idx': {
                        'question': int(sample['idx/question']),
                        'paragraph': int(sample['idx/paragraph']),
                        'answer': int(sample['idx/answer']),
                    },
                    'prediction': to_int_bool(pred),
                }
            )
            references.append(to_int_bool(label))
        except Exception as e:
            print(f'[Error] Invalid sample format in ds_split: {sample["idx"] if "idx" in sample else sample}')
            raise e

    # Run the official scorer
    results = multirc_metric.compute(predictions=structured_preds, references=references)
    return results
