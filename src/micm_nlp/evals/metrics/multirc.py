import evaluate


def compute_multirc(predictions, labels, ds_split):

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
