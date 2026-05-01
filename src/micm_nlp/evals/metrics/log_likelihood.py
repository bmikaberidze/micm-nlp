import numpy as np


def compute_log_likelihood_accurac(predictions, labels, config, ds_split):

    ll_args = config.task.metric_groups[0].args
    group_by = ll_args.group_by
    correct_flag = ll_args.correct_flag

    def compute_mcqa_accuracy(log_likelihoods, group_ids, correct_flags):

        # print(log_likelihoods.shape, group_ids.shape, correct_flags.shape)
        questions_max_ll = {}
        for ll, gid, cf in zip(log_likelihoods, group_ids, correct_flags, strict=True):
            if gid not in questions_max_ll or ll > questions_max_ll[gid][0]:
                questions_max_ll[gid] = (ll, cf)

        # print(questions_max_ll)
        accuracy = sum([1 for _, cf in questions_max_ll.values() if cf]) / len(questions_max_ll)
        return accuracy

    group_ids = [sample[group_by] for sample in ds_split]  # or 'question_id'
    correct_flags = [sample[correct_flag] for sample in ds_split]  # or derive from 'correct_idx'

    sequence_ll = predictions[:, 0]  # (n_samples,)
    sequence_lengths = predictions[:, 1]  # (n_samples,)
    normalized_ll = sequence_ll / np.maximum(sequence_lengths, 1)

    accuracy = compute_mcqa_accuracy(
        log_likelihoods=normalized_ll,  # or sequence_ll for unnormalized
        group_ids=group_ids,
        correct_flags=correct_flags,
    )

    results = {
        'accuracy': accuracy,
        'mean_ll': float(np.mean(sequence_ll)),
        'mean_normalized_ll': float(np.mean(normalized_ll)),
    }

    return results
