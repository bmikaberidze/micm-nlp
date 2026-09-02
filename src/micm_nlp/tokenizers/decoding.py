"""Label-aware decoding helpers.

``decode`` and ``batch_decode`` drop every position equal to ``label_pad_id`` before
handing token ids to the tokenizer, so the ignore index used in label tensors
(``-100`` by convention) never reaches it and never shows up as text.
"""


def decode(texts, tokenizer, label_pad_id, skip_special_tokens=True):
    """Decode one label sequence, dropping the padding the loss ignored.

    Label tensors are padded with ``label_pad_id`` (``-100`` by convention) rather
    than the tokenizer's pad id, so they cannot be handed to ``tokenizer.decode``
    directly -- the padding is not a valid token id. This masks it out first.

    :param texts: a single 1-D tensor of token ids.
    :param tokenizer: tokenizer to decode with.
    :param label_pad_id: the id used as label padding.
    :param skip_special_tokens: passed through to the tokenizer.
    :returns: the decoded string, stripped.
    """
    mask = texts != label_pad_id
    masked = texts[mask]  # texts is assumed to be a single tensor (1D)
    decoded = tokenizer.decode(masked, skip_special_tokens=skip_special_tokens)
    return decoded.strip()


def batch_decode(texts, tokenizer, label_pad_id, skip_special_tokens=True):
    """Batched :func:`decode`: one string per row, each stripped.

    :param texts: 2-D tensor of token ids, one label sequence per row.
    :param tokenizer: tokenizer to decode with.
    :param label_pad_id: the id used as label padding.
    :param skip_special_tokens: passed through to the tokenizer.
    :returns: one decoded string per row.
    """
    mask = texts != label_pad_id
    masked = [t[m] for t, m in zip(texts, mask, strict=True)]
    decoded = tokenizer.batch_decode(masked, skip_special_tokens=skip_special_tokens)
    return [d.strip() for d in decoded]
