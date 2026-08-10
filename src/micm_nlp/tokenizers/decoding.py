"""Label-aware decoding helpers.

``decode`` and ``batch_decode`` drop every position equal to ``label_pad_id`` before
handing token ids to the tokenizer, so the ignore index used in label tensors
(``-100`` by convention) never reaches it and never shows up as text.
"""


def decode(texts, tokenizer, label_pad_id, skip_special_tokens=True):
    mask = texts != label_pad_id
    masked = texts[mask]  # texts is assumed to be a single tensor (1D)
    decoded = tokenizer.decode(masked, skip_special_tokens=skip_special_tokens)
    return decoded.strip()


def batch_decode(texts, tokenizer, label_pad_id, skip_special_tokens=True):
    mask = texts != label_pad_id
    masked = [t[m] for t, m in zip(texts, mask, strict=True)]
    decoded = tokenizer.batch_decode(masked, skip_special_tokens=skip_special_tokens)
    return [d.strip() for d in decoded]
