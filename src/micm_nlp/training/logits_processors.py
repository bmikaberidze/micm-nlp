import torch
from transformers import LogitsProcessor


class ConstrainedPrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_texts):
        self.tokenizer = tokenizer
        self.allowed_token_seqs = [
            [tokenizer.pad_token_id, *tokenizer.encode(text, add_special_tokens=False)] for text in allowed_texts
        ]
        self._cached_prefix_len = None  # inferred on first step

    def __call__(self, input_ids, scores):
        batch_size, cur_len = input_ids.shape
        scores.shape[-1]

        if self._cached_prefix_len is None:
            # Detect decoder prefix length (e.g. 1 for <pad>)
            self._cached_prefix_len = cur_len - 1

        prefix_len = self._cached_prefix_len
        gen_parts = input_ids[:, prefix_len:]  # [B, generated_len]

        # === Result scores: all -inf by default ===
        filtered_scores = torch.full_like(scores, float('-inf'))

        for b in range(batch_size):
            generated = gen_parts[b].tolist()

            # Find completions that still match the generated prefix
            matching = [seq for seq in self.allowed_token_seqs if seq[: len(generated)] == generated]

            if not matching:
                # No matches → only allow EOS
                filtered_scores[b, self.tokenizer.eos_token_id] = 0
                continue

            allowed_next = set()
            for seq in matching:
                if len(seq) > len(generated):
                    allowed_next.add(seq[len(generated)])
                else:
                    allowed_next.add(self.tokenizer.eos_token_id)

            for tok_id in allowed_next:
                filtered_scores[b, tok_id] = scores[b, tok_id]

        return filtered_scores
