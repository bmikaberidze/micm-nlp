# string_f1.py
import re
import string

import evaluate
from datasets import Features, Value

_CITATION = ''
_DESCRIPTION = 'String-level F1 for QA-style span overlap'


def normalize_answer(s):
    """Lowercase, remove punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _CITATION)
class StringF1(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description='',
            features=Features(
                {
                    'predictions': Value('string'),
                    'references': Value('string'),
                }
            ),
            reference_urls=[],
        )

    def _compute(self, predictions, references):
        scores = [f1_score(p, r) for p, r in zip(predictions, references, strict=True)]
        return {'f1': sum(scores) / len(scores)}
