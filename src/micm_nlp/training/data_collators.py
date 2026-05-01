from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import (
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    XLNetTokenizer,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

try:
    from transformers.tokenization_utils_base import PaddingStrategy
except ImportError:
    from transformers.utils import PaddingStrategy  # older transformers versions

from micm_nlp.datasets.dataset import DATASET


# Pad sequences dynamically within the batch
class DataCollatorForPLMWithPadding(DataCollatorForPermutationLanguageModeling):
    def __init__(
        self,
        tokenizer: XLNetTokenizer,
        max_length: int = 128,
        padding: str = 'max_length',  # max_length | true (longest) | false (do_not_pad)
        plm_probability: float = 1 / 6,
        max_span_length: int = 5,  # maximum length of a span of masked tokens
        return_tensors: str = 'pt',
    ):
        super().__init__(
            tokenizer=tokenizer,
            plm_probability=plm_probability,
            max_span_length=max_span_length,
            return_tensors=return_tensors,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors

    def torch_call(self, features):

        # Set padding to the right
        self.tokenizer.padding_side = 'right'

        # Use tokenizer's pad method to pad sequences dynamically within the batch
        batch = self.tokenizer.pad(
            features, padding=self.padding, max_length=self.max_length, return_tensors=self.return_tensors
        )

        # Convert padded batch back into a list of dictionaries
        batch = [
            {'input_ids': input_ids, 'attention_mask': attention_mask}
            for input_ids, attention_mask in zip(batch['input_ids'], batch['attention_mask'], strict=True)
        ]
        # print(batch[0])
        # exit()

        # Generate perm_mask and target_mapping using the inherited functionality
        batch = super().torch_call(batch)

        return batch


class DataCollatorTaskIDDecorator:
    """Decorator for adding task_ids to any data collator."""

    def __init__(self, base_collator, task_id):
        self.base_collator = base_collator  # Wrap any existing DataCollator
        self.task_id = task_id

    def __call__(self, batch):
        # Use the original collator to process the batch
        print(batch[0])
        batch = self.base_collator(batch)
        exit()
        try:
            # Detect the format of input_ids (e.g., device, tensor type)
            input_ids = batch[DATASET.keys.input_ids]

            # Create task_ids tensor with the same properties as input_ids
            task_ids = torch.tensor(
                [self.task_id] * len(input_ids),  # Repeat task_id for each example
                device=input_ids.device,  # Match the device (CPU/GPU)
                dtype=torch.long,  # Use long dtype for task IDs
            )

            # Add task_ids to the batch
            batch[DATASET.keys.task_ids] = task_ids

        except KeyError as e:
            # Handle missing "input_ids" key in the batch
            raise ValueError(f"Required key 'input_ids' is missing in the batch: {e}") from e

        return batch


class DataCollatorForSeq2SeqWithShiftLabels(DataCollatorForSeq2Seq):
    def __init__(self, *args, shift_labels_by=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_labels_by = shift_labels_by

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)

        if self.shift_labels_by and 'labels' in batch:
            labels = batch['labels']
            pad_value = -100
            pad_shape = (labels.shape[0], self.shift_labels_by)
            soft_prompt_pad = torch.full(pad_shape, pad_value, dtype=labels.dtype, device=labels.device)
            batch['labels'] = torch.cat([soft_prompt_pad, labels], dim=1)

        return batch


@dataclass
class CustomDataCollatorForSeq2Seq_2(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Any | None = None
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str = 'pt'

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # for label in labels:
        #     print(label)
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch['labels'] = list(labels)
                else:
                    batch['labels'] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch['labels'] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == 'right'
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch['labels'] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == 'right'
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]
        # for label in batch["labels"]:
        #     print(label)
        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get('labels', None) is not None:
            if return_tensors == 'pt':
                import torch

                batch['labels'] = torch.tensor(batch['labels'], dtype=torch.int64)
            elif return_tensors == 'tf':
                import tensorflow as tf

                batch['labels'] = tf.constant(batch['labels'], dtype=tf.int64)
            else:
                batch['labels'] = np.array(batch['labels'], dtype=np.int64)
        else:
            batch['labels'] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, 'prepare_decoder_input_ids_from_labels')
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch['labels'])
            batch['decoder_input_ids'] = decoder_input_ids

        # for label in batch["labels"]:
        #     print(label)

        # for decoder_input_id in batch["decoder_input_ids"]:
        #     print(decoder_input_id)
        # print("\n=== COLLATOR DEBUG ===")
        # print(f"Has labels: {batch.get('labels') is not None}")
        # print(f"Batch size: {len(batch['input_ids'])}")
        # print(f"Label sample shape: {batch['labels'].shape if batch.get('labels') is not None else 'None'}")

        return batch


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = 'pt'

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if 'label' in batch:
            batch['labels'] = batch['label']
            del batch['label']
        if 'label_ids' in batch:
            batch['labels'] = batch['label_ids']
            del batch['label_ids']
        return batch
