"""Tokenizer classes that HuggingFace does not ship.

Two tokenizers that exist because a published experiment needed them, both built by
wrapping a stock HuggingFace tokenizer rather than by training a new one:

``BertByT5Tokenizer``
    A byte-level ByT5 vocabulary wearing BERT's special tokens, so byte-level
    segmentation can drive an encoder-only BERT-style model. From the Georgian
    tokenization comparison (`ICNLSP 2024
    <https://aclanthology.org/2024.icnlsp-1.22/>`_), where byte-level segmentation was
    one of the methods evaluated.

``CustomXlmRoberta``
    XLM-R's multilingual SentencePiece vocabulary re-dressed with the special tokens
    and post-processor of a different target architecture.

This module mirrors :mod:`micm_nlp.models.architectures` and stays small for the same
reason: a tokenizer that can be selected by name through ``tokenizer.cls`` needs no
code here.
"""

from transformers import AutoTokenizer, BertTokenizerFast, ByT5Tokenizer

from micm_nlp.enums import ModelArchSE


class BertByT5Tokenizer(ByT5Tokenizer):
    """ """

    bert_tok_name = 'bert-base-uncased'

    def __init__(self, byt5_name='google/byt5-small', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.byt5 = ByT5Tokenizer.from_pretrained(byt5_name)
        self.bert = BertTokenizerFast.from_pretrained(self.bert_tok_name)

        from micm_nlp.tokenizers.tokenizer import add_special_tokens

        add_special_tokens(self.byt5, ModelArchSE.BERT)

        self.byt5.build_inputs_with_special_tokens = self.bert.build_inputs_with_special_tokens

        self.__dict__.update(self.byt5.__dict__)

        # print(self.get_vocab())
        # print()
        # # self._vocab_size = len(self.get_vocab())
        # print(self.get_vocab())
        # print()
        # exit()

        # print(self.mask_token, self.sep_token, self.cls_token, self.pad_token, self.unk_token)
        # print(self.mask_token_id, self.sep_token_id, self.cls_token_id, self.pad_token_id, self.unk_token_id)
        # exit()

    # @property
    # def vocab_size(self):
    #     if hasattr(self, '_vocab_size'):
    #         return self._vocab_size
    #     return self._utf_vocab_size


class CustomXlmRoberta:
    """ """

    hf_name = 'xlm-roberta-base'

    def __init__(self, model_arch=ModelArchSE.BERT):

        self.xlmr = AutoTokenizer.from_pretrained(self.hf_name)

        from micm_nlp.tokenizers.tokenizer import add_post_processor, add_special_tokens

        add_special_tokens(self.xlmr, model_arch)
        add_post_processor(self.xlmr, model_arch)

        self.__dict__.update(self.xlmr.__dict__)
