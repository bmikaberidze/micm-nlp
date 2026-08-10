"""XLM-RoBERTa's tokenizer re-dressed for another architecture.

``CustomXlmRoberta`` loads ``xlm-roberta-base``'s tokenizer, then applies the special
tokens and post-processor of a *different* target architecture (BERT by default), so
XLM-R's multilingual SentencePiece vocabulary can feed a model that expects another
family's input format.
"""

from transformers import AutoTokenizer

from micm_nlp.enums import ModelArchSE


class CustomXlmRoberta:
    """ """

    hf_name = 'xlm-roberta-base'

    def __init__(self, model_arch=ModelArchSE.BERT):

        self.xlmr = AutoTokenizer.from_pretrained(self.hf_name)

        from micm_nlp.tokenizers.tokenizer import add_post_processor, add_special_tokens

        add_special_tokens(self.xlmr, model_arch)
        add_post_processor(self.xlmr, model_arch)

        self.__dict__.update(self.xlmr.__dict__)
