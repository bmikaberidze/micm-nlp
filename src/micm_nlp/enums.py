"""Every categorical choice the configuration layer recognises.

String enums, so a YAML value compares equal to its member without conversion:
``config.mode == ModeSE.FINETUNE`` works on the raw string ``'finetune'``.

Roughly three families live here — pipeline vocabulary (``ModeSE``, ``DeviceSE``),
dataset and task vocabulary (``DsCatSE``, ``DsTypeSE``, ``DsSplitSE``, ``TaskCatSE``,
``TaskNameSE``, ``EvalTypeSE``), and per-architecture special-token tables
(``BertTokenSE``, ``RobertaTokenSE``, ``XLMRobertaTokenSE``, ``T5TokenSE``, …) that
the tokenizer factory consults when adding special tokens and post-processors.

Note ``model.architecture`` is deliberately *not* validated against ``ModelArchSE``:
it is a free-form string used for run-directory naming.
"""

try:
    from enum import StrEnum

except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042  (enum.StrEnum doesn't exist on Python <3.11)
        pass


# Device types:
class DeviceSE(StrEnum):
    """Where a run executes. ``GPU`` and ``CUDA`` are both accepted spellings."""

    CPU = 'cpu'
    GPU = 'gpu'
    CUDA = 'cuda'


# Config modes:
class ModeSE(StrEnum):
    """What a run does: train from scratch, finetune a checkpoint, evaluate,
    test, clean a workspace, or preprocess a dataset without training.
    """

    TRAIN = 'train'
    FINETUNE = 'finetune'
    EVALUATE = 'evaluate'
    TEST = 'test'
    CLEAN = 'clean'
    PREPROCESS = 'preprocess'


class SentTokTypeSE(StrEnum):
    """Sentence splitter to use. ``KA`` selects
    :class:`~micm_nlp.tokenizers.ka_sen_tok.KaSenTok`, the Georgian splitter; the
    others are NLTK's and spaCy's.
    """

    KA = 'kast'
    NLTK = 'nltkst'
    SPACY = 'spacyst'


class WordTokTypeSE(StrEnum):
    """Word tokenizer to use when a step needs words rather than subwords --
    NLTK's whitespace or punctuation tokenizer.
    """

    NLTK_WHITESPACE = 'nltk_whitespace'
    NLTK_PUNCT = 'nltk_punct'


class TokTypeSE(StrEnum):
    """Subword algorithm to train or load. ``BYTE_LEVEL`` is the ByT5-style
    byte vocabulary; the SentencePiece entries distinguish the native trainer
    from HuggingFace's.
    """

    BPE = 'bpe'
    BYTE_LEVEL = 'byte_level'
    BYTE_LEVEL_BPE = 'byte_level_bpe'
    NATIVE_SENTPIECE = 'native_sentpiece'
    HUGGINGFACE_SENTPIECE = 'huggingface_sentpiece'
    WORDPIECE = 'wordpiece'


class TokAlgSE(StrEnum):
    """Training algorithm for a SentencePiece tokenizer: BPE or unigram."""

    BPE = 'bpe'
    UNIGRAM = 'unigram'


# Dataset splits:
class DsSplitSE(StrEnum):
    """Dataset split names. ``NONE`` is the empty string, for a dataset held as a
    single unsplit table.
    """

    NONE = ''
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'


class DsStateSE(StrEnum):
    """How far a dataset has been processed, and which variant is on disk --
    tokenized, split, subsetted, or filtered by length.
    """

    TOKENIZED = 'tokenized'
    SPLITS = 'splits'
    SUBSET = 'subset'
    SHORT = 'short'
    LONG = 'long'


class SaveDatasetAsSE(StrEnum):
    """On-disk format when writing a dataset: CSV, or HuggingFace's own
    ``save_to_disk`` layout.
    """

    CSV = 'csv'
    HUGGINGFACE = 'huggingface'


# Dataset categories:
class DsCatSE(StrEnum):
    """Top level of the ``artefacts/datasets`` tree: raw text, corpora,
    benchmarks, or collections.
    """

    RAW = 'raw'
    CORPORA = 'corpora'
    BENCHMARKS = 'benchmarks'
    COLLECTIOS = 'collections'


# Dataset types:
class DsTypeSE(StrEnum):
    """How a dataset is loaded. ``HUGGINGFACE`` pulls from the Hub;
    ``HUGGINGFACE_SAVED`` reads a local ``save_to_disk`` directory.
    """

    TEXT = 'text'
    JSON = 'json'
    CSV = 'csv'
    HUGGINGFACE = 'huggingface'
    HUGGINGFACE_SAVED = 'huggingface_saved'


# Model architectures:
class ModelArchSE(StrEnum):
    """Architecture families the tokenizer factory knows how to dress -- it
    selects the special-token table and post-processor.

    Note this is *not* what ``model.architecture`` in a config is validated
    against; that field is free-form and used for run-directory naming.
    """

    BERT = 'bert'
    ROBERTA = 'roberta'
    ELECTRA = 'electra'
    XLNET = 'xlnet'
    XGLM = 'xglm'
    XLMR = 'xlmr'
    AYA = 'aya'
    T5 = 't5'


# Pretrained model or tokenizer sources:
class PretSourceSE(StrEnum):
    """Where a pretrained model or tokenizer comes from: the Hub, or a local path."""

    HUGGINGFACE = 'huggingface'
    LOCAL = 'local'


# Downstream task categories:
class TaskCatSE(StrEnum):
    """Task family, which decides the head and the loss.
    ``TEXT_GENERATION`` is for decoder-only models, ``TEXT_TO_TEXT`` for
    encoder-decoder ones.
    """

    LANGUAGE_MODELING = 'language_modeling'
    TEXT_CLASSIFICATION = 'text_classification'
    TEXT_PAIR_CLASSIFICATION = 'text_pair_classification'
    TOKEN_CLASSIFICATION = 'token_classification'
    STRUCTURAL_ANALYSIS = 'structural_analysis'
    TEXT_SIMILARITIY = 'text_similarity'
    TEXT_GENERATION = 'text_generation'  # decoder-only models
    TEXT_TO_TEXT = 'text_to_text'  # encoder-decoder models


# Downstream tasks:
class TaskNameSE(StrEnum):
    """The concrete task: masked and permutation language modelling, sentiment
    analysis, NER, POS tagging, topic detection.
    """

    MLM = 'mlm'  # Masked Language Modeling
    DMLM = 'dmlm'  # Dynmaic Masked Language Modeling
    SA = 'sa'  # Sentiment Analysis
    NER = 'ner'  # Named Entity Recognition
    POS = 'pos'  # Part-of-Speech Tagging
    PLM = 'plm'  # Permutation Language Modeling
    TOPIC = 'topic'  # Topic Detection


# Huggingface evaluation types:
class EvalTypeSE(StrEnum):
    """What kind of ``evaluate`` object to load. A *metric* scores predictions
    against labels, a *comparison* scores two models against each other, and a
    *measurement* describes a dataset rather than a model.
    """

    # A metric is used to evaluate a model's performance and usually
    # involves the model's predictions as well as some ground truth labels.
    METRIC = 'metric'
    # A comparison is used to compare two models. This can e.g. be done
    # by comparing their predictions to ground truth labels and computing their agreement.
    COMPARISON = 'comparison'
    # With measurements, one can investigate a dataset's properties.
    MEASUREMENT = 'measurement'


# BertTokenizer = { unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]' }
class BertTokenSE(StrEnum):
    """BERT's special tokens.

    One member per role, so the tokenizer factory can ask any of these tables for
    ``BOS``/``EOS``/``PAD``/... without knowing which architecture it is holding.
    :meth:`additional` returns the tokens beyond that common set.
    """

    BOS = '[CLS]'
    EOS = '[SEP]'
    SEP = '[SEP]'
    CLS = '[CLS]'
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'

    @classmethod
    def additional(cls):
        """Special tokens beyond the common BOS/EOS/SEP/CLS/PAD/UNK/MASK set."""
        return []


# ElectraTokenizer = { unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]' }
class ElectraTokenSE(StrEnum):
    """ELECTRA's special tokens -- identical to BERT's.

    One member per role, so the tokenizer factory can ask any of these tables for
    ``BOS``/``EOS``/``PAD``/... without knowing which architecture it is holding.
    :meth:`additional` returns the tokens beyond that common set.
    """

    BOS = '[CLS]'
    EOS = '[SEP]'
    SEP = '[SEP]'
    CLS = '[CLS]'
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'

    @classmethod
    def additional(cls):
        """Special tokens beyond the common BOS/EOS/SEP/CLS/PAD/UNK/MASK set."""
        return []


# RobertaTokenizer = { bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>' }
class RobertaTokenSE(StrEnum):
    """RoBERTa's special tokens.

    One member per role, so the tokenizer factory can ask any of these tables for
    ``BOS``/``EOS``/``PAD``/... without knowing which architecture it is holding.
    :meth:`additional` returns the tokens beyond that common set.
    """

    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'

    @classmethod
    def additional(cls):
        """Special tokens beyond the common BOS/EOS/SEP/CLS/PAD/UNK/MASK set."""
        return []


# XLMRobertaTokenizer = { bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>' }
class XLMRobertaTokenSE(StrEnum):
    """XLM-R's special tokens -- identical to RoBERTa's.

    One member per role, so the tokenizer factory can ask any of these tables for
    ``BOS``/``EOS``/``PAD``/... without knowing which architecture it is holding.
    :meth:`additional` returns the tokens beyond that common set.
    """

    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'

    @classmethod
    def additional(cls):
        """Special tokens beyond the common BOS/EOS/SEP/CLS/PAD/UNK/MASK set."""
        return []


# XLNetTokenizer = { bos_token='<s>', eos_token='</s>', unk_token='<unk>', sep_token='<sep>', pad_token='<pad>', cls_token='<cls>', mask_token='<mask>', additional_special_tokens=['<eop>', '<eod>'] }
class XLNetTokenSE(StrEnum):
    """XLNet's special tokens, including the end-of-paragraph and
    end-of-document markers it adds beyond the usual set.

    One member per role, so the tokenizer factory can ask any of these tables for
    ``BOS``/``EOS``/``PAD``/... without knowing which architecture it is holding.
    :meth:`additional` returns the tokens beyond that common set.
    """

    BOS = '<s>'
    EOS = '</s>'
    SEP = '<sep>'
    CLS = '<cls>'
    UNK = '<unk>'
    PAD = '<pad>'
    MASK = '<mask>'
    EOP = '<eop>'
    EOD = '<eod>'

    @classmethod
    def additional(cls):
        """Special tokens beyond the common BOS/EOS/SEP/CLS/PAD/UNK/MASK set."""
        return [cls.EOP, cls.EOD]


class T5TokenSE(StrEnum):
    """T5's special tokens.

    One member per role, so the tokenizer factory can ask any of these tables for
    ``BOS``/``EOS``/``PAD``/... without knowing which architecture it is holding.
    :meth:`additional` returns the tokens beyond that common set.
    """

    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'

    @classmethod
    def additional(cls):
        """Special tokens beyond the common BOS/EOS/SEP/CLS/PAD/UNK/MASK set."""
        return []
