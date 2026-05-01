try:
    from enum import StrEnum

except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042  (enum.StrEnum doesn't exist on Python <3.11)
        pass


# Device types:
class DeviceSE(StrEnum):
    CPU = 'cpu'
    GPU = 'gpu'
    CUDA = 'cuda'


# Config modes:
class ModeSE(StrEnum):
    TRAIN = 'train'
    FINETUNE = 'finetune'
    EVALUATE = 'evaluate'
    TEST = 'test'
    CLEAN = 'clean'
    PREPROCESS = 'preprocess'


class SentTokTypeSE(StrEnum):
    KA = 'kast'
    NLTK = 'nltkst'
    SPACY = 'spacyst'


class WordTokTypeSE(StrEnum):
    NLTK_WHITESPACE = 'nltk_whitespace'
    NLTK_PUNCT = 'nltk_punct'


class TokTypeSE(StrEnum):
    BPE = 'bpe'
    BYTE_LEVEL = 'byte_level'
    BYTE_LEVEL_BPE = 'byte_level_bpe'
    NATIVE_SENTPIECE = 'native_sentpiece'
    HUGGINGFACE_SENTPIECE = 'huggingface_sentpiece'
    WORDPIECE = 'wordpiece'


class TokAlgSE(StrEnum):
    BPE = 'bpe'
    UNIGRAM = 'unigram'


# Dataset splits:
class DsSplitSE(StrEnum):
    NONE = ''
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'


class DsStateSE(StrEnum):
    TOKENIZED = 'tokenized'
    SPLITS = 'splits'
    SUBSET = 'subset'
    SHORT = 'short'
    LONG = 'long'


class SaveDatasetAsSE(StrEnum):
    CSV = 'csv'
    HUGGINGFACE = 'huggingface'


# Dataset categories:
class DsCatSE(StrEnum):
    RAW = 'raw'
    CORPORA = 'corpora'
    BENCHMARKS = 'benchmarks'
    COLLECTIOS = 'collections'


# Dataset types:
class DsTypeSE(StrEnum):
    TEXT = 'text'
    JSON = 'json'
    CSV = 'csv'
    HUGGINGFACE = 'huggingface'
    HUGGINGFACE_SAVED = 'huggingface_saved'


# Model architectures:
class ModelArchSE(StrEnum):
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
    HUGGINGFACE = 'huggingface'
    LOCAL = 'local'


# Downstream task categories:
class TaskCatSE(StrEnum):
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
    MLM = 'mlm'  # Masked Language Modeling
    DMLM = 'dmlm'  # Dynmaic Masked Language Modeling
    SA = 'sa'  # Sentiment Analysis
    NER = 'ner'  # Named Entity Recognition
    POS = 'pos'  # Part-of-Speech Tagging
    PLM = 'plm'  # Permutation Language Modeling
    TOPIC = 'topic'  # Topic Detection


# Huggingface evaluation types:
class EvalTypeSE(StrEnum):
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
    BOS = '[CLS]'
    EOS = '[SEP]'
    SEP = '[SEP]'
    CLS = '[CLS]'
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'

    @classmethod
    def additional(cls):
        return []


# ElectraTokenizer = { unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]' }
class ElectraTokenSE(StrEnum):
    BOS = '[CLS]'
    EOS = '[SEP]'
    SEP = '[SEP]'
    CLS = '[CLS]'
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'

    @classmethod
    def additional(cls):
        return []


# RobertaTokenizer = { bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>' }
class RobertaTokenSE(StrEnum):
    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'

    @classmethod
    def additional(cls):
        return []


# XLMRobertaTokenizer = { bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>' }
class XLMRobertaTokenSE(StrEnum):
    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'

    @classmethod
    def additional(cls):
        return []


# XLNetTokenizer = { bos_token='<s>', eos_token='</s>', unk_token='<unk>', sep_token='<sep>', pad_token='<pad>', cls_token='<cls>', mask_token='<mask>', additional_special_tokens=['<eop>', '<eod>'] }
class XLNetTokenSE(StrEnum):
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
        return [cls.EOP, cls.EOD]


class T5TokenSE(StrEnum):
    BOS = '<s>'
    EOS = '</s>'
    SEP = '</s>'
    CLS = '<s>'
    PAD = '<pad>'
    UNK = '<unk>'
    MASK = '<mask>'

    @classmethod
    def additional(cls):
        return []
