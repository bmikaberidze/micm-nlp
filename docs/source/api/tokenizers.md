# Tokenizers

| Module | Role |
|---|---|
| {doc}`tokenizer </autoapi/micm_nlp/tokenizers/tokenizer/index>` | `AutoTokenizer` factory; special tokens and post-processors |
| {doc}`xlm_roberta </autoapi/micm_nlp/tokenizers/xlm_roberta/index>` | XLM-R tokenizer adapted to a target architecture |
| {doc}`decoding </autoapi/micm_nlp/tokenizers/decoding/index>` | Label-aware `decode` / `batch_decode` |
| {doc}`bert_byt5 </autoapi/micm_nlp/tokenizers/bert_byt5/index>` | `BertByT5Tokenizer` — byte-level tokenizer with BERT-style special tokens |
| {doc}`ka_sen_tok </autoapi/micm_nlp/tokenizers/lib/sent/ka_sen_tok/index>` | `KaSenTok` — Georgian sentence tokenizer |

`bert_byt5` and `ka_sen_tok` are the tokenizers compared in *A Comparison of
Different Tokenization Methods for the Georgian Language*
([ICNLSP 2024](https://aclanthology.org/2024.icnlsp-1.22/)).

```{toctree}
:hidden:

/autoapi/micm_nlp/tokenizers/tokenizer/index
/autoapi/micm_nlp/tokenizers/xlm_roberta/index
/autoapi/micm_nlp/tokenizers/decoding/index
/autoapi/micm_nlp/tokenizers/bert_byt5/index
/autoapi/micm_nlp/tokenizers/lib/sent/ka_sen_tok/index
```
