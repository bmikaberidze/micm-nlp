import csv
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm

import micm_nlp.path as path
import micm_nlp.utils as utils
from micm_nlp.enums import (
    DsSplitSE,
    DsStateSE,
    DsTypeSE,
    PretSourceSE,
    SaveDatasetAsSE,
    TaskCatSE,
)
from micm_nlp.tokenizers import tokenizer as tok_module


class DATASET:
    seed = 42

    keys = SimpleNamespace()
    keys.task_ids = 'task_ids'
    keys.targets = 'targets'
    keys.labels = 'labels'
    keys.inputs = 'inputs'
    keys.input_ids = 'input_ids'
    keys.attention_mask = 'attention_mask'
    keys.special_tokens_mask = 'special_tokens_mask'
    keys.length = 'length'

    # Dataset types map to file extensions:
    FILE_EXT_MAP: ClassVar[dict] = {
        DsTypeSE.CSV: 'csv',
        DsTypeSE.TEXT: 'txt',
        DsTypeSE.JSON: 'json',
    }

    def __init__(self, config, hf_datasets=None):

        if hf_datasets is None:
            hf_datasets = []
        self._config = config
        self._hf_datasets = hf_datasets

        self._load()
        self._print_details()

    @property
    def hf(self):
        return self._hf

    @hf.setter
    def hf(self, hf_dataset):
        self._hf = hf_dataset
        self._extract_splits_form_hf(hf_dataset)

    def _get_split(self, split_name):
        if split_name == DsSplitSE.TRAIN:
            return self.train
        elif split_name == DsSplitSE.TEST:
            return self.test
        elif split_name == DsSplitSE.VALIDATION:
            return self.validation
        else:
            raise ValueError(f'Invalid split name: {split_name}')

    def _set_split(self, split_name, split):
        if split_name == DsSplitSE.TRAIN:
            self.train = split
        elif split_name == DsSplitSE.TEST:
            self.test = split
        elif split_name == DsSplitSE.VALIDATION:
            self.validation = split
        else:
            raise ValueError(f'Invalid split name: {split_name}')

    def _extract_splits_form_hf(self, hf_dataset):
        def get_split(split_key):
            split = self.split_map[split_key]
            return hf_dataset[split] if split in hf_dataset else None

        self.train = get_split(DsSplitSE.TRAIN)
        self.test = get_split(DsSplitSE.TEST)
        self.validation = get_split(DsSplitSE.VALIDATION)

    def _bundle_hf_dataset_dict(self):
        dataset = {self.split_map[DsSplitSE.TRAIN]: self.train}
        if self.test:
            dataset[self.split_map[DsSplitSE.TEST]] = self.test
        if self.validation:
            dataset[self.split_map[DsSplitSE.VALIDATION]] = self.validation
        self._hf = DatasetDict(dataset)

    def _set_keys(self):
        """
        Set Keys
        """
        self.inputs_key = self._config.ds.input.key
        self.labels_key = self._config.ds.label.key if self._config.ds.label else None
        self.task_ids_key = self._config.ds.task_id.key if self._config.ds.task_id else None

    def _set_dir(self):
        """
        Set Dir
        """
        self._dir = f'{path.datasets_dir()}/{self._config.ds.category}/{self._config.ds.dirs}'

    def _set_name(self):
        """
        Set Name
        """
        self.name = self.get_name(self._config)

    def _load(self):
        """
        Load Dataset
        """
        self._set_keys()
        self._set_dir()
        self._set_name()
        ds_conf = self._config.ds
        with_splits = ds_conf.comes_with_splits
        self.split_map = {
            DsSplitSE.NONE: DsSplitSE.TRAIN,
            DsSplitSE.TRAIN: DsSplitSE.TRAIN if isinstance(with_splits.train, bool) else with_splits.train,
            DsSplitSE.TEST: DsSplitSE.TEST if isinstance(with_splits.test, bool) else with_splits.test,
            DsSplitSE.VALIDATION: DsSplitSE.VALIDATION
            if isinstance(with_splits.validation, bool)
            else with_splits.validation,
        }

        def load(ds_name):
            hf_dataset = None
            data_files = {}
            if ds_conf.type == DsTypeSE.HUGGINGFACE_SAVED:
                print(f'Load Dataset From: {self._dir}')
                hf_dataset = load_from_disk(self._dir)

            elif ds_conf.type == DsTypeSE.HUGGINGFACE:
                print(f'Load Dataset From HuggingFace: {ds_conf.dirs} {"/" + ds_name if ds_name else ""}')
                hf_dataset = load_dataset(ds_conf.dirs, ds_name) if ds_name else load_dataset(ds_conf.dirs)

            else:
                file_ext = self.FILE_EXT_MAP[ds_conf.type]

                def add_data_file(split_key):
                    ds_key = self.split_map[split_key]
                    ds_suff = '' if split_key == DsSplitSE.NONE else f'_{split_key}'
                    ds_file_path = f'{self._dir}/{ds_name}{ds_suff}.{file_ext}'
                    data_files[ds_key] = ds_file_path
                    print(f'Load Dataset From: {ds_file_path}')

                if not ds_conf.comes_with_splits.train:
                    add_data_file(DsSplitSE.NONE)
                else:
                    add_data_file(DsSplitSE.TRAIN)
                if ds_conf.comes_with_splits.test:
                    add_data_file(DsSplitSE.TEST)
                if ds_conf.comes_with_splits.validation:
                    add_data_file(DsSplitSE.VALIDATION)

                hf_dataset = load_dataset(ds_conf.type, data_files=data_files)

            return hf_dataset, data_files

        def add_task_id(dataset, task_id):
            if task_id is None:
                return dataset
            # task_id = int(task_id)
            return dataset.map(
                lambda batch: {
                    self.task_ids_key: [task_id]
                    *
                    # Get the length of the first field in the batch.
                    len(batch[next(iter(batch))])
                },
                batched=True,
            )

        data_files = {}
        task_id = getattr(self._config.task, 'id', None)

        if self._hf_datasets:
            # Concatenate all datasets
            hf_dataset = self.get_concatenated_dataset(self._hf_datasets)

        elif isinstance(ds_conf.name, str) or ds_conf.name is None:
            hf_dataset, data_files = load(ds_conf.name)
            if task_id is not None:
                hf_dataset = add_task_id(hf_dataset, task_id)

        else:
            # Load all datasets
            for ds_name in ds_conf.name:
                hf_dataset, _ = load(ds_name)
                if task_id is not None:
                    hf_dataset = add_task_id(hf_dataset, task_id)
                    task_id += 1
                self._hf_datasets.append(hf_dataset)
            # Concatenate all datasets
            hf_dataset = self.get_concatenated_dataset(self._hf_datasets)

        print('\nRaw Dataset Detaild >')
        print(hf_dataset)
        # exit()

        # Standardize input, labels, and task_ids keys
        hf_dataset = self._standardize_keys(hf_dataset)

        # Sets hf dataset and runs self._extract_splits_form_hf(dataset)
        self.hf = hf_dataset

        # In case of K-Fold Cross-Validation
        if hasattr(ds_conf, 'cross_validation') and ds_conf.type != DsTypeSE.HUGGINGFACE_SAVED:
            self._setup_cross_validation(data_files)

    def _standardize_keys(self, hf_dataset):

        def rename_in_all_splits(old_key, new_key):
            dataset_dict = {}
            for split, ds in hf_dataset.items():
                # First rename new_key if it already exists
                if new_key in ds.features:
                    ds = ds.rename_column(new_key, f'old_{new_key}')
                dataset_dict[split] = ds.rename_column(old_key, new_key)
            return DatasetDict(dataset_dict)

        first_split = next(iter(hf_dataset.keys()))
        features = hf_dataset[first_split].features
        if (
            self.inputs_key in features
            and self.inputs_key != self.keys.inputs
            and getattr(self._config.ds.input, 'standardize_key', False)
        ):
            hf_dataset = rename_in_all_splits(self.inputs_key, self.keys.inputs)
            self.inputs_key = self.keys.inputs

        if (
            self.labels_key in features
            and self.labels_key != self.keys.labels
            and getattr(self._config.ds.label, 'standardize_key', False)
        ):
            hf_dataset = rename_in_all_splits(self.labels_key, self.keys.labels)
            self.labels_key = self.keys.labels

        if (
            self.task_ids_key in features
            and self.task_ids_key != self.keys.task_ids
            and getattr(self._config.ds.task_id, 'standardize_key', False)
        ):
            hf_dataset = rename_in_all_splits(self.task_ids_key, self.keys.task_ids)
            self.task_ids_key = self.keys.task_ids

        return hf_dataset

    def _setup_cross_validation(self, data_files):
        ds_conf = self._config.ds
        # In case of K-Fold Cross-Validation
        if hasattr(ds_conf, 'cross_validation') and ds_conf.type != DsTypeSE.HUGGINGFACE_SAVED:
            folds = ds_conf.cross_validation.folds
            step = int(100 // folds)
            train_df = {'train': data_files[self.split_map[DsSplitSE.TRAIN]]}
            self.train_folds = load_dataset(
                ds_conf.type,
                data_files=train_df,
                split=[f'train[:{k}%]+train[{k + step}%:]' for k in range(0, 100, step)],
            )
            if not ds_conf.cross_validation.iterate_test_split:
                self.validation_folds = load_dataset(
                    ds_conf.type, data_files=train_df, split=[f'train[{k}%:{k + step}%]' for k in range(0, 100, step)]
                )
            else:
                half_step = int(step / 2)
                self.validation_folds = load_dataset(
                    ds_conf.type,
                    data_files=train_df,
                    split=[f'train[{k}%:{k + half_step}%]' for k in range(0, 100, step)],
                )
                self.test_folds = load_dataset(
                    ds_conf.type,
                    data_files=train_df,
                    split=[f'train[{k + half_step}%:{k + step}%]' for k in range(0, 100, step)],
                )

    def _set_paths(self):
        """
        Set Paths
        """
        if DsTypeSE.HUGGINGFACE_SAVED == self._config.ds.type:
            self._save_path = self._dir
            self._tokenized_path = self._save_path
        else:
            self._save_path = f'{self._dir}/{self.name}'
            tok_dirs = ''
            tok_type = self._config.tokenizer.type
            tok_source = self._config.tokenizer.source
            if tok_source == PretSourceSE.LOCAL:
                tok_dirs = tok_type + self._tokenizer.path.split(tok_type)[-1]
            elif tok_source == PretSourceSE.HUGGINGFACE:
                tok_dirs = f'{tok_type}/{self._config.tokenizer.name}'
            self._tokenized_path = f'{self._save_path}/{tok_dirs}'

    # --- Preprocess ---

    def preprocess(self, tokenizer):
        """
        Preprocess Dataset
        """
        self._tokenizer = tokenizer
        self._set_paths()
        if self._config.ds.preproc_rules and not self._hf_datasets:
            self._subset()
            self._tokenize()
            self._split_by_tokens_len()
            self._split_train_val_test()
            self._bundle_hf_dataset_dict()

        self._print_details()

    def _subset(self):
        """
        Subset Dataset and Use Only X% Of It
        """
        subset_rules = getattr(self._config.ds.preproc_rules, 'subset', None)
        if subset_rules and subset_rules.run and 0 < subset_rules.use < 1:
            print('Subset Dataset...')

            def subset(set):
                subset = None
                if set:
                    samples = int(len(set) * subset_rules.use)
                    subset = set.shuffle(seed=self.seed).select(range(samples))
                    print(f'Use {samples} samples from {len(set)} ({subset_rules.use * 100}%)')
                return subset

            self.train = subset(self.train)
            self.test = subset(self.test)
            self.validation = subset(self.validation)

            if subset_rules.save_as:
                self.save(subset_rules.save_as, f'{DsStateSE.SUBSET}/{subset_rules.use}')

    def _split_train_val_test(self):
        """
        Split Dataset Into Train, Test, and Validations Sets
        """

        def split(source_split_name: DsSplitSE, target_split_name: DsSplitSE, target_split_size, shuffle):
            if source_split_name == target_split_name:
                return

            print(f' Separate {target_split_name} set from {source_split_name} set...')
            source_split = self._get_split(source_split_name)

            splits = source_split.train_test_split(test_size=target_split_size, shuffle=shuffle, seed=self.seed)

            self._set_split(source_split_name, splits[DsSplitSE.TRAIN])
            self._set_split(target_split_name, splits[DsSplitSE.TEST])

        split_conf = getattr(self._config.ds.preproc_rules, 'split', None)
        source_split_name = getattr(split_conf, 'source', DsSplitSE.TRAIN)
        if split_conf and split_conf.run and (split_conf.test or split_conf.validation):
            print('Split Dataset Into Sets...')
            if split_conf.test:
                split_conf.validation = split_conf.validation / (1 - split_conf.test)
                split(source_split_name, DsSplitSE.TEST, split_conf.test, split_conf.shuffle)
            if split_conf.validation:
                print(' Separate Validation Set...')
                split(source_split_name, DsSplitSE.VALIDATION, split_conf.validation, split_conf.shuffle)

            # Test Function
            # train = 100
            # test = train * split_conf.test
            # eval = (train - test) * split_conf.eval
            # train -= test + eval
            # print(test, eval, train)

            # exit()

            def check():
                """
                Check Split Sets VS Configuration
                """
                ds_conf = self._config.ds
                split_conf = ds_conf.preproc_rules.split
                must_exist = SimpleNamespace()
                must_exist.train = True  # if ds_conf.comes_with_splits.train else False
                must_exist.test = True if ds_conf.comes_with_splits.test or (split_conf and split_conf.test) else False
                must_exist.validation = (
                    True if ds_conf.comes_with_splits.validation or (split_conf and split_conf.validation) else False
                )
                exists = SimpleNamespace()
                exists.train = True if self.train else False
                exists.test = True if self.test else False
                exists.validation = True if self.validation else False
                if must_exist != exists:
                    print('Exists: ', exists)
                    print('Must exist: ', must_exist)
                    raise Exception('DS.check_split - existing datasets mismatch with configuration.')

            check()

            if split_conf.save_as:
                self.save(split_conf.save_as, DsStateSE.SPLITS)

    def _split_by_tokens_len(self):
        """
        Split Dataset By Tokens Length
        """
        split_by_tokens_len = getattr(self._config.ds.preproc_rules, 'split_by_tokens_len', None)
        if split_by_tokens_len and split_by_tokens_len.run:
            print('Split Dataset By Tokens Length...')
            length_k = self.keys.length
            split_treshold = split_by_tokens_len.treshold

            def split(dataset):
                print(f'Filter {DsStateSE.SHORT} samples...')
                short_samples_ds = dataset.filter(lambda sample: sample[length_k] <= split_treshold)
                print(f'Filter {DsStateSE.LONG} samples...')
                long_samples_ds = dataset.filter(lambda sample: sample[length_k] > split_treshold)
                del dataset
                gc.collect()
                return short_samples_ds, long_samples_ds

            if self.train:
                self.train_short, self.train_long = split(self.train)
            if self.test:
                self.test_short, self.test_long = split(self.test)
            if self.validation:
                self.validation_short, self.validation_long = split(self.validation)
            if split_by_tokens_len.save_as:
                self.save(
                    split_by_tokens_len.save_as,
                    DsStateSE.SHORT,
                    self.train_short,
                    self.test_short if self.test else None,
                    self.validation_short if self.validation else None,
                )
                self.save(
                    split_by_tokens_len.save_as,
                    DsStateSE.LONG,
                    self.train_long,
                    self.test_long if self.test else None,
                    self.validation_long if self.validation else None,
                )

    def _tokenize(self):
        """
        Preprocess Dataset for LM Training or Finetuning
        """
        tokenize = getattr(self._config.ds.preproc_rules, 'tokenize', None)
        if tokenize and tokenize.run:

            def tokenized(dataset):
                return all(key in dataset.features for key in [self.keys.input_ids, self.keys.attention_mask])

            if self.train and not tokenized(self.train):
                self.train = self._run_tokenization_rules(DsSplitSE.TRAIN, self.train)
            if self.test and not tokenized(self.test):
                self.test = self._run_tokenization_rules(DsSplitSE.TEST, self.test)
            if self.validation and not tokenized(self.validation):
                self.validation = self._run_tokenization_rules(DsSplitSE.VALIDATION, self.validation)

            if hasattr(self, 'train_folds'):
                for idx, fold in enumerate(self.train_folds):
                    if not tokenized(fold):
                        self.train_folds[idx] = self._run_tokenization_rules(DsSplitSE.TRAIN, fold)
                for idx, fold in enumerate(self.validation_folds):
                    if not tokenized(fold):
                        self.validation_folds[idx] = self._run_tokenization_rules(DsSplitSE.VALIDATION, fold)
                if hasattr(self, 'test_folds'):
                    for idx, fold in enumerate(self.test_folds):
                        if not tokenized(fold):
                            self.test_folds[idx] = self._run_tokenization_rules(DsSplitSE.TEST, fold)

            if tokenize.save_as:
                tokenizer_name = self._tokenizer.name_or_path.replace('/', '|')
                self.save(tokenize.save_as, f'{DsStateSE.TOKENIZED}|{tokenizer_name}')

            def prin(dataset, num=10):
                """
                Print Preprocessed Dataset
                """
                print('Printing Preprocessed Dataset...')
                text_key = self.inputs_key
                for i in range(0, len(dataset[text_key])):
                    text = dataset[text_key][i]
                    mask = dataset[self.keys.attention_mask][i]
                    ids = dataset[self.keys.input_ids][i]
                    print(len(text), text)
                    print(len(mask), mask)
                    print(len(ids), ids)
                    if self.labels_key:
                        labels = dataset[self.labels_key][i]
                        print(len(labels), labels)
                    print('---')
                    if i == num:
                        break

            # prin()

    def _run_tokenization_rules(self, split_name, dataset):
        """
        Run Pre Tokenization, Tokenization, and Post Tokenization Rules
        """
        print(f'\nRun Tokenization Rules for {split_name} set...')

        preproc_rules = self._config.ds.preproc_rules
        tokenize = preproc_rules.tokenize
        pre_tok_rules = tokenize.pre_rules
        tok_rules = tokenize.rules
        post_tok_rules = tokenize.post_rules
        task_category = self._config.task.category
        target_tok_rules = utils.copy_simple_nsp(tok_rules)
        if hasattr(tokenize, 'target_rules'):
            utils.update_simple_nsp(target_tok_rules, tokenize.target_rules)
        else:
            target_tok_rules = utils.copy_simple_nsp(tok_rules)

        # #
        # PRE TOKENIZE STAGE ======================================================================
        #
        #
        # Set tokenizer.return_length flag according to other rules
        split_by_tokens_len = getattr(preproc_rules, 'split_by_tokens_len', None)
        if (split_by_tokens_len and split_by_tokens_len.run) or post_tok_rules.sort_by_len:
            tok_rules.return_length = True
        #
        # Split text into Sentences
        if pre_tok_rules.split_sentences:
            print('Split Texts Into Sentences...')
            dataset = self._split_samples_into_sents(dataset)
        #
        # Format input and labels as text-to-text and text-generation model needs
        if task_category in [TaskCatSE.TEXT_TO_TEXT, TaskCatSE.TEXT_GENERATION]:
            pre_tok_rules.label_name_to_id = False
            tok_rules.is_split_into_words = False
            post_tok_rules.concat_samples = False

            use_verbalizer = hasattr(self._config.ds.label, 'verbalizer') and getattr(
                pre_tok_rules, 'label_name_to_verbalizer', False
            )
            if use_verbalizer:
                verbalizer = dict(self._config.ds.label.verbalizer)
                dataset = dataset.map(
                    lambda batch: {self.labels_key: [verbalizer[label] for label in batch[self.labels_key]]},
                    batched=True,
                )

            if pre_tok_rules.text_to_text_rules:
                dataset = self._format_into_text_to_text(dataset)

            if pre_tok_rules.keep_target_texts_for_save:
                if self.keys.targets in dataset.column_names:
                    dataset = dataset.rename_column(self.keys.targets, f'old_{self.keys.targets}')
                dataset = dataset.add_column(self.keys.targets, dataset[self.labels_key])

        #
        # Convert label names into indexes
        if pre_tok_rules.label_name_to_id:
            dataset = self._label_names_to_ids(dataset)

        # #
        # TOKENIZE STAGE ==========================================================================
        #
        #
        def _tokenize_batch(samples_batch):
            """
            Tokenize Batch
            """
            texts = samples_batch[self.inputs_key]
            #
            # If text in binary cast it to utf-8 string
            texts = utils.to_utf8_if_binary(texts)
            #
            # Tokenize text pairs too for text-pair classification tasks
            texts_pair = None
            if task_category == TaskCatSE.TEXT_PAIR_CLASSIFICATION:
                # If text pair in binary cast it to utf-8 string
                input_key_2 = self._config.ds.input.key_2
                texts_pair = utils.to_utf8_if_binary(samples_batch[input_key_2])
            #
            # Tokenize labels too for text-to-text or generation tasks
            labels = None
            if task_category in [TaskCatSE.TEXT_TO_TEXT, TaskCatSE.TEXT_GENERATION]:
                # If label in binary cast it to utf-8 string
                labels = utils.to_utf8_if_binary(samples_batch[self.labels_key])
            #
            # Append EOS Tokens in the end of each TEXT in case of "append eos token" or "concatenate samples" rule
            # (If EOS Tokens are not already appended while splitting texts into sentences)
            if (pre_tok_rules.append_eos_token or post_tok_rules.concat_samples) and not pre_tok_rules.split_sentences:
                if texts_pair:
                    texts_pair = self._append_eos_token(texts_pair)
                else:
                    texts = self._append_eos_token(texts)
                if labels:
                    labels = self._append_eos_token(labels)
            #
            # Add text pair to tokenizer if it exists
            if texts_pair:
                tok_rules.text_pair = texts_pair
            #
            # Decoder-Only Text-Generation: concatenate input and labels
            if task_category == TaskCatSE.TEXT_GENERATION:
                texts = [t + l for t, l in zip(texts, labels, strict=True)]
            #
            # Tokenize
            encoded_batch = self._tokenizer(texts, **dict(tok_rules))
            #
            # Tokenize Targets
            if labels:
                # Encoder-Decoder Text-to-Text
                if task_category == TaskCatSE.TEXT_TO_TEXT:
                    encoded_labels = self._tokenizer(text_target=labels, **vars(target_tok_rules))
                    encoded_batch[self.keys.labels] = encoded_labels[self.keys.input_ids]
                #
                # Decoder-Only Text-Generation: mask the prompts part in labels
                elif task_category == TaskCatSE.TEXT_GENERATION:
                    # Tokenize labels separately to get their token lengths
                    labels_encoded = self._tokenizer(labels, add_special_tokens=False)
                    labels_lengths = [len(x) for x in labels_encoded[self.keys.input_ids]]
                    # Clone input_ids and mask everything except the label tokens at the end
                    masked_labels = []
                    for input_ids, label_len in zip(encoded_batch[self.keys.input_ids], labels_lengths, strict=True):
                        input_ids = list(input_ids)  # ensure list
                        masked = [-100] * (len(input_ids) - label_len) + input_ids[-label_len:]
                        masked_labels.append(masked)
                    encoded_batch[self.keys.labels] = masked_labels

                    for i, (ids, lbl) in enumerate(
                        zip(encoded_batch[self.keys.input_ids], encoded_batch[self.keys.labels], strict=True)
                    ):
                        assert len(ids) == len(lbl), f'Sample {i}: len(input_ids)={len(ids)}, len(labels)={len(lbl)}'
            #
            # End Truncation with EOS
            if tok_rules.truncation and post_tok_rules.end_truncation_with_eos:
                encoded_batch = self._end_truncation_with_eos(
                    encoded_batch, tok_rules.max_length, target_tok_rules.max_length
                )

            #
            # Compare with T5 lib tokenized inputs and labels
            def compare_with_t5_preproc(encoded_batch, samples_batch):
                def report_mismatch(seq_a, seq_b, name):
                    if len(seq_a) != len(seq_b):
                        return
                    for j, (a, b) in enumerate(zip(seq_a, seq_b, strict=True)):
                        if a != b:
                            print(f'\n❌ Mismatch found in sample {i}')
                            print(f'Tokens A: {self._tokenizer.convert_ids_to_tokens(seq_a[j : j + 5])}')
                            print(f'Tokens B: {self._tokenizer.convert_ids_to_tokens(seq_b[j : j + 5])}')
                            return

                for i in range(len(encoded_batch[self.keys.input_ids])):
                    input_ids_a = list(encoded_batch[self.keys.input_ids][i])
                    input_ids_b = list(samples_batch['old_inputs'][i])
                    labels_a = list(encoded_batch[self.keys.labels][i])
                    labels_b = list(samples_batch['old_targets'][i])
                    # print(labels_a)
                    # print(labels_b)
                    # exit()

                    if input_ids_a != input_ids_b or labels_a != labels_b:
                        if input_ids_a != input_ids_b:
                            report_mismatch(input_ids_a, input_ids_b, self.keys.input_ids)
                        if labels_a != labels_b:
                            report_mismatch(labels_a, labels_b, self.keys.labels)

            # compare_with_t5_preproc(encoded_batch, samples_batch)
            #
            # Expand each word Label to according Tokens in case of Token Classification
            if tok_rules.is_split_into_words:
                encoded_batch = self._expand_labels_to_tokens(samples_batch, encoded_batch)
            #
            return encoded_batch

        #
        # Tokenize and add EOS Tokens in case of concatenating samples
        encoded = dataset.map(lambda batch: _tokenize_batch(batch), batched=True)
        #
        # Delete dataset and collect garbage to free memory
        del dataset
        gc.collect()

        # #
        # POST TOKENIZE STAGE =====================================================================
        #
        #
        # Concatenate samples
        if post_tok_rules.concat_samples:
            print('Concatenat Samples...')
            encoded = self._concatenate_samples(encoded)
        #
        # Sort dataset by samples tokens length
        if post_tok_rules.sort_by_len:
            print('Sort Dataset By Tokens Length...')
            encoded = encoded.sort(self.keys.length)

        #
        # print(encoded)
        return encoded

    def _append_eos_token(self, texts):
        """
        Add the EOS token in the end of each text
        """
        eos_token = self._tokenizer.eos_token or self._tokenizer.sep_token
        if isinstance(texts, str):
            return texts + eos_token

        return [text + eos_token for text in texts]

    def _end_truncation_with_eos(self, encoded, max_length, targets_max_length):
        """
        Replace the last token with EOS in sequences truncated to max_length.
        Applies to input_ids and, if text-to-text, also to labels.
        """

        def replace_last_token_if_truncated(batch_key, max_length):
            for _i, seq in enumerate(encoded[batch_key]):
                if len(seq) == max_length:
                    seq[-1] = self._tokenizer.eos_token_id
                    # print(f"🔁 Replaced {batch_key}[{i}][-1] with EOS", flush=True)

        replace_last_token_if_truncated(self.keys.input_ids, max_length)

        if self._config.task.category in [TaskCatSE.TEXT_TO_TEXT, TaskCatSE.TEXT_GENERATION]:
            replace_last_token_if_truncated(self.keys.labels, targets_max_length)

        return encoded

    def _split_samples_into_sents(self, dataset):
        """
        Tokenize Texts As Sentences
        """
        input_k = self.inputs_key
        texts = dataset[input_k]
        tokenize = self._config.ds.preproc_rules.tokenize
        sentence_tokenizer = tokenize.pre_rules.split_sentences
        concat_samples = tokenize.post_rules.concat_samples
        sents_dataset = {input_k: []}
        # Tokenize Texts As Sentences
        for text in tqdm(texts, total=len(texts)):
            sentences = tok_module.tokenize_sentences(text, sentence_tokenizer)
            # print(text)
            # print(sentences)

            # If you are going to concatenate samples
            # Add EOS Tokens only at the end of the last sentence
            if concat_samples and len(sentences):
                sentences[-1] = self._append_eos_token(sentences[-1])
            sents_dataset[input_k].extend(sentences)
        return Dataset.from_dict(sents_dataset)

    def _expand_labels_to_tokens(self, samples_batch, encoded):
        """
        Expand each word Label to according Tokens in case of Token Classification
        """
        # print('Expand each word Label to according Tokens in case of Token Classification')
        expanded_labels = []
        for batch_index in range(0, len(encoded[self.keys.input_ids])):
            token_labels = []
            try:  # To which word belongss each token
                word_ids = encoded.word_ids(batch_index=batch_index)
            except (ValueError, AttributeError):
                word_ids = self._custom_word_ids(samples_batch, encoded, batch_index)
            word_labels = samples_batch[self.labels_key][batch_index]
            for wid in word_ids:
                label = self._config.ds.label.pad_id if wid is None else word_labels[wid]
                token_labels.append(label)
            expanded_labels.append(token_labels)
        encoded[self.keys.labels] = expanded_labels
        # p = 0
        # print(encoded.word_ids(batch_index=p))
        # print(encoded[self.keys.input_ids][p])
        # print(encoded[self.keys.attention_mask][p])
        # print(expanded_labels[0])
        # exit()
        return encoded

    def _custom_word_ids(self, samples_batch, encoded, batch_index):
        word_start = 1 if self._config.ds.preproc_rules.tokenize.rules.add_special_tokens else 0
        word_ids = [None] * len(encoded[self.keys.input_ids][batch_index])
        custom_input_ids = []
        words = samples_batch[self.inputs_key][batch_index]
        for i, word in enumerate(words):
            word_input_ids = self._tokenizer.encode(word, add_special_tokens=False)
            custom_input_ids.extend(word_input_ids)
            for j in range(word_start, word_start + len(word_input_ids)):
                word_ids[j] = i
            word_start += len(word_input_ids)
        input_ids = encoded[self.keys.input_ids][batch_index]
        if not utils.is_sublist(custom_input_ids, input_ids):
            print('input_ids:', input_ids)
            print('custom_input_ids:', custom_input_ids)
            raise Exception('In custom word ids function Input IDs does not match with custom input IDs!')
        return word_ids

    def _concatenate_samples(self, dataset):
        """
        Concatenate samples to
        Maximize the use of the allowed input sequence length
        """
        tokenize = self._config.ds.preproc_rules.tokenize
        rules = tokenize.rules
        concat_rules = tokenize.post_rules.concat_samples
        min_len_multi = concat_rules.min_len_multiplier
        min_len = concat_rules.min_len

        input_k = self.inputs_key
        length_k = self.keys.length
        input_ids_k = self.keys.input_ids
        attention_mask_k = self.keys.attention_mask
        special_tokens_mask_k = self.keys.special_tokens_mask
        conc_dataset = {
            # input_k: [],
            input_ids_k: [],
            attention_mask_k: [],
        }
        if rules.return_length:
            conc_dataset[length_k] = []
        if rules.return_special_tokens_mask:
            conc_dataset[special_tokens_mask_k] = []

        # Adjust a sample's length to a multiple of a minimum length
        def get_conc_min_len(sample):
            sample_len = len(sample[input_ids_k])
            calc_len = min_len_multi * (sample_len - (sample_len % min_len))
            return max(min_len, calc_len)

        def conc_sample_padd(sample):
            padd_len = rules.max_length - len(sample[input_ids_k])
            sample[attention_mask_k] += padd_len * [0]
            sample[input_ids_k] += padd_len * [self._tokenizer.pad_token_id]
            return sample

        def conc_dataset_add(sample):
            eos_id = self._tokenizer.eos_token_id
            if sample[input_ids_k][-2] == sample[input_ids_k][-1] == eos_id:
                for k in conc_dataset.keys():
                    if k == input_k:
                        pass
                    elif k == length_k:
                        sample[k] -= 1
                    elif k == special_tokens_mask_k:
                        sample[k] = [*sample[k][:-2], eos_id]
                    else:
                        sample[k] = sample[k][:-1]
            if concat_rules.padd:
                sample = conc_sample_padd(sample)
            for k in conc_dataset.keys():
                conc_dataset[k].append(sample[k])

        conc_sample = dataset[0]
        conc_min_len = get_conc_min_len(conc_sample)
        for curr_sample in tqdm(dataset, total=len(dataset)):
            if conc_sample == curr_sample:
                continue
            conc_len = len(conc_sample[input_ids_k])
            possible_len = len(curr_sample[input_ids_k]) + conc_len
            # print(f'{conc_len} < {conc_min_len} and {possible_len} < {rules.max_length}')
            # if conc_len < min_len or (conc_len < conc_min_len and possible_len < rules.max_length):
            if conc_len < conc_min_len and possible_len < rules.max_length:
                for k in conc_sample.keys():
                    if k == input_k:
                        conc_sample[k] += ' ' + curr_sample[k]
                    else:
                        if not rules.add_special_tokens:
                            conc_sample[k] += curr_sample[k]
                        else:
                            if k == length_k:
                                conc_sample[k] += curr_sample[k] - 2
                            else:
                                conc_sample[k] = conc_sample[k][:-1] + curr_sample[k][1:]
            else:
                conc_dataset_add(conc_sample)
                conc_sample = curr_sample
                conc_min_len = get_conc_min_len(conc_sample)
        conc_dataset_add(conc_sample)
        return Dataset.from_dict(conc_dataset)

    def _label_names_to_ids(self, dataset):
        """
        Convert Label Names Into Indexes
        """
        label_names = self._config.ds.label.names

        def _tokenize_batch(samples_batch):

            all_labels = samples_batch[self.labels_key]
            for i, word_labels in enumerate(all_labels):
                if self._config.task.category in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION]:
                    sentence_label_name = word_labels
                    all_labels[i] = label_names.index(sentence_label_name)

                elif self._config.task.category == TaskCatSE.TOKEN_CLASSIFICATION:
                    for j, token_label_name in enumerate(word_labels):
                        all_labels[i][j] = label_names.index(token_label_name)

            return samples_batch

        return dataset.map(lambda samples_batch: _tokenize_batch(samples_batch), batched=True)

    def _format_into_text_to_text(self, dataset):
        """
        Format a dataset into a text-to-text structure for T5-style tasks.

        Args:
            dataset: A dataset (e.g., from Hugging Face datasets library) with input and label keys.

        Returns:
            A dataset formatted for text-to-text tasks.
        """
        ds = self._config.ds
        labels_key = self.labels_key
        labels_sub_key = getattr(self._config.ds.label, 'sub_key', None)
        inputs_key_1 = self.inputs_key
        inputs_key_2 = getattr(self._config.ds.input, 'key_2', None)
        inputs_key_3 = getattr(self._config.ds.input, 'key_3', None)

        t2t_rules = ds.preproc_rules.tokenize.pre_rules.text_to_text_rules
        task_prefix = t2t_rules.task_prefix
        include_keys = t2t_rules.include_keys

        def _format(samples_batch):

            # Build the input text
            input_texts = []
            for i in range(len(samples_batch[inputs_key_1])):
                text = f'{self.name}'
                text += f' - {task_prefix}' if task_prefix else ''
                text += f' - {inputs_key_1}:' if include_keys else ':'
                text += f' {samples_batch[inputs_key_1][i]}'
                if inputs_key_2:
                    text += f' | {inputs_key_2}:' if include_keys else ' |'
                    text += f' {samples_batch[inputs_key_2][i]}'
                if inputs_key_3:
                    text += f' | {inputs_key_3}:' if include_keys else ' |'
                    text += f' {samples_batch[inputs_key_3][i]}'
                input_texts.append(text)

            # Extract and format the labels
            labels = samples_batch[labels_key]
            if self._config.task.name == 'qa' and (
                self._config.ds.name == 'record' or self._config.ds.dirs.startswith('squad')
            ):
                # Handle QA tasks
                unanswerable_label = 'unanswerable'
                labels = [
                    (sample[labels_sub_key][0] if sample[labels_sub_key] else unanswerable_label)
                    if labels_sub_key
                    else (sample[0] if sample and sample[0] else unanswerable_label)
                    for sample in samples_batch[labels_key]
                ]

            if isinstance(labels[0], (bool, int)):  # Handle both boolean and integer labels
                labels = [ds.label.names[int(label)] for label in labels]  # Convert bool to int and map to strings
            elif isinstance(labels[0], float):
                pass  # float32
            elif not isinstance(labels[0], str):  # Ensure labels are strings
                raise ValueError(f'Labels must be integers, strings, or booleans, but got {type(labels[0])}')

            return {
                inputs_key_1: input_texts,
                f'temp_{labels_key}': labels,
            }

        # Remove all original columns except first input and task_ids (if exists) keys

        remove_columns = dataset.column_names
        remove_columns.remove(inputs_key_1)
        if self.task_ids_key in dataset.column_names:
            remove_columns.remove(self.task_ids_key)

        # Apply formatting / set label columns key and cast it to string
        formatted_dataset = dataset.map(
            _format,
            batched=True,
            remove_columns=remove_columns,
        )
        formatted_dataset = formatted_dataset.rename_column(f'temp_{labels_key}', labels_key)

        # Define base feature schema
        formatted_features = {
            inputs_key_1: Value('string'),
            labels_key: Value('string'),
        }

        # Add 'task_ids' column if it exists
        if self.task_ids_key in dataset.column_names:
            formatted_features[self.task_ids_key] = Value('int64')

        # Apply the schema
        formatted_dataset = formatted_dataset.cast(Features(formatted_features))

        # print(formatted_dataset)
        return formatted_dataset

    def get_concatenated_dataset(self, datasets):

        train_split = self.split_map[DsSplitSE.TRAIN]
        test_split = self.split_map[DsSplitSE.TEST]
        valid_split = self.split_map[DsSplitSE.VALIDATION]

        train_datasets = [dataset[train_split] for dataset in datasets if train_split in dataset]
        test_datasets = [dataset[test_split] for dataset in datasets if test_split in dataset]
        valid_datasets = [dataset[valid_split] for dataset in datasets if valid_split in dataset]

        conc_datasets = {
            split: concatenate_datasets(datasets)
            for split, datasets in [
                (train_split, train_datasets),
                (test_split, test_datasets),
                (valid_split, valid_datasets),
            ]
            if datasets  # Only add non-empty datasets
        }

        dataset = DatasetDict(conc_datasets) if conc_datasets else None

        return dataset

    def _get_lengths(self, key):
        """
        Returns a list of lengths across train/test/validation for the given key.
        If use_length_feature=True and a record has 'length', that is used instead of len(e[key]).
        Results are cached in self.lengths.
        """
        self.lengths = getattr(self, 'lengths', {})
        if key not in self.lengths:
            lengths = []
            for dataset in [self.train, self.test, self.validation]:
                if not dataset:
                    continue
                if key not in dataset[0]:
                    continue
                for e in dataset:
                    if key == self.inputs_key and self.keys.length in e:
                        lengths.append(int(e[self.keys.length]))
                    else:
                        lengths.append(len(e[key]))
            self.lengths[key] = sorted(lengths)
        return self.lengths[key]

    def _percentile_length(self, key, perc=0.999):
        """
        Calculates the length at the given percentile from the cached lengths.
        Defaults to 90th percentile.
        """
        lengths = self._get_lengths(key)
        if not lengths:
            return 0
        idx = int(len(lengths) * perc)
        return lengths[idx] if idx < len(lengths) else lengths[-1]

    def longest_label_len(self):
        """
        Returns the max length for labels across all splits.
        """
        lengths = self._get_lengths(self.labels_key)
        return max(lengths) if lengths else 0

    def longest_input_len(self):
        """
        Returns the max length for inputs across all splits.
        """
        lengths = self._get_lengths(self.inputs_key)
        return max(lengths) if lengths else 0

    def label_len_longer_than(self, perc=0.999):
        """
        Returns the length at the given percentile among all labels (default 90%).
        """
        return self._percentile_length(self.labels_key, perc=perc)

    def input_len_longer_than(self, perc=0.999):
        """
        Returns the length at the given percentile among all inputs (default 90%).
        """
        return self._percentile_length(self.inputs_key, perc=perc)

    def _print_details(self):
        """
        Print Dataset Counts
        """
        print('\nDataset Details >')
        # print(f'Config: {utils.json_dumps_simple_nsp(self._config.ds)}')

        print(f'Save Path: {self._save_path}') if hasattr(self, '_save_path') else None

        if self._config.ds.label and getattr(self._config.ds.label, 'names', None):
            print(f'Labels: {self._config.ds.label.names}')

        print(self.hf)

        def print_counts(dataset, set_name):
            samples_len = len(dataset)
            if self.keys.length in dataset.features:
                total_tokens_len = sum(dataset[self.keys.length])
            elif self.keys.attention_mask in dataset.features:
                total_tokens_len = sum(mask.count(1) for mask in dataset[self.keys.attention_mask])
            else:
                total_tokens_len = 0
            print(f'{set_name} - samples: {samples_len} | tokens: {total_tokens_len}')

        if self.train:
            print_counts(self.train, DsSplitSE.TRAIN)
            train_short_split = f'{DsSplitSE.TRAIN}_{DsStateSE.SHORT}'
            train_long_split = f'{DsSplitSE.TRAIN}_{DsStateSE.LONG}'
            if getattr(self, train_short_split, None):
                print_counts(self.train_short, train_short_split)
            if getattr(self, train_long_split, None):
                print_counts(self.train_long, train_long_split)
        if self.test:
            print_counts(self.test, DsSplitSE.TEST)
            test_short_split = f'{DsSplitSE.TEST}_{DsStateSE.SHORT}'
            test_long_split = f'{DsSplitSE.TEST}_{DsStateSE.LONG}'
            if getattr(self, test_short_split, None):
                print_counts(self.test_short, test_short_split)
            if getattr(self, test_long_split, None):
                print_counts(self.test_long, test_long_split)
        if self.validation:
            print_counts(self.validation, DsSplitSE.VALIDATION)
            valid_short_split = f'{DsSplitSE.VALIDATION}_{DsStateSE.SHORT}'
            valid_long_split = f'{DsSplitSE.VALIDATION}_{DsStateSE.LONG}'
            if getattr(self, valid_short_split, None):
                print_counts(self.validation_short, valid_short_split)
            if getattr(self, valid_long_split, None):
                print_counts(self.validation_long, valid_long_split)
        print()

        self.print_examples_from_splits()
        # exit()

    def print_examples_from_splits(self, num_examples=1):
        """
        Print a few examples from each dataset split.
        """

        def print_examples(dataset, split_name):
            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break
                print(f'{split_name} example {i + 1}:')
                for key, value in example.items():
                    utils.p(f'     {key}: {value}')
                print()

        if self.train:
            print_examples(self.train, 'Train')
        if self.test:
            print_examples(self.test, 'Test')
        if self.validation:
            print_examples(self.validation, 'Validation')

    def get_first_available_split(self):
        for split in [DsSplitSE.TEST, DsSplitSE.VALIDATION, DsSplitSE.TRAIN]:
            try:
                dataset = getattr(self, split)
                if dataset is not None:
                    return dataset, split
            except AttributeError:
                continue
        raise ValueError('No valid splits available in dataset')

    def analyze_lengths(self):
        """
        Computes and returns a dictionary of input/label length stats for easy CSV logging or pretty printing.
        """

        dataset, used_split = self.get_first_available_split()
        print(f'Using {used_split} split')

        calc_label_lens = self.labels_key and isinstance(dataset[self.labels_key][0], str)
        label_lengths = (
            {
                'label_max': self.longest_label_len(),
            }
            if calc_label_lens
            else {}
        )

        input_lengths = {
            'input_max': self.longest_input_len(),
        }

        percentiles = [0.9999, 0.999, 0.99, 0.9, 0.8, 0.65, 0.5]
        for perc in percentiles:
            perc_key = f'{int(perc * 10000):04d}'  # e.g. 9000, 9900
            input_lengths[f'input_{perc_key}'] = self.input_len_longer_than(perc)
            if calc_label_lens:
                label_lengths[f'label_{perc_key}'] = self.label_len_longer_than(perc)

        lengths = input_lengths

        if calc_label_lens:
            lengths.update(label_lengths)

        return lengths

    def _save_as_csv(self, dirs):
        save_path = f'{self._save_path}/{dirs}' if dirs else self._save_path
        print('Save Dataset as CSV: ', save_path)
        Path(save_path).mkdir(parents=True, exist_ok=True)

        def save_split(split, set_name):
            save_file_path = f'{save_path}/{self.name}_{set_name}.csv'
            with open(save_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                # 1) Build 'fieldnames' to include both inputs and labels if present
                fieldnames = [self.inputs_key]
                if self.labels_key and self.labels_key in split[0]:
                    fieldnames.append(self.labels_key)

                # 2) Create a DictWriter with those fieldnames
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writeheader()

                # 3) Write out each sample
                for sample in tqdm(split, desc=f'Writing {set_name} data'):
                    row = {self.inputs_key: utils.to_utf8_if_binary(sample.get(self.inputs_key, ''))}

                    # If the labels key exists in this sample, add it to the row
                    if self.labels_key and self.keys.targets in sample:
                        row[self.labels_key] = utils.to_utf8_if_binary(sample.get(self.keys.targets, ''))
                    elif self.labels_key and self.labels_key in sample:
                        row[self.labels_key] = utils.to_utf8_if_binary(sample.get(self.labels_key, ''))

                    # 4) Write the row to CSV
                    writer.writerow(row)

        if self.train:
            save_split(self.train, DsSplitSE.TRAIN)
        if self.test:
            save_split(self.test, DsSplitSE.TEST)
        if self.validation:
            save_split(self.validation, DsSplitSE.VALIDATION)

    def _save_as_huggingface(self, dirs, train_split, test_split=None, validation_split=None):
        """
        Save as Huggingface Dataset
        """
        save_path = f'{self._tokenized_path}/{dirs}' if dirs else self._tokenized_path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        print('\nSave as Huggingface Dataset: ', save_path)

        def save_tok_counts(set_name, dataset):
            length_k = self.keys.length
            if length_k in dataset.features:
                with open(f'{save_path}/{length_k}s_{set_name}.txt', 'w') as f:
                    for _, l in enumerate(dataset[length_k]):
                        f.write(f'{l}\n')

        dataset = {}
        if train_split:
            train_split = train_split
            dataset[DsSplitSE.TRAIN] = train_split
            save_tok_counts(DsSplitSE.TRAIN, train_split)
        if test_split:
            if self.keys.length in test_split.features:
                test_split = test_split.sort(self.keys.length)
            dataset[DsSplitSE.TEST] = test_split
            save_tok_counts(DsSplitSE.TEST, test_split)
        if validation_split:
            if self.keys.length in validation_split.features:
                validation_split = validation_split.sort(self.keys.length)
            dataset[DsSplitSE.VALIDATION] = validation_split
            save_tok_counts(DsSplitSE.VALIDATION, validation_split)
        DatasetDict(dataset).save_to_disk(save_path)

    def save(self, save_as, dirs, train_split=None, test_split=None, validation_split=None):
        if save_as == SaveDatasetAsSE.CSV:
            self._save_as_csv(dirs)
        elif save_as == SaveDatasetAsSE.HUGGINGFACE:
            train_split = train_split if train_split else self.train
            test_split = test_split if test_split else self.test
            validation_split = validation_split if validation_split else self.validation
            self._save_as_huggingface(dirs, train_split, test_split, validation_split)

    @staticmethod
    def get_name(config):
        name = None
        if hasattr(config, 'ds'):
            name = getattr(config.ds, 'descriptive_name', None) or config.ds.name
        config.descriptive_name = name
        return name
