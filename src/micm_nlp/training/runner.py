import inspect
import math
import os
import random
import shutil
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
import wandb
from transformers import (
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from wandb.sdk.wandb_settings import Settings

import micm_nlp.path as nlpka_path
import micm_nlp.utils as utils
from micm_nlp.enums import DeviceSE, DsSplitSE, ModelArchSE, ModeSE, TaskCatSE
from micm_nlp.evals.eval import get_compute_metrics
from micm_nlp.models.peft import PEFT
from micm_nlp.models.xpe import is_xpe_config
from micm_nlp.training.callbacks import (
    DownstreamFineTuningCallback,
    EmptyCudaCacheCallback,
    LossEarlyStoppingCallback,
    NormalizePromptEncoderEmbeddings,
    ParamNormLogger,
    get_preprocess_logits_for_metrics,
)
from micm_nlp.training.trainers import custom_trainer_class_factory


class TRAINER:
    # Modules searched (in order) when resolving class names from YAML.
    DATA_COLLATOR_SOURCE_MODULES: ClassVar[list[str]] = ['transformers', 'micm_nlp.training.data_collators']
    TRAINING_ARGS_SOURCE_MODULES: ClassVar[list[str]] = ['transformers']
    TRAINER_SOURCE_MODULES: ClassVar[list[str]] = ['transformers', 'micm_nlp.training.trainers']

    def __init__(self, model, dataset, tokenizer=None):
        self._model = model
        self._config = model._config
        self._tokenizer = tokenizer if tokenizer else dataset._tokenizer
        self._dataset = dataset
        self._setup_trainer()
        self.print_details()

    # -- Run loop ----------------------------------------------------------

    def run(self):
        full_shot_res = None
        zero_shot_res = None
        test_pref = 'test'
        test_z_pref = 'test_zero'
        run_test = getattr(self._config.test, 'run', True)
        zero_shot = getattr(self._config.test, 'zero_shot', False)
        zero_shot_only = getattr(self._config.test, 'zero_shot_only', False)
        run_eval_before_train = getattr(self._config.eval, 'before_training', False)
        run_eval_after_train = getattr(self._config.eval, 'after_training', False)
        run_eval_before_train_on_test = getattr(self._config.eval, 'before_training_on_test', False)
        run_eval_after_train_on_test = getattr(self._config.eval, 'after_training_on_test', False)

        # Initialize Weights and Biases
        self._model.hf.wandb_run = self._init_wandb() if not wandb.run else None
        try:
            # Zero Shot Testing
            if run_test and zero_shot:
                zero_shot_res = self._test(test_z_pref)

            if not (run_test and zero_shot and zero_shot_only):
                if self._config.mode in [ModeSE.TRAIN, ModeSE.FINETUNE]:
                    self._evaluate() if run_eval_before_train else None
                    self._evaluate(DsSplitSE.TEST, test_z_pref) if run_eval_before_train_on_test else None
                    self._train()
                    self._evaluate() if run_eval_after_train else None
                    self._evaluate(DsSplitSE.TEST, test_pref) if run_eval_after_train_on_test else None
                elif self._config.mode == ModeSE.EVALUATE:
                    self._evaluate()
                    self._evaluate(DsSplitSE.TEST, test_pref)
                if run_test or self._config.mode == ModeSE.TEST:
                    full_shot_res = self._test(test_pref)

            # Finish Weights and Biases
            if self._model.hf.wandb_run:
                self._model.hf.wandb_run.finish()

            return SimpleNamespace(full_shot=full_shot_res, zero_shot=zero_shot_res)

        finally:
            if self._model.hf.wandb_run:
                self._model.hf.wandb_run.finish()

    # -- Train / Evaluate / Test -------------------------------------------

    def _train(self):
        utils.p('\n[green]Train Model...[/green]')
        print(f'Task name:  {self._config.task.name}')
        self.trainer.compute_metrics = get_compute_metrics(
            self._config,
            self._model.label_pad_id,
            self._metric_prefix,
            self._model.eval_path,
            self._tokenizer,
            self._dataset.validation,
        )
        self.trainer.train()
        self._load_best_model()

        if getattr(self._config.custom_training_args, 'save_final_model', False):
            if getattr(self._config.custom_training_args, 'keep_only_final_model', False):
                shutil.rmtree(self._model.path)
            self.trainer.save_model(self._model.path)

    def _evaluate(self, ds_split_name=DsSplitSE.VALIDATION, metric_key_prefix='eval'):
        eval_res = None
        ds_split = self._dataset.validation if ds_split_name == DsSplitSE.VALIDATION else self._dataset.test
        if ds_split:
            utils.p(f'\n[green]Evaluate Model on [bold]{ds_split_name}[/bold] set...[/green]')
            print(f'Task name: {self._config.task.name}')

            if self._config.model.architecture == ModelArchSE.XLNET:
                import torch._dynamo

                torch._dynamo.config.suppress_errors = True
                print('Suppressing Errors for XLNET: torch._dynamo.config.suppress_errors = True')

            self.trainer.compute_metrics = get_compute_metrics(
                self._config,
                self._model.label_pad_id,
                self._metric_prefix,
                self._model.eval_path,
                self._tokenizer,
                ds_split,
            )
            eval_res = self.trainer.evaluate(ds_split, metric_key_prefix=metric_key_prefix)
            utils.p(eval_res)
        return eval_res

    def _test(self, metric_key_prefix='test'):
        test_res = None
        if self._dataset.test:
            utils.p('\n[green]Test Model...[/green]')
            print('Task name: ', self._config.task.name)

            self.trainer.compute_metrics = get_compute_metrics(
                self._config,
                self._model.label_pad_id,
                self._metric_prefix,
                self._model.eval_path,
                self._tokenizer,
                self._dataset.test,
            )

            test_res = self.trainer.predict(self._dataset.test, metric_key_prefix=metric_key_prefix)
            utils.p(test_res.metrics)

            if self._config.test.save_predictions:
                self._save_predictions(test_res)

        return test_res

    def _save_predictions(self, test_res):
        predictions_to_save = []
        labels = test_res.label_ids
        predictions = test_res.predictions

        if self._config.task.preproc_rules.filter_padded:
            masked_preds, masked_labs = [], []
            for preds, labs in zip(predictions, labels, strict=True):
                mask = labs != self._model.label_pad_id
                masked_preds.append(preds[mask])
                masked_labs.append(labs[mask])
            predictions, labels = masked_preds, masked_labs

        if self._config.task.preproc_rules.label_id_to_name:
            label_names = np.array(self._config.ds.label.names)
            named_preds, named_labs = [], []
            for preds, labs in zip(predictions, labels, strict=True):
                named_preds.append(label_names[preds])
                named_labs.append(label_names[labs])
            predictions, labels = named_preds, named_labs

        print_num = 5
        for prediction, label, sample in zip(predictions, labels, self._dataset.test, strict=True):
            input_ids = sample[self._dataset.keys.input_ids]
            readable_tokens = [self._tokenizer.decode(tok_id) for tok_id in input_ids]

            if self._config.task.category in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION]:
                readable_tokens = ' '.join(readable_tokens)
            elif self._config.task.category == TaskCatSE.TOKEN_CLASSIFICATION:
                predictions_row = prediction.tolist() if isinstance(prediction, np.ndarray) else list(prediction)
                labels_row = label.tolist() if isinstance(label, np.ndarray) else list(label)
                tokens_row = readable_tokens[1:-1]
                predictions_to_save.extend([predictions_row, labels_row, tokens_row, []])

            if print_num:
                print_num -= 1
                utils.p('Predicted: ', prediction, 'True: ', label, 'Text: ', readable_tokens)

        csv_file_path = f'{nlpka_path.evals_dir()}/predictions/{self._model.name}.csv'

        if self._config.task.category in [TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION]:
            df = pd.DataFrame(predictions_to_save, columns=['Predicted', 'True', 'Text'])
            df.to_csv(csv_file_path, index=False)
        elif self._config.task.category == TaskCatSE.TOKEN_CLASSIFICATION:
            max_len = max(len(row) for row in predictions_to_save)
            standardized_rows = [row + [''] * (max_len - len(row)) for row in predictions_to_save]
            df = pd.DataFrame(standardized_rows)
            df.to_csv(csv_file_path, index=False)
        print(f'Predictions data to {csv_file_path}')

    def _load_best_model(self):
        best_checkpoint_path = self.trainer.state.best_model_checkpoint
        if best_checkpoint_path:
            best_checkpoint = int(best_checkpoint_path.split('/')[-1].replace(self._model.checkpoint_pref, ''))
            current_checkpoint = self.trainer.state.global_step
            if best_checkpoint != current_checkpoint:
                utils.p('\n[green]Custom load of the best model...[/green]')
                print('Path: ', best_checkpoint_path)
                if PEFT.is_peft(self._model):
                    self.trainer.model = PEFT.from_pretrained(self.trainer.model.base_model, best_checkpoint_path)
                else:
                    self.trainer.model = self.trainer.model.from_pretrained(best_checkpoint_path)
        else:
            utils.p('\n[red]No best model found...[/red]')

        temp_wandb_run = self._model.hf.wandb_run
        self._model.hf = self.trainer.model
        self._model.hf.wandb_run = temp_wandb_run

    # -- Trainer setup -----------------------------------------------------

    def _setup_trainer(self):
        self._setup_wandb()
        self._setup_metrics()
        self._setup_trainer_callbacks()
        self._setup_data_collator()
        self._setup_training_args()

        print('Setup Trainer...')
        TrainerCls = utils.resolve_cls(
            getattr(self._config.trainer, 'cls', None) if self._config.trainer else None,
            self.TRAINER_SOURCE_MODULES,
            'trainer.cls',
        )
        trainer_init_args = {
            'model': self._model.hf,
            'args': self.training_args,
            'processing_class': self._tokenizer,
            'data_collator': self.data_collator,
            'train_dataset': self._dataset.train,
            'eval_dataset': self._dataset.validation,
            'callbacks': self._trainer_callbacks,
            'compute_metrics': self.compute_metrics,
        }
        if not issubclass(TrainerCls, Seq2SeqTrainer):
            trainer_init_args['preprocess_logits_for_metrics'] = self.preprocess_logits_for_metrics

        CustomTrainer = custom_trainer_class_factory(TrainerCls)
        self.trainer = CustomTrainer(
            custom_args=self._config.custom_training_args,
            **trainer_init_args,
        )

    # -- Data collator -----------------------------------------------------

    def _setup_data_collator(self):
        print('Setup data collator...')

        cfg = self._config.data_collator
        if cfg.args is None:
            from micm_nlp.config import _Flex

            cfg.args = _Flex()
        args = cfg.args

        DataCollator = utils.resolve_cls(
            getattr(cfg, 'cls', None),
            self.DATA_COLLATOR_SOURCE_MODULES,
            'data_collator.cls',
        )
        accepted = set(inspect.signature(DataCollator.__init__).parameters)

        # Runtime-derived kwargs: inject only if the chosen collator accepts them
        # and the user didn't already set them in YAML.
        if 'label_pad_token_id' in accepted and getattr(args, 'label_pad_token_id', None) is None:
            args.label_pad_token_id = self._model.label_pad_id

        if 'shift_labels_by' in accepted and getattr(args, 'shift_labels_by', None) is None:
            total_virt_tokens = PEFT.get_total_virtual_tokens(self._model)
            if total_virt_tokens:
                args.shift_labels_by = total_virt_tokens

        if 'max_length' in accepted:
            args.max_length = min(
                getattr(args, 'max_length', None) or self._model.max_length,
                self._model.max_length,
            )

        collator_kwargs = {k: v for k, v in dict(args).items() if v is not None}
        self.data_collator = DataCollator(
            **collator_kwargs,
            tokenizer=self._tokenizer,
        )

    # -- Metrics -----------------------------------------------------------

    def _setup_metrics(self):
        print('Setup evaluation functions...')
        if self._config.task.category == TaskCatSE.LANGUAGE_MODELING:
            self._metric_prefix = ''
        else:
            config_name = self._config.file_path.split('/')[-1]
            self._metric_prefix = f'{config_name}/'

        self.compute_metrics = get_compute_metrics(
            self._config,
            self._model.label_pad_id,
            self._metric_prefix,
            self._model.eval_path,
            self._tokenizer,
            self._dataset.validation,
        )
        self.preprocess_logits_for_metrics = get_preprocess_logits_for_metrics(
            self._config, num_virtual_tokens=PEFT.get_total_virtual_tokens(self._model)
        )

    # -- W&B ---------------------------------------------------------------

    def _setup_wandb(self):
        print('Setup Weights and Biases...')
        wandb_proj_name_params = [
            self._config.model.architecture,
            self._config.task.category,
            self._config.task.name,
        ]
        self.wandb_project_name = '_'.join([str(p) for p in wandb_proj_name_params if p is not None])

    def _init_wandb(self):
        nlpka_path.wandb_dir().mkdir(parents=True, exist_ok=True)
        try:
            wandb_run = wandb.init(
                name=self._model.name,
                project=self.wandb_project_name,
                config=self._config.model_dump(),
                dir=str(nlpka_path.wandb_dir()),
                settings=Settings(init_timeout=300),
            )
        except wandb.errors.CommError as e:
            print(f'[W&B WARNING] Init failed with CommError: {e}. Retrying in offline mode.')
            os.environ['WANDB_MODE'] = 'offline'
            wandb_run = wandb.init(
                name=self._model.name + '_offline',
                project=self.wandb_project_name,
                config=self._config.model_dump(),
                dir=str(nlpka_path.wandb_dir()),
                settings=Settings(init_timeout=300),
            )
        return wandb_run

    # -- Callbacks ---------------------------------------------------------

    def _setup_trainer_callbacks(self):
        print('Setup Trainer Callbacks...')
        self._trainer_callbacks = []

        self._trainer_callbacks.append(ParamNormLogger())

        if self._model.device == DeviceSE.CUDA:
            steps = self._config.cuda.empty_cache_steps
            if steps:
                self._trainer_callbacks.append(EmptyCudaCacheCallback(steps))

        if self._config.eval.downstream_tasks:
            self._trainer_callbacks.append(DownstreamFineTuningCallback(self._config, self._model.path))

        if self._config.custom_training_args.early_stopping_patience:
            self._trainer_callbacks.append(
                LossEarlyStoppingCallback(
                    early_stopping_patience=self._config.custom_training_args.early_stopping_patience,
                    early_stopping_threshold=self._config.custom_training_args.early_stopping_threshold,
                    early_stopping_after=self._config.custom_training_args.early_stopping_after,
                )
            )

        peft = getattr(self._config.task, 'peft', None)
        if peft and is_xpe_config(peft):
            self._trainer_callbacks.append(NormalizePromptEncoderEmbeddings())

        utils.p('List of Callbacks: ', self._trainer_callbacks)

    # -- Training args -----------------------------------------------------

    def _training_args_block(self):
        """Returns the mutable args sub-block, lazily creating it if absent."""
        ta = self._config.training_args
        if ta.args is None:
            from micm_nlp.config import _Flex

            ta.args = _Flex()
        return ta.args

    def _calculate_eval_steps(self):
        targs = self._training_args_block()
        eval_during_train = getattr(self._config.eval, 'during_training', None)
        if eval_during_train == 0:
            return 0
        elif eval_during_train is None:
            return getattr(targs, 'eval_steps', None)
        else:
            training_steps = (
                math.ceil(len(self._dataset.train) / self._model.effective_batch_size) * targs.num_train_epochs
            )
            base_interval = training_steps // (eval_during_train + 1)
            offset = max(1, int(training_steps * 0.01))
            return base_interval + offset

    def _setup_eval_steps(self):
        print('Setup Eval Strategy...')
        targs = self._training_args_block()
        eval_during_train = getattr(self._config.eval, 'during_training', None)
        if eval_during_train is not None:
            if eval_during_train == 0:
                targs.eval_strategy = 'no'
                targs.load_best_model_at_end = False
            elif eval_during_train > 0:
                targs.eval_strategy = 'steps'
                targs.eval_steps = self._calculate_eval_steps()

        eval_strategy = getattr(targs, 'eval_strategy', 'no')
        save_strategy = getattr(targs, 'save_strategy', 'no')
        if eval_strategy != 'no' and save_strategy != 'no' and getattr(targs, 'load_best_model_at_end', False):
            targs.save_strategy = eval_strategy
            targs.save_steps = getattr(targs, 'eval_steps', None)

        print(f' Eval Strategy: {getattr(targs, "eval_strategy", None)}')
        print(f' Eval Steps: {getattr(targs, "eval_steps", None)}')
        print(f' Save Strategy: {getattr(targs, "save_strategy", None)}')
        print(f' Save Steps: {getattr(targs, "save_steps", None)}')

    def _setup_training_args(self):
        print('Setup TrainingArguments...')
        self._setup_eval_steps()

        targs = self._training_args_block()
        targs.fp16 = getattr(targs, 'fp16', False) and self._model.device == torch.device(DeviceSE.CUDA)

        reproducibility = getattr(targs, 'full_determinism', False)
        if not reproducibility:
            targs.seed = random.randint(0, 2**32 - 1)

        if getattr(targs, 'metric_for_best_model', None) is not None:
            targs.metric_for_best_model = self._metric_prefix + targs.metric_for_best_model

        if getattr(targs, 'optim_args', None) is not None:
            targs.optim_args = ', '.join(f'{key}={val}' for key, val in dict(targs.optim_args).items())

        if getattr(targs, 'lr_scheduler_kwargs', None) is not None:
            targs.lr_scheduler_kwargs = dict(targs.lr_scheduler_kwargs)

        TArgs = utils.resolve_cls(
            getattr(self._config.training_args, 'cls', None),
            self.TRAINING_ARGS_SOURCE_MODULES,
            'training_args.cls',
        )

        if issubclass(TArgs, Seq2SeqTrainingArguments):
            if hasattr(self._config, 'generation_config'):
                max_length = getattr(self._config.generation_config, 'max_length', 0)
                max_new_tokens = getattr(self._config.generation_config, 'max_new_tokens', 0)
                if not (max_new_tokens + max_length):
                    self._config.generation_config.max_new_tokens = self._dataset.longest_label_len() + 5
                force_words_ids = getattr(self._config.generation_config, 'force_words_ids', None)
                if force_words_ids:
                    print('\nGeneration config > force_words_ids: ', force_words_ids)
                    force_words_ids = [
                        self._tokenizer(word, add_special_tokens=False).input_ids for word in force_words_ids
                    ]
                    self._config.generation_config.force_words_ids = force_words_ids

                generation_whitelist = getattr(self._config.custom_training_args, 'generation_whitelist', None)
                if generation_whitelist:
                    self._config.generation_config.early_stopping = False
                    self._config.generation_config.do_sample = False
                    self._config.generation_config.num_beams = 1

                targs.generation_config = GenerationConfig(**dict(self._config.generation_config))

        targs_kwargs = {k: v for k, v in dict(targs).items() if v is not None}
        output_dir = self._model.path if self._config.mode != ModeSE.TEST else self._model.eval_path
        self.training_args = TArgs(
            run_name=self._model.name,
            output_dir=output_dir,
            logging_dir=self._model.logs_path,
            **targs_kwargs,
        )

    # -- Print details -----------------------------------------------------

    def print_details(self):
        m = self._model
        print('\nModel Details >')
        if hasattr(m, 'pret_path'):
            print(f' - Pretrained Path: {m.pret_path}')
        print(f' - Arch: {self._config.model.architecture}')
        print(f' - Uuid4: {getattr(m, "uuid4", None)}')
        print(f' - Name: {getattr(m, "name", None)}')
        print(f' - Class: {type(m._model).__name__}')
        from micm_nlp.models.model import MODEL

        base_model = MODEL.get_base_model(m._model)
        print(f' - Base Model Class: {type(base_model).__name__}')
        print(f' - Vocabulary size: {self._tokenizer.vocab_size}')
        print(f' - Embedding dimension: {m.embedding_dim}')
        print(f' - Max sequence length: {getattr(m, "max_length", None)}')
        print(f' - DataCollator max length: {getattr(self._config.data_collator.args, "max_length", None)}')

        utils.p(' - Parameters:')
        utils.p(f' - - All: {m.param_size:,}')
        utils.p(f' - - Trainable: [red]{m.trainable_param_size:,} ({m.trainable_param_size_ratio:.4f}%)[/red]')
        utils.p('\n - Model Config: ', m._model.config)

        print('\nRun Config >')
        utils.p(self._config)

        print('\nEnvironment Details >')
        print(f' - Available Device: {m.device}')
        for var in [
            'CUDA_VISIBLE_DEVICES',
            'MASTER_PORT',
            'RANK',
            'LOCAL_RANK',
            'WORLD_SIZE',
            'LOCAL_WORLD_SIZE',
            'MASTER_ADDR',
            'NCCL_DEBUG',
            'TORCH_CPP_LOG_LEVEL',
            'TORCH_DISTRIBUTED_DEBUG',
            'CUDA_HOME',
            'NCCL_CUDA_PATH',
            'LD_LIBRARY_PATH',
        ]:
            print(f' - {var}: {os.getenv(var)}')

        if hasattr(self, 'trainer'):
            print('\nTrainer Details >')
            print(f' - Distributed State: {self.trainer.args.distributed_state}')
            print(f' - Place Model on Device: {self.trainer.place_model_on_device}')
            print(f' - Data Parallelism: {self.trainer.args.parallel_mode}')
            print(f' - ZeRO: {self.trainer.is_deepspeed_enabled}')
            print(f' - FSDP: {self.trainer.is_fsdp_enabled}')
            print(f' - FSDP_XLA: {self.trainer.is_fsdp_xla_enabled}')
            print(f' - Model Parallelism: {self.trainer.is_model_parallel}')

            print('\nBatch Details >')
            self.print_batch_examples(DsSplitSE.TRAIN)
            self.print_batch_examples(DsSplitSE.VALIDATION, 1)
            self.print_batch_examples(DsSplitSE.TEST, 1)

        utils.p(f'\n[green][bold]Model is ready to {self._config.mode}![/bold][/green]\n')

    def print_batch_examples(self, split, batches=2, samples_per_batch=2):
        dataloader = None
        if split == DsSplitSE.TRAIN and self._dataset.train:
            dataloader = self.trainer.get_train_dataloader()
        elif split == DsSplitSE.VALIDATION and self._dataset.validation:
            dataloader = self.trainer.get_eval_dataloader()
        elif split == DsSplitSE.TEST and self._dataset.test:
            dataloader = self.trainer.get_test_dataloader(self._dataset.test)
        if not dataloader:
            return

        from micm_nlp.tokenizers.decoding import decode

        for i, batch in enumerate(dataloader):
            if i == batches:
                break
            if self._dataset.keys.input_ids in batch:
                input_ids = batch[self._dataset.keys.input_ids]
                print()
                print(f'{split.upper()} Batch {i + 1} - size: {len(input_ids)} | max_len: {len(input_ids[0])}')
                print(f'{split.upper()} Batch {i + 1} - items: {len(batch)}, keys: {batch.keys()}')

                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f'    Tensor Item: {key}, Tensor Shape: {value.shape}, Device: {value.device}')

                for j, ids in enumerate(input_ids):
                    if j == samples_per_batch:
                        break
                    ids = ids.tolist()
                    all_count = len(ids)
                    pad_count = ids.count(self._tokenizer.pad_token_id)
                    real_count = all_count - pad_count
                    mask_count = ids.count(self._tokenizer.mask_token_id)

                    print(
                        f'{split.upper()} Sample {j + 1} - len: {real_count} | pads: {pad_count} | masks: {mask_count} ({(mask_count / real_count * 100):.2f}%)'
                    )
                    print(f'Inputs: {self._tokenizer.decode(ids, skip_special_tokens=False)}')

                    if self._dataset.keys.labels in batch:
                        labels = batch[self._dataset.keys.labels][j]
                        if self._config.task.category == TaskCatSE.TEXT_TO_TEXT:
                            decoded_target = decode(
                                labels, self._tokenizer, self._model.label_pad_id, skip_special_tokens=False
                            )
                            print(f'Target: {decoded_target}')
                            print(f'Labels: {labels}')
                        else:
                            print(f'Label: {labels}')
