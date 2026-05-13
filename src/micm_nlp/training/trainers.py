import random
from typing import Any

# PyTorch
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

# Transformers
from transformers import LogitsProcessorList, Seq2SeqTrainer, Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled, logging, version

import micm_nlp.utils as utils
from micm_nlp.training.batching import TokenBudgetBatchSampler, calibrate_token_budget
from micm_nlp.training.logits_processors import ConstrainedPrefixLogitsProcessor

logger = logging.get_logger(__name__)

# Optional datasets module
if is_datasets_available():
    import datasets


def _get_lengths(dataset, length_column_name: str) -> list[int]:
    """Read per-sample sequence lengths, falling back to deriving from
    ``input_ids`` when the named column has been stripped by
    ``_remove_unused_columns``.

    Mirrors the fallback in HF Trainer's ``_get_eval_sampler`` so the
    token-budget sampler/calibrator works whether or not the user has
    listed the length column in ``custom_args.usable_columns``.
    """
    if length_column_name in dataset.column_names:
        return dataset[length_column_name]
    return [len(f['input_ids']) for f in dataset]


def build_inference_dataloader_kwargs(
    *,
    dataset,
    args,                 # HF TrainingArguments-like
    data_collator,        # the collator instance, used both for collate_fn and (token-budget path) pad_to_multiple_of
    token_budget: int | None,
) -> dict:
    """Construct DataLoader kwargs for eval/test.

    When ``token_budget`` is None, returns the legacy fixed-batch kwargs
    (caller still needs to add ``sampler``). When ``token_budget`` is an
    int, returns kwargs using a ``TokenBudgetBatchSampler`` instead of
    ``batch_size`` + ``sampler`` (those keys are omitted because PyTorch
    rejects them alongside ``batch_sampler``).
    """
    base = {
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
    }

    if token_budget is None:
        # Legacy path: caller adds sampler + drop_last + prefetch_factor
        base['batch_size'] = args.eval_batch_size
        return base

    # Token-budget path: batch_sampler is XOR with batch_size/sampler.
    # pad_to_multiple_of lives on the data collator (DataCollatorForSeq2Seq),
    # not on HF TrainingArguments — read it from there with a fallback to 1.
    lengths = _get_lengths(dataset, args.length_column_name)
    pad_multiple = getattr(data_collator, 'pad_to_multiple_of', None) or 1
    base['batch_sampler'] = TokenBudgetBatchSampler(
        lengths=lengths,
        token_budget=token_budget,
        pad_multiple=pad_multiple,
    )
    return base


# Build a custom trainer class that inherits from the base trainer class and adds custom functionality
def custom_trainer_class_factory(BaseTrainer: Trainer | Seq2SeqTrainer):
    class CustomTrainer(CustomTrainerMixin, BaseTrainer):
        def __init__(self, *args, custom_args=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__init_custom_trainer__(custom_args)

    return CustomTrainer


class CustomTrainerMixin:
    """
    A mixin class that adds custom functionality to either a Trainer or Seq2SeqTrainer instance.
    I think this is copied from transformers 4.39.1
    """

    def __init_custom_trainer__(self, custom_args):
        self.custom_args = custom_args
        self._inspected_optimizer = False  # So we only log once

    def _print_param_names(self, param_set):
        param_to_name = {p: n for n, p in self.model.named_parameters()}
        for p in param_set:
            name = param_to_name.get(p, '[unknown]')
            print(f'  - {name} {tuple(p.shape)}')

    def _inspect_optimizer_weight_decay(self):
        # Create a mapping from param object to its name
        {p: n for n, p in self.model.named_parameters()}

        utils.p('\n[red]Inspecting optimizer parameter groups:[/red]')
        for i, group in enumerate(self.optimizer.param_groups):
            print(f'\nParameter group {i}:')
            for key, value in group.items():
                if key != 'params':
                    print(f'{key}: {value}')
            print('Parameters in this group:')
            self._print_param_names(group['params'])
        # exit()

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        if not self._inspected_optimizer:
            self._inspect_optimizer_weight_decay()
            self._inspected_optimizer = True

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # Custom Step 0: Get custom optimizer grouped parameters
        custom_optimizer_grouped_parameters_args = getattr(self.custom_args, 'optimizer_grouped_parameters', None)
        if not custom_optimizer_grouped_parameters_args:
            return super().create_optimizer()

        if self.optimizer is None:
            opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

            # Custom Step 1: Assign params to each custom group
            custom_optimizer_grouped_parameters = []
            for opt_param in custom_optimizer_grouped_parameters_args:
                param_name_parts = getattr(opt_param, 'param_name_parts', [])
                matched_params = [
                    param
                    for name, param in opt_model.named_parameters()
                    if any(part in name for part in param_name_parts) and param.requires_grad
                ]
                if not matched_params:
                    print(f'[WARNING] No parameters matched for {param_name_parts} (skipping)')
                    continue
                opt_param.params = matched_params
                opt_param.weight_decay = getattr(opt_param, 'weight_decay', self.args.weight_decay)
                del opt_param.param_name_parts  # clean up config dict
                custom_optimizer_grouped_parameters.append(opt_param)

            # Custom Step 2: Collect all matched param objects into one set
            custom_parameters_set = set()
            for opt_param in custom_optimizer_grouped_parameters:
                custom_parameters_set.update(opt_param.params)  # assumes opt_param is a dict
            # print('\ncustom_parameters_set:')
            # self._print_param_names(custom_parameters_set)
            # exit()

            decay_parameters = self.get_decay_parameter_names(opt_model)
            # custom_parameters_set = {param for param in custom_parameters_set if param not in decay_parameters}
            #
            optimizer_grouped_parameters = [
                # Custom Step 3: Add custom optimizer grouped parameters
                *[dict(opt_param) for opt_param in custom_optimizer_grouped_parameters],
                # Custom Step 4: Exclude custom parameters from default optimizer grouped parameters
                {
                    'params': [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad and p not in custom_parameters_set)
                    ],
                    'weight_decay': self.args.weight_decay,
                },
                {
                    'params': [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad and p not in custom_parameters_set)
                    ],
                    'weight_decay': 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if 'params' in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop('params')

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if 'model' in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop('model')

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if 'optimizer_dict' in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop('optimizer_dict')

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == 'Adam8bit':
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, torch.nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f'skipped {module}: {skipped / 2**20}M params')
                        manager.register_module_override(module, 'weight', {'optim_bits': 32})
                        logger.debug(f'bitsandbytes: will optimize {module} in fp32')
                logger.info(f'skipped: {skipped / 2**20}M params')

        # if is_sagemaker_mp_enabled():
        #     self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

        dataloader_params = {
            'batch_size': self._train_batch_size,
            'collate_fn': data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'persistent_workers': self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.custom_args.random_task_exclusion:
                batch_sampler = RandomTaskExclusionBatchSampler(
                    dataset=train_dataset, batch_size=self._train_batch_size, drop_last=self.args.dataloader_drop_last
                )
                dataloader_params['batch_sampler'] = batch_sampler
                del dataloader_params['batch_size']
            else:
                sampler = (
                    SequentialSampler(train_dataset)
                    if self.custom_args.train_force_sequential
                    else self._get_train_sampler()
                )
                dataloader_params['sampler'] = sampler

            # print(train_dataset['label'])
            # print(train_dataset['labels'])
            # print(train_dataset['input_ids'])
            # from collections import Counter
            # # Assuming your dataset has "targets" field
            # class_counts = Counter(train_dataset)
            # print(f"Class distribution: {class_counts}")

            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker
            dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _resolve_token_budget(self, stage: str, dataset) -> int | None:
        """Resolve the configured token budget for 'eval' or 'test'.

        Returns:
            None — token-budget mode disabled, use legacy fixed batch size.
            int — the token budget to apply (either user-supplied or
                  freshly calibrated and cached on the trainer instance).
        """
        field = f'{stage}_max_tokens_per_batch'
        configured = getattr(self.custom_args, field, None)
        if configured is None:
            return None
        if isinstance(configured, int):
            return configured
        # 'auto': calibrate once, cache on the trainer.
        cache_attr = f'_token_budget_{stage}'
        if getattr(self, cache_attr, None) is None:
            lengths = _get_lengths(dataset, self.args.length_column_name)
            pad_multiple = getattr(self.data_collator, 'pad_to_multiple_of', None) or 1
            budget = calibrate_token_budget(
                model=self.model,
                lengths=lengths,
                pad_multiple=pad_multiple,
            )
            setattr(self, cache_attr, budget)
            print(f'[trainer] {stage} token budget calibrated: {budget}')
        return getattr(self, cache_attr)

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError('Trainer: evaluation requires an eval_dataset.')

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, '_eval_dataloader') and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description='evaluation')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='evaluation')

        budget = self._resolve_token_budget('eval', eval_dataset)
        dataloader_params = build_inference_dataloader_kwargs(
            dataset=eval_dataset,
            args=self.args,
            data_collator=data_collator,
            token_budget=budget,
        )

        if budget is None and not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params['sampler'] = (
                SequentialSampler(eval_dataset)
                if self.custom_args.eval_force_sequential
                else self._get_eval_sampler(eval_dataset)
            )
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test DataLoader. When ``test_max_tokens_per_batch`` is
        set, uses TokenBudgetBatchSampler; otherwise falls back to the
        legacy fixed-batch path with SequentialSampler / LengthGroupedSampler.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description='test')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='test')

        budget = self._resolve_token_budget('test', test_dataset)
        dataloader_params = build_inference_dataloader_kwargs(
            dataset=test_dataset,
            args=self.args,
            data_collator=data_collator,
            token_budget=budget,
        )

        if budget is None and not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params['sampler'] = (
                SequentialSampler(test_dataset)
                if self.custom_args.test_force_sequential
                else self._get_eval_sampler(test_dataset)
            )
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def _load_best_model(self):
        print('\nLoading the best model...')
        super()._load_best_model()

    def _pad_tensors_to_max_len(self, tensor, max_length):

        # If tensor contains -100, assume it's labels
        pad_ignore_index = -100
        contains_ignore_index = (tensor == pad_ignore_index).any().item()

        if contains_ignore_index:
            pad_token_id = pad_ignore_index
        else:
            if self.processing_class is not None and hasattr(self.processing_class, 'pad_token_id'):
                # If PAD token is not defined at least EOS token has to be defined
                pad_token_id = (
                    self.processing_class.pad_token_id
                    if self.processing_class.pad_token_id is not None
                    else self.processing_class.eos_token_id
                )
            else:
                if self.model.config.pad_token_id is not None:
                    pad_token_id = self.model.config.pad_token_id
                else:
                    raise ValueError(
                        'Pad_token_id must be set in the configuration of the model, in order to pad tensors'
                    )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _remove_unused_columns(self, dataset: 'datasets.Dataset', description: str | None = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        if getattr(self.custom_args, 'usable_columns', None):
            ignored_columns = [col for col in ignored_columns if col not in self.custom_args.usable_columns]

        if len(ignored_columns) > 0:
            dset_description = '' if description is None else f'in the {description} set'
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f'`{self.model.__class__.__name__}.forward` and have been ignored: {", ".join(ignored_columns)}.'
                f' If {", ".join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, '
                ' you can safely ignore this message.'
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "No columns in the dataset match the model's forward method signature. "
                f'The following columns have been ignored: [{", ".join(ignored_columns)}]. '
                'Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.'
            )

        if version.parse(datasets.__version__) < version.parse('1.4.0'):
            dataset.set_format(
                type=dataset.format['type'], columns=columns, format_kwargs=dataset.format['format_kwargs']
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    # Inject a custom logits processor if generation_whitelist is set
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
        **gen_kwargs,
    ) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None]:

        # Inject custom logits processor
        generation_whitelist = getattr(self.custom_args, 'generation_whitelist', None)
        if generation_whitelist is not None:
            processor = ConstrainedPrefixLogitsProcessor(self.processing_class, generation_whitelist)
            gen_kwargs['logits_processor'] = LogitsProcessorList([processor])

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)


class RandomTaskExclusionBatchSampler(BatchSampler):
    """
    A custom BatchSampler that:
    - Randomly excludes tasks per batch
    - Samples each example at most once per epoch
    - Continues until all tasks are exhausted (even if only 1 or 2 tasks remain)
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group dataset indices by task
        self.original_indices_by_task = {}
        task_ids = dataset['task_ids']
        for idx, t_id in enumerate(task_ids):
            self.original_indices_by_task.setdefault(t_id, []).append(idx)

        self.task_ids = list(self.original_indices_by_task.keys())

        # Optionally shuffle each task's list of indices
        for t_id in self.task_ids:
            random.shuffle(self.original_indices_by_task[t_id])

        # We'll define length in a minimal sense (approx. #batches in one epoch)
        self.total_examples = sum(len(v) for v in self.original_indices_by_task.values())
        self.num_batches = self.total_examples // self.batch_size

    def __iter__(self):
        # We'll copy out the index lists so we can remove from them
        indices_by_task = {t: lst[:] for t, lst in self.original_indices_by_task.items()}

        # We track how many examples we've used so far
        used_samples = 0

        while True:
            # Figure out which tasks still have indices left
            available_tasks = [t for t in self.task_ids if len(indices_by_task[t]) > 0]

            # If no tasks remain, we are done
            if len(available_tasks) == 0:
                break

            # Decide how many tasks to include this batch
            if len(available_tasks) <= 2:
                # If we have 1 or 2 tasks left, just include all of them
                included_tasks = available_tasks
            else:
                # We have more than 2 tasks -> do random sub-selection
                r = random.randint(2, len(available_tasks))
                included_tasks = random.sample(available_tasks, r)

            # Collect candidate indices from the included tasks
            candidate_indices = []
            for t_id in included_tasks:
                candidate_indices.extend(indices_by_task[t_id])

            # If not enough indices to form a full batch:
            if len(candidate_indices) < self.batch_size:
                if self.drop_last:
                    # If we must drop the final partial batch, break here
                    break
                else:
                    # We'll yield whatever is left
                    batch_indices = candidate_indices
            else:
                # Randomly sample a batch from the candidate pool
                batch_indices = random.sample(candidate_indices, self.batch_size)

            yield batch_indices

            # Remove these sampled indices from the index pools
            for idx in batch_indices:
                for t_id in included_tasks:
                    # If idx is in that task's pool, remove it
                    if idx in indices_by_task[t_id]:
                        indices_by_task[t_id].remove(idx)
                        break

            used_samples += len(batch_indices)

    def __len__(self):
        # This is just an approximate "max # of batches" in one epoch
        return self.num_batches
