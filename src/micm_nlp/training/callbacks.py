"""Trainer callbacks.

- ``CustomEarlyStoppingCallback`` — early stopping decoupled from checkpoint
  selection and gated by an ``early_stopping_after`` floor, so a run cannot stop
  during its first fraction of steps.
- ``NormalizePromptEncoderEmbeddings`` — normalises prompt-encoder embeddings each
  step and logs the mean norm.
- ``ParamNormLogger`` — logs parameter and parameter-update norms to W&B.
- ``EmptyCudaCacheCallback`` — periodic ``torch.cuda.empty_cache()``.
- ``DownstreamFineTuningCallback`` — fine-tunes on downstream tasks at evaluation
  and save points.

``NormalizePromptEncoderEmbeddings`` only registers when
``peft.encoder_embedding_normalize`` is set to ``'unit'`` or ``'clip'``; without it
the callback would be a no-op that still logged a zero norm every step.

.. note::
   Until 0.2.0 this callback was never registered at all — the trainer looked for its
   settings under ``task.peft`` while ``peft`` is a top-level block. Results produced
   before that fix did **not** normalise, whatever their config said.
"""

import torch
import wandb
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from micm_nlp.enums import PretSourceSE
from micm_nlp.models.xpe import CrossPromptEncoder

# from transformers.trainer_utils import get_last_checkpoint


class EmptyCudaCacheCallback(TrainerCallback):
    """A custom callback that empties the CUDA cache at specified intervals."""

    def __init__(self, empty_cache_steps=None):
        """:param empty_cache_steps: free the cache every N steps; ``None`` disables."""
        self.empty_cache_steps = empty_cache_steps
        self.device = torch.cuda.current_device()
        self.gb_coeff = 1024 * 1024 * 1024

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Empty the CUDA allocator cache on the configured step interval."""
        if self.empty_cache_steps and state.global_step % self.empty_cache_steps == 0:
            print('Empty CUDA cache!')
            torch.cuda.empty_cache()


class DownstreamFineTuningCallback(TrainerCallback):
    """
    A custom callback that performs downstream fine-tuning on evaluation.
    """

    def __init__(self, config, model_path):
        """:param config: the run config; ``eval.downstream_tasks`` steers this callback.
        :param model_path: directory of the checkpoint to fine-tune from.
        """
        self._is_training = False
        self._model_path = model_path
        self._config = config

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Mark training as in progress, so evaluations can tell which phase they are in."""
        self._is_training = True

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Mark training as finished."""
        self._is_training = False

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Run the downstream fine-tuning probe, if this evaluation warrants one.

        Two cases qualify: the step-0 evaluation of an already-pretrained model
        (a baseline before any training), and any evaluation outside training.
        """
        pret = self._config.model.pretrained
        # print('\n on_evaluate >>>>>>>', self._is_training, state.global_step, pret.name, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        if (state.global_step == 0 and (pret.name or pret.time_id)) or (
            state.global_step != 0 and not self._is_training
        ):
            self.finetune_on_downstream_tasks(state.global_step)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Probe the just-saved checkpoint on the downstream tasks.

        Skipped at step 0, and during training when
        ``eval.downstream_tasks.not_while_training`` is set -- the probe is
        expensive, so a run can defer every one of them to the end.
        """
        # print('\n on_save >>>>>>>', self._is_training, state.global_step, self._config.model.pretrained.name, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        not_while_training = self._config.eval.downstream_tasks.not_while_training
        if state.global_step != 0 and not (self._is_training and not_while_training):
            self.finetune_on_downstream_tasks(state.global_step)

    def finetune_on_downstream_tasks(self, state_global_step):
        """
        Finetune the model on all downstream tasks.
        """

        from micm_nlp.config import CONFIG
        from micm_nlp.models.scripts.run import run as finetune

        for conf_path in self._config.eval.downstream_tasks.config_paths:
            # Load Downstream Tasks Configuration
            config = CONFIG.from_yaml(conf_path)
            # Set up the pretrained model for evaluatiing it on downstream tasks
            # First, copy the tokenizer and model configs from the main config
            config.tokenizer = self._config.tokenizer.model_copy(deep=True)
            config.model = self._config.model.model_copy(deep=True)
            # If state_global_step is 0, it means that the model is not trained yet, and we can only evaluate the starting point pre-trained model we are finetuning on
            # If state_global_step is more than 0, it means that the model was trained, and we can evaluate it
            #   In case the training is ongoing, we can use state_global_step as the model's last checkpoint
            #   In case the training is finished, we don't set checkpoint, and the model seeks the best checkpoint automatically
            if state_global_step:
                config.model.pretrained.source = PretSourceSE.LOCAL
                config.model.pretrained.name = self._model_path.split(f'/{self._config.model.architecture}/')[-1]
                config.model.pretrained.checkpoint = state_global_step if self._is_training else None
            # print('finetune_on_downstream_tasks', self._is_training, state_global_step, config.model.pretrained.checkpoint)
            # return
            finetune(config)


# class PromptEncoderSaver(TrainerCallback):
#     '''
#     A custom callback that saves the prompt encoder.
#     '''
#     def on_save(self, args, state, control, model=None, **kwargs):
#         output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
#         prompt_encoder_path = os.path.join(output_dir, PEFT.prompt_encoder_file)

#         prompt_encoder = getattr(model, 'prompt_encoder', None)

#         if prompt_encoder:
#             torch.save(prompt_encoder.state_dict(), prompt_encoder_path)
#             print(f"✅ Saved prompt embeddings at step {state.global_step}")


class ParamNormLogger(TrainerCallback):
    """Log the mean parameter norm and mean per-step update norm to wandb.

    A training-stability probe: a norm that grows without bound, or an update norm
    that collapses, shows up here before it shows up in the loss. Only trainable
    parameters are measured -- with PEFT that is the adapter, not the backbone.

    It keeps a CPU copy of every trainable parameter between steps in order to
    difference them, so memory scales with the trainable parameter count.
    """

    def __init__(self):
        """Start with no previous step recorded; the first step logs no update norm."""
        self.prev_params = {}

    def on_step_end(self, args, state, control, **kwargs):
        """Measure this step's parameter norms and log them."""
        model = kwargs['model']
        param_norms = []
        param_update_norms = []

        new_prev_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                current = param.data.detach().cpu()
                param_norms.append(current.norm())

                if name in self.prev_params:
                    update = (current - self.prev_params[name]).norm()
                    param_update_norms.append(update)

                new_prev_params[name] = current.clone()
        self.prev_params = new_prev_params

        wandb.log(
            {
                'train/param_norm': torch.stack(param_norms).mean().item(),
                'train/param_update_norm': torch.stack(param_update_norms).mean().item() if param_update_norms else 0.0,
            }
        )


class NormalizePromptEncoderEmbeddings(TrainerCallback):
    """
    A custom callback that normalizes the prompt encoder embeddings.
    """

    def on_step_end(self, args, state, control, **kwargs):
        """Renormalise the prompt encoder's embeddings and log their mean norm.

        Does nothing unless the active adapter's prompt encoder is a
        :class:`~micm_nlp.models.xpe.encoder.CrossPromptEncoder`; the encoder itself
        decides what normalisation means, from its own config.
        """
        # NB: must be on_step_end (fires + receives model via kwargs, like
        # ParamNormLogger). on_optimizer_step did NOT pass `model` under
        # transformers 4.48 -> the body silently early-returned and no
        # normalization ever happened.
        model = kwargs.get('model')
        if model is None:
            return
        active_adapter = getattr(model, 'active_adapter', None)
        if not hasattr(model, 'prompt_encoder'):
            return
        if isinstance(model.prompt_encoder, torch.nn.ModuleDict) and active_adapter in model.prompt_encoder:
            prompt_encoder = model.prompt_encoder[active_adapter]
        else:
            prompt_encoder = getattr(model.prompt_encoder, active_adapter, None)
        if not prompt_encoder:
            return
        if isinstance(prompt_encoder, CrossPromptEncoder):
            mean_norm = prompt_encoder.normalize_embeddings()
            wandb.log({'train/xpe_embedd_norm': mean_norm})


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    """Early stopping decoupled from model selection, gated by an
    ``early_stopping_after`` floor (fraction of max_steps before stopping is
    allowed).

    The monitored signal is chosen by ``early_stopping_metric``:

    - ``'metric_for_best_model'`` (sentinel): delegate to the Trainer's
      ``args.metric_for_best_model`` + ``args.greater_is_better`` — i.e. stop
      on the same metric used to pick the best checkpoint.
    - any other string (e.g. ``'eval_loss'``): treat it as a literal metric
      key; direction inferred (``'loss'`` in the name → lower-is-better, else
      greater-is-better). This preserves the original eval_loss behavior and
      keeps early stopping SEPARABLE from selection.

    Default ``'eval_loss'`` reproduces the pre-existing behavior.
    """

    SENTINEL_BEST = 'metric_for_best_model'

    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.0,
                 early_stopping_after=0.5, early_stopping_metric='eval_loss'):
        """:param early_stopping_patience: evaluations without improvement before stopping.
        :param early_stopping_threshold: how much counts as an improvement.
        :param early_stopping_after: fraction of ``max_steps`` that must elapse
            before stopping is allowed at all, so a slow start is not cut short.
        :param early_stopping_metric: metric to monitor -- a literal key, or the
            sentinel ``'metric_for_best_model'`` to follow the Trainer's own
            selection metric.
        """
        super().__init__(
            early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold
        )
        self.best_metric = None
        self.patience_counter = 0
        self.early_stopping_after = early_stopping_after
        self.early_stopping_metric = early_stopping_metric or 'eval_loss'

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Consider stopping, once past the ``early_stopping_after`` floor.

        Evaluations before that step return unchanged, so patience is not spent
        while the model is still warming up.
        """
        current_step = state.global_step
        required_min_step = int(state.max_steps * self.early_stopping_after)

        if current_step < required_min_step:
            # Skip early stopping before threshold step
            return control

        # Resolve which metric to monitor (decoupled from selection by default).
        if self.early_stopping_metric == self.SENTINEL_BEST:
            metric_to_check = args.metric_for_best_model or 'eval_loss'
            greater_is_better = bool(args.greater_is_better)
        else:
            metric_to_check = self.early_stopping_metric
            greater_is_better = 'loss' not in metric_to_check.lower()
        if not metric_to_check.startswith('eval_'):
            metric_to_check = f'eval_{metric_to_check}'

        current = metrics.get(metric_to_check)
        if current is None:
            # Robustness: the toolkit may expand metric_for_best_model into a
            # task-prefixed key (e.g. 'eval_tune.<cfg>/accuracy'). Match by suffix.
            suffix = metric_to_check.split('/')[-1]
            cands = [v for k, v in metrics.items()
                     if k.startswith('eval_') and k.endswith(suffix)]
            current = cands[0] if cands else None

        if self.best_metric is None or current is None:
            self.best_metric = current
            return control

        if greater_is_better:
            improved = current > self.best_metric + self.early_stopping_threshold
        else:
            improved = current < self.best_metric - self.early_stopping_threshold

        if improved:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(f'[EarlyStopping] Triggered at step {current_step} '
                      f'on {metric_to_check} (greater_is_better={greater_is_better})')
                control.should_training_stop = True

        print(f'\nEarly stopping patience counter: {self.patience_counter}/{self.early_stopping_patience}\n')

        return control
