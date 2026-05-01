import torch
import wandb
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from micm_nlp.enums import PretSourceSE, TaskCatSE
from micm_nlp.models.xpe import CrossPromptEncoder

# from transformers.trainer_utils import get_last_checkpoint


def get_preprocess_logits_for_metrics(config, num_virtual_tokens=None):
    print('Get preprocess_logits_for_metrics function...')
    pred_axis = config.task.preproc_rules.prediction_axis
    is_log_likelihood = 'log_likelihood' in config.task.metric_groups[0].metrics[0]
    is_causal_lm = config.task.category == TaskCatSE.TEXT_GENERATION
    # Symmetric half of the runner's shift_labels_by auto-inject: when peft's
    # CausalLM/Seq2SeqLM forward prepends virtual-token labels internally, logits
    # come out at L+n while batch labels stay at L. Trim the prefix here. The
    # shape guard below keeps this a no-op for every other wiring (TokenCls with
    # a shift-labels collator, non-peft, LoRA, etc.) — no task-category branch needed.
    trim_prefix = int(num_virtual_tokens) if num_virtual_tokens else 0

    def preprocess_logits_for_metrics(logits, labels):
        # Handle models like T5 returning multiple outputs
        # Extract the first element, which contains the logits tensor
        if isinstance(logits, tuple):
            logits = logits[0]

        if trim_prefix and labels is not None and logits.shape[-2] == labels.shape[-1] + trim_prefix:
            logits = logits[..., trim_prefix:, :]

        if torch.isnan(logits).any():
            print('⚠️ NAN detected in logits!')
        if labels is not None and torch.isnan(labels).any():
            print('⚠️ NAN detected in labels!')

        if is_log_likelihood:
            # Shift for causal LM: position t predicts token t+1
            shift_logits = logits[..., :-1, :].contiguous()  # (batch, seq-1, vocab)
            shift_labels = labels[..., 1:].contiguous()  # (batch, seq-1)

            # Compute log softmax over vocabulary
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            # Gather log-probs for actual label tokens
            # Replace -100 with 0 temporarily for gather (will mask out later)
            labels_for_gather = shift_labels.clone()
            labels_for_gather[labels_for_gather == -100] = 0

            # Gather log-prob of the label token at each position
            label_log_probs = log_probs.gather(-1, labels_for_gather.unsqueeze(-1)).squeeze(-1)

            # Create mask for valid (non -100) positions
            mask = (shift_labels != -100).float()

            # Zero out log-probs at masked positions
            label_log_probs = label_log_probs * mask

            # # Stack for compute_metrics
            # return torch.stack([label_log_probs, mask], dim=-1)

            # Aggregate
            sequence_ll = label_log_probs.sum(dim=-1)  # (batch,) - total LL per sample
            sequence_lengths = mask.sum(dim=-1)  # (batch,) - target length per sample

            # Return shape (batch, 2)
            return torch.stack([sequence_ll, sequence_lengths], dim=-1)

        else:
            if is_causal_lm:
                # Causal-LM shift: logits at position t predict token t+1,
                # so the prediction for label at position t comes from logits at t-1.
                shift_logits = logits[..., :-1, :]
                predictions = torch.argmax(shift_logits, dim=pred_axis).to(torch.long)
                pad = torch.zeros_like(predictions[..., :1])
                return torch.cat([pad, predictions], dim=-1)
            predictions = torch.argmax(logits, dim=pred_axis)
            return predictions.to(torch.long)

    return preprocess_logits_for_metrics


class EmptyCudaCacheCallback(TrainerCallback):
    """A custom callback that empties the CUDA cache at specified intervals."""

    def __init__(self, empty_cache_steps=None):
        self.empty_cache_steps = empty_cache_steps
        self.device = torch.cuda.current_device()
        self.gb_coeff = 1024 * 1024 * 1024

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.empty_cache_steps and state.global_step % self.empty_cache_steps == 0:
            print('Empty CUDA cache!')
            torch.cuda.empty_cache()


class DownstreamFineTuningCallback(TrainerCallback):
    """
    A custom callback that performs downstream fine-tuning on evaluation.
    """

    def __init__(self, config, model_path):
        self._is_training = False
        self._model_path = model_path
        self._config = config

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._is_training = True

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._is_training = False

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pret = self._config.model.pretrained
        # print('\n on_evaluate >>>>>>>', self._is_training, state.global_step, pret.name, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        if (state.global_step == 0 and (pret.name or pret.time_id)) or (
            state.global_step != 0 and not self._is_training
        ):
            self.finetune_on_downstream_tasks(state.global_step)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
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
    def __init__(self):
        self.prev_params = {}

    def on_step_end(self, args, state, control, **kwargs):

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

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
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
            wandb.log({'train/ape_embedd_norm': mean_norm})


class LossEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.0, early_stopping_after=0.5):
        super().__init__(
            early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold
        )
        self.best_metric = None
        self.patience_counter = 0
        self.early_stopping_after = early_stopping_after

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = 'eval_loss'
        current_step = state.global_step
        required_min_step = int(state.max_steps * self.early_stopping_after)

        if current_step < required_min_step:
            # Skip early stopping before threshold step
            return control

        current = metrics.get(metric_to_check)

        if self.best_metric is None or current is None:
            self.best_metric = current
            return control

        if current < self.best_metric - self.early_stopping_threshold:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(f'[EarlyStopping] Triggered at step {current_step}')
                control.should_training_stop = True

        print(f'\nEarly stopping patience counter: {self.patience_counter}/{self.early_stopping_patience}\n')

        return control
