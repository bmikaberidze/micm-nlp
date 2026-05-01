import copy
import os
import uuid
from typing import ClassVar

import torch

import micm_nlp.utils as utils
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.enums import DeviceSE, ModeSE, PretSourceSE
from micm_nlp.models.peft import PEFT
from micm_nlp.path import evals_dir, find_dirs_by_prefix, models_dir


class MODEL:
    checkpoint_pref = 'checkpoint-'

    # Modules searched (in order) when resolving class names from YAML.
    CLS_SOURCE_MODULES: ClassVar[list[str]] = ['transformers']

    def __init__(self, config):
        self._config = copy.deepcopy(config)
        self._set_paths()
        self._setup_model()

    def reinit(self, config):
        self.__init__(config)

    @property
    def hf(self):
        return self._model

    @hf.setter
    def hf(self, value):
        self._model = value

    # -- Path management ---------------------------------------------------

    def _get_init_path(self, name=''):
        init_path = f'{models_dir()}/{self._config.model.architecture}'
        return f'{init_path}/{name}' if name else init_path

    def _set_paths(self):

        self._set_name()

        self.eval_path = str(evals_dir() / 'runs' / self.name)
        self.logs_path = f'{self.eval_path}/logs'

        mode = self._config.mode
        if mode in [ModeSE.FINETUNE, ModeSE.EVALUATE, ModeSE.TEST]:
            self.pret_path = self._get_pret_path()

        if mode in [ModeSE.TRAIN, ModeSE.FINETUNE, ModeSE.EVALUATE]:
            if mode == ModeSE.TRAIN:
                self.path = self._get_init_path(self.name)

            elif mode in [ModeSE.FINETUNE, ModeSE.EVALUATE]:
                task_name = self._config.task.name
                if self._config.model.pretrained.source == PretSourceSE.HUGGINGFACE:
                    self.path = self._get_init_path(self.pret_path)
                else:
                    self.path = self.pret_path
                self.path = f'{self.path}/{task_name}/{self.name}'

            MODEL.store_path_by_uuid4_in_envs(self.uuid4, self.path)

    def _set_name(self):

        self.uuid4 = str(uuid.uuid4())
        self._config.model.uuid4 = self.uuid4

        pret = self._config.model.pretrained
        self.pret_uuid4 = self._get_pret_uuid4()
        pret_info = self.pret_uuid4 if self.pret_uuid4 else pret.name.replace('/', '|')

        slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
        slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)

        name_params = [
            self.uuid4,
            pret_info,
            slurm_job_id,
            slurm_task_id,
            self._config.ds.dirs.replace('/', '|'),
            DATASET.get_name(self._config),
        ]

        mode = self._config.mode
        if mode in [ModeSE.TRAIN, ModeSE.FINETUNE]:
            targs = self._config.training_args.args
            self.effective_batch_size = targs.per_device_train_batch_size * targs.gradient_accumulation_steps
            name_params.extend(
                [
                    targs.num_train_epochs,
                    self.effective_batch_size,
                ]
            )

        self.name = '_'.join([str(p) for p in name_params if p is not None])

    def _get_pret_uuid4(self):
        pret_uuid4 = None
        pret = self._config.model.pretrained
        if pret.source == PretSourceSE.LOCAL:
            pret_uuid4 = getattr(pret, 'uuid4', None)
            if not pret_uuid4 and pret.name:
                pret_uuid4 = MODEL.extract_uuid_from_name(pret.name)
        return pret_uuid4

    def _get_pret_path(self, pret=None):
        pret_path = None
        pret = pret if pret else self._config.model.pretrained
        if pret.source == PretSourceSE.HUGGINGFACE:
            pret_path = pret.name
        elif pret.source == PretSourceSE.LOCAL:
            pret_path = self._get_init_path(pret.name) if pret.name else MODEL.find_path_by_uuid4(pret.uuid4)
            pret_path = MODEL.get_last_checkpoint_path(pret_path, pret.checkpoint)
        return pret_path

    # -- Model setup -------------------------------------------------------

    def _setup_model(self):
        print('Setup Model...')

        if self._config.mode in [ModeSE.FINETUNE, ModeSE.EVALUATE, ModeSE.TEST]:
            if not self.pret_path:
                raise Exception('Pretrained model path not set')

        if not hasattr(self, '_model'):
            if self._config.mode == ModeSE.TRAIN:
                self._model = self._init_from_scratch()
            else:
                self._model = self._load_pretrained(**self._task_derived_kwargs())

            if self._model is None:
                raise Exception('Model not set')

            if getattr(self._config.model.pretrained, 'save_as', None) == PretSourceSE.HUGGINGFACE:
                print(f'Saving model to {self.path}')
                self._model.save_pretrained(self.path)

            PEFT.setup_model(self) if PEFT.is_peft(self) else None

            self._set_model_properties()
            self._setup_model_device()

        print('Model is Loaded!')

    def _setup_model_device(self):
        device = DeviceSE.CUDA if torch.cuda.is_available() else DeviceSE.CPU
        print(f'\nSending Model to available {device} device...')
        self.device = torch.device(device)
        self._model.to(self.device)

    def _task_derived_kwargs(self):
        """Task-aware kwargs injected into from_pretrained (single source of truth for
        values already declared elsewhere in the config, e.g. num_labels ← ds.label.number).
        Class selection stays entirely in YAML via `model.pretrained.cls`.
        """
        task = self._config.task.category
        if task in ['text_classification', 'text_pair_classification', 'token_classification']:
            return {'num_labels': self._config.ds.label.number}
        return {}

    def _resolve_cls(self, cls_name, yaml_path):
        return utils.resolve_cls(cls_name, self.CLS_SOURCE_MODULES, yaml_path)

    def _pretrained_args(self):
        """Extra kwargs from `model.pretrained.args`, splatted into from_pretrained.
        Use this for knobs like torch_dtype, device_map, or config overrides
        (e.g. use_mems_eval: false for XLNet).
        """
        args = getattr(self._config.model.pretrained, 'args', None)
        return {k: v for k, v in dict(args).items() if v is not None} if args else {}

    def _load_pretrained(self, **task_kwargs):
        pret = self._config.model.pretrained
        ModelCls = self._resolve_cls(getattr(pret, 'cls', None), 'model.pretrained.cls')
        return ModelCls.from_pretrained(self.pret_path, **task_kwargs, **self._pretrained_args())

    def _init_from_scratch(self):
        init = self._config.model.init
        if init is None:
            raise ValueError('model.init is required in TRAIN mode.')

        HFConfig = self._resolve_cls(
            getattr(init.config, 'cls', None) if init.config else None,
            'model.init.config.cls',
        )
        ModelCls = self._resolve_cls(getattr(init, 'cls', None), 'model.init.cls')
        config_args = dict(init.config.args) if init.config and init.config.args else {}
        config_args = {k: v for k, v in config_args.items() if v is not None}

        hf_config = HFConfig(
            vocab_size=self._config.tokenizer.vocab_size,
            **config_args,
        )
        return ModelCls(config=hf_config)

    # -- Model properties --------------------------------------------------

    def _set_model_properties(self):
        self._set_max_length()
        self._set_embedding_dim()
        self.param_size = sum(p.numel() for p in self._model.parameters())
        self.trainable_param_size = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self.trainable_param_size_ratio = self.trainable_param_size / self.param_size * 100
        self._config.model.param_size = f'{self.param_size:,}'
        self._config.model.trainable_param_size = f'{self.trainable_param_size:,}'
        self._config.model.trainable_param_size_ratio = f'{self.trainable_param_size_ratio:.4f}'
        self.label_pad_id = getattr(self._config.ds.label, 'padded', -100)

    def _set_max_length(self):
        self.max_length = getattr(
            self._model.config,
            'max_position_embeddings',
            getattr(
                self._model.config,
                'n_positions',
                getattr(self._model.config, 'mem_len', getattr(self._model.config, 'max_length', None)),
            ),
        )
        if self.max_length is None:
            print('Warning: max_length is not set in the model configuration.')

    def _set_embedding_dim(self):
        self.embedding_dim = getattr(self._model.config, 'hidden_size', getattr(self._model.config, 'd_model', None))
        if self.embedding_dim is None:
            print('Warning: embedding_dim is not set in the model configuration.')

    # -- Debug / print -----------------------------------------------------

    def print_named_parameters(self, requires_grad=None, model=None):
        model = model if model else self._model
        for name, param in model.named_parameters():
            if requires_grad is None or requires_grad == param.requires_grad:
                print(f'name: {name} | grad: {param.requires_grad} | mean: {param.mean().item()}')

    # -- Static checkpoint / path utilities ---------------------------------

    @staticmethod
    def get_base_model(model):
        if hasattr(model, 'get_base_model'):
            return model.get_base_model()
        elif hasattr(model, 'base_model'):
            return model.base_model
        return model

    @staticmethod
    def get_last_checkpoint(path):
        dirs = os.listdir(path)
        checkpoints = [int(d.split('-')[-1]) for d in dirs if d.startswith(MODEL.checkpoint_pref)]
        return max(checkpoints) if checkpoints else None

    @staticmethod
    def get_last_checkpoint_path(path, last_checkpoint=None):
        last_checkpoint = last_checkpoint if last_checkpoint else MODEL.get_last_checkpoint(path)
        return os.path.join(path, f'{MODEL.checkpoint_pref}{last_checkpoint}') if last_checkpoint else path

    @staticmethod
    def get_last_checkpoint_path_by_uuid4(source_model_uuid4):
        model_path = MODEL.find_path_by_uuid4(source_model_uuid4)
        return MODEL.get_last_checkpoint_path(model_path)

    @staticmethod
    def store_path_by_uuid4_in_envs(uuid4, path):
        os.environ[f'MODEL_PATH_{uuid4}'] = path
        utils.p(f'\n[green]Model path is stored in envvar MODEL_PATH_{uuid4}:[/green]\n {path}')

    @staticmethod
    def get_path_by_uuid4_from_envs(uuid4):
        return os.environ.get(f'MODEL_PATH_{uuid4}', None)

    @staticmethod
    def find_path_by_uuid4(uuid4, root_path=None):
        path = MODEL.get_path_by_uuid4_from_envs(uuid4)
        if not path:
            root_path = root_path if root_path else str(models_dir())
            paths = find_dirs_by_prefix(root_dir=root_path, dir_prefix=str(uuid4))
            if len(paths) > 1:
                raise Exception('By this UUID4 multiple model directories are found!')
            if not len(paths):
                raise Exception('By this UUID4 no model directory is found!')
            path = paths[0]
            MODEL.store_path_by_uuid4_in_envs(uuid4, path)
        return path

    @staticmethod
    def get_uuid_path_dict(root_path=None):
        from tqdm import tqdm

        uuid_path_dict = {}
        root_path = root_path if root_path else str(models_dir())
        for dirpath, dirnames, _ in tqdm(os.walk(root_path), desc='Walking through directories'):
            for dirname in dirnames:
                key = dirname.split('_')[0]
                uuid_path_dict[key] = os.path.join(dirpath, dirname)
        return uuid_path_dict

    @staticmethod
    def extract_uuid_from_name(name):
        uuid = None
        prefix = name[:38]
        if utils.is_valid_uuid(prefix):
            uuid = prefix
        return uuid
