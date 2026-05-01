"""The :class:`CrossPromptEncoder` module (XPE core)."""

from typing import TYPE_CHECKING

import torch
import torch.nn.init as init
from peft.utils.save_and_load import load_peft_weights

import micm_nlp.utils as utils
from micm_nlp.models.xpe.enums import CrossPromptEncoderReparameterizationType
from micm_nlp.models.xpe.heads import gen_attn_head, gen_lstm_head, gen_mlp_head

if TYPE_CHECKING:
    from micm_nlp.models.xpe.config import CrossPromptEncoderConfig


class CrossPromptEncoder(torch.nn.Module):
    """
    The CrossPromptEncoder is a neural network module designed to generate virtual token embeddings, supporting various embedding strategies.

    Args:
        config ([`CrossPromptEncoderConfig`]): The configuration of the cross prompt encoder.

    Example:

    ```py
    >>> from micm_nlp.models.cross_prompt_encoder import CrossPromptEncoder, CrossPromptEncoderConfig

    >>> config = CrossPromptEncoderConfig(
    ...     peft_type="P_TUNING",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     encoder_num_layers=12,
    ...     encoder_reparameterization_type="MLP",
    ...     encoder_hidden_size=768,
    ...     encoder_embedding_type="FULLY_SHARED",
    ... )

    >>> prompt_encoder = CrossPromptEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The Standard Soft Prompt (SPT) embedding layer (Skips XPE)
        - **xpe_embedding** (`torch.nn.Embedding`) -- The Cross Prompt Encoder (XPE) embedding layer (XPE input embeddings).
        - **xpe_head** (`torch.nn.Module`) -- The Cross Prompt Encoder (XPE) head of the prompt encoder if `encoder_reparameterization_type="MLP"`.
        - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the prompt encoder.
        - **encoder_type** (Union[[`CrossPromptEncoderReparameterizationType`], `str`]): The encoder type of the prompt encoder.

    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config: 'CrossPromptEncoderConfig'):
        super().__init__()
        self.token_dim = config.token_dim
        self.input_size = config.encoder_input_size or self.token_dim
        self.output_size = self.token_dim
        self.num_heads = config.encoder_num_heads
        self.encoder_num_layers = config.encoder_num_layers
        self.encoder_dropout = config.encoder_dropout
        self.hidden_size = config.encoder_hidden_size
        self.encoder_type = config.encoder_reparameterization_type
        self.embedding_init_type = config.encoder_embedding_init_type
        self.encoder_ratio = config.encoder_ratio

        # Number of total virtual tokens
        self.num_transformer_submodules = config.num_transformer_submodules
        self.total_virtual_tokens = config.num_virtual_tokens * self.num_transformer_submodules

        utils.p(f'[yellow]num_virtual_tokens: {config.num_virtual_tokens}[/yellow]')
        utils.p(f'[yellow]num_transformer_submodules: {self.num_transformer_submodules}[/yellow]')
        utils.p(f'[yellow]total_virtual_tokens: {self.total_virtual_tokens}[/yellow]')

        # Initialize Embeddings and XPE Head, even if pretrained weights are provided.
        # This ensures that the model is always in a valid state, even if pretrained model is missing some weights.
        self.init_state_dict_path = config.encoder_init_state_dict_path

        # freeze
        self.encoder_freeze = config.encoder_freeze
        self.embedding_freeze = config.encoder_embedding_freeze

        # normalization
        self.embedding_normalize = config.encoder_embedding_normalize
        self.embedding_normalize_max_norm = config.encoder_embedding_normalize_max_norm

        # virtual tokens for XPE and SPT (Standard Soft Prompt)
        self.xpe_virtual_tokens = 0
        self.spt_virtual_tokens = 0
        if self.encoder_ratio == 0:
            self.spt_virtual_tokens = self.total_virtual_tokens
        elif self.encoder_ratio == 1:
            self.xpe_virtual_tokens = self.total_virtual_tokens
        elif self.encoder_ratio > 0 and self.encoder_ratio < 1:
            self.xpe_virtual_tokens = max(
                1, min(self.total_virtual_tokens - 1, int(self.total_virtual_tokens * self.encoder_ratio))
            )
            self.spt_virtual_tokens = self.total_virtual_tokens - self.xpe_virtual_tokens
        else:
            raise ValueError(f'Unknown encoder ratio: {self.encoder_ratio}')

        # Print virtual tokens
        utils.p(f'[yellow]xpe_virtual_tokens: {self.xpe_virtual_tokens}[/yellow]')
        utils.p(f'[yellow]spt_virtual_tokens: {self.spt_virtual_tokens}[/yellow]')

        # Initialize embeddings if virtual tokens are present
        if self.spt_virtual_tokens:
            self.embedding = self.init_embeddings(self.spt_virtual_tokens, self.input_size)
            utils.p(f'[yellow]embedding: {self.embedding}[/yellow]')
        if self.xpe_virtual_tokens:
            self.xpe_embedding = self.init_embeddings(self.xpe_virtual_tokens, self.input_size)
            utils.p(f'[yellow]xpe_embedding: {self.xpe_embedding}[/yellow]')

        if self.encoder_ratio > 0:
            # Initialize encoder heads
            utils.p(f'[yellow]Randomly initialize {self.encoder_type} head:[/yellow]')

            if self.encoder_type == CrossPromptEncoderReparameterizationType.NONE:
                self.xpe_head = torch.nn.Identity()

            elif self.encoder_type == CrossPromptEncoderReparameterizationType.MLP:
                self.xpe_head = gen_mlp_head(
                    self.input_size, self.hidden_size, self.output_size, self.encoder_num_layers, self.encoder_dropout
                )

            elif self.encoder_type == CrossPromptEncoderReparameterizationType.LSTM:
                self.xpe_head = gen_lstm_head(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    num_layers=self.encoder_num_layers,
                    dropout=self.encoder_dropout,
                )

            elif self.encoder_type == CrossPromptEncoderReparameterizationType.ATTN:
                self.xpe_head = gen_attn_head(
                    num_heads=self.num_heads,
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    dropout=config.encoder_dropout,
                )

            else:
                raise ValueError('Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.')

            utils.p(f'[yellow]{self.xpe_head}[/yellow]')

        # Load pretrained weights if provided
        if self.init_state_dict_path:
            self.load_pretrained_state()

        # Freeze embeddings or entire encoder if required
        self.set_grad_requirements()

    def init_embeddings(self, num: int, dim: int):
        embedding = torch.nn.Embedding(num, dim)
        if self.embedding_init_type == 'xavier_uniform':
            init.xavier_uniform_(embedding.weight)
        elif self.embedding_init_type == 'xavier_normal':
            init.xavier_normal_(embedding.weight)
        elif self.embedding_init_type == 'hf_default':
            pass
        else:
            raise ValueError(f'Unknown initialization type: {self.embedding_init_type}')
        return embedding

    def print_trainable_layers(self):
        """
        Print all trainable layer names in the model.
        """
        trainable_params_count = 0
        trainable_layers = []
        trainable_data_norms = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_layers.append(name)
                trainable_data_norms.append(param.data.norm().detach().cpu())
                trainable_params_count += param.numel()
        utils.p('[red]CrossPromptEncoder - Trainable params:[/red]', f'{trainable_params_count:,}')
        utils.p(
            '[red]CrossPromptEncoder - Trainable data norm:[/red]',
            round(torch.stack(trainable_data_norms).mean().item(), 3),
        )
        utils.p('[red]CrossPromptEncoder - Trainable layers:[/red]', trainable_layers)

    def print_all_layers(self):
        """
        Print all layers in the model, grouped by trainable and non-trainable.
        """
        trainable_layers = []
        trainable_data_norms = []
        trainable_params_count = 0

        non_trainable_layers = []
        non_trainable_data_norms = []
        non_trainable_params_count = 0

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_layers.append(name)
                trainable_data_norms.append(param.data.norm().detach().cpu())
                trainable_params_count += param.numel()
            else:
                non_trainable_layers.append(name)
                non_trainable_data_norms.append(param.data.norm().detach().cpu())
                non_trainable_params_count += param.numel()

        # Print trainable parameters
        utils.p('[red]\nXPE Trainable Parameters:[/red]')
        utils.p(f'  Count: {trainable_params_count:,}')
        utils.p(
            f'  Avg. Data Norm: {round(torch.stack(trainable_data_norms).mean().item(), 3) if trainable_data_norms else "N/A"}'
        )
        utils.p('  Layers: ')
        for layer in trainable_layers:
            utils.p(f'    {layer}')

        # Print non-trainable parameters
        utils.p('[blue]\nXPE Frozen Parameters:[/blue]')
        utils.p(f'  Count: {non_trainable_params_count:,}')
        utils.p(
            f'  Avg. Data Norm: {round(torch.stack(non_trainable_data_norms).mean().item(), 3) if non_trainable_data_norms else "N/A"}'
        )
        utils.p('  Layers: ')
        for layer in non_trainable_layers:
            utils.p(f'    {layer}')

    def load_pretrained_state(self):
        """
        Load pretrained weights for embeddings and encoder heads (MLP or LSTM).
        Initialize embedding parameters only.
        """
        utils.p(f"\nInitialize CrossPromptEncoder from pretrained: '{self.init_state_dict_path}'")

        pretrained_state_dict = load_peft_weights(self.init_state_dict_path)
        pretrained_state_dict.pop('prompt_embeddings', None)
        pretrained_state_dict = {k.replace('prompt_encoder.', ''): v for k, v in pretrained_state_dict.items()}
        pretrained_state_dict = {k.replace('default.', ''): v for k, v in pretrained_state_dict.items()}
        utils.p('Pretrained parameters keys and shapes: ', {k: v.shape for k, v in pretrained_state_dict.items()})

        # Initialize embedding parameters
        pret_embedding_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('base_model')}
        utils.p('XPE parameters keys and shapes: ', {k: v.shape for k, v in self.state_dict().items()})
        for name, param in pret_embedding_dict.items():
            if name in self.state_dict():
                utils.p(f'[yellow]Initialize: {name}[/yellow]   {param.shape}')
                self.state_dict()[name].copy_(param)

            # # WARNING: Temporary workaround for loading old APE weights
            # elif name == 'embedding.weight' and self.encoder_ratio == 1:
            #     utils.p(f'[yellow]Initialize: {name}[/yellow]   {param.shape}')
            #     self.state_dict()['xpe_embedding.weight'].copy_(param)

            else:
                utils.p(f'⚠️ Pretrained parameter {name} not found in model. Skipping.')

    def set_grad_requirements(self):

        utils.p(f'[yellow]Setting embedding trainability to {not self.embedding_freeze}...[/yellow]')
        for name, param in self.named_parameters():
            if 'embedding' in name and 'original_module' not in name:
                param.requires_grad = False if self.embedding_freeze else True

        if self.encoder_ratio > 0:
            utils.p(f'[yellow]Setting encoder trainability to {not self.encoder_freeze}...[/yellow]')
            for name, param in self.named_parameters():
                if 'xpe_head' in name and 'original_module' not in name:
                    param.requires_grad = False if self.encoder_freeze else True

    def forward(self, indices, task_ids=None):
        """
        Forward pass for the PromptEncoder.

        Args:
            indices (torch.Tensor): Indices of virtual tokens. Shape: (batch_size, num_virtual_tokens)
            task_ids (torch.Tensor): Task IDs for each example in the batch. Shape: (batch_size,)
        """
        assert indices.size(1) == self.total_virtual_tokens, (
            f'indices.shape[1] ({indices.size(1)}) does not match total_virtual_tokens ({self.total_virtual_tokens})'
        )

        if self.encoder_ratio == 0:
            spt_embeds = self.embedding(indices)  # Shape: (batch_size, num_virtual_tokens, token_dim)
            return spt_embeds

        if self.encoder_ratio == 1:
            xpe_embeds = self.xpe_embedding(indices)
            xpe_encoded_embeds = self.xpe_head(xpe_embeds)
            return xpe_encoded_embeds

        if self.encoder_ratio > 0 and self.encoder_ratio < 1:
            spt_embeds = self.embedding(indices[:, : self.spt_virtual_tokens])
            # xpe_embedding is 0-indexed (sized xpe_virtual_tokens); the slice carries
            # values [spt..total-1], so shift back to [0..xpe-1] for a valid lookup.
            xpe_embeds = self.xpe_embedding(indices[:, self.spt_virtual_tokens :] - self.spt_virtual_tokens)
            xpe_encoded_embeds = self.xpe_head(xpe_embeds)
            embeds = torch.cat([spt_embeds, xpe_encoded_embeds], dim=1)
            return embeds

        raise ValueError(f'Unknown encoder ratio: {self.encoder_ratio}')

    def get_device(self):
        embedding = self.embedding if hasattr(self, 'embedding') else self.xpe_embedding
        return embedding.weight.device

    def normalize_embeddings(self):
        norms_after = []
        if self.embedding_normalize:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if 'embedding' in name and param.ndim == 2 and param.requires_grad:
                        norm = param.norm(dim=-1, keepdim=True).clamp(min=1e-6)

                        if self.embedding_normalize == 'unit':
                            param.div_(norm)
                        elif self.embedding_normalize == 'clip':
                            scale = norm.clamp(max=self.embedding_normalize_max_norm) / norm
                            param.mul_(scale)

                        # Store mean norm after normalization
                        norms_after.append(param.norm(dim=-1).mean().item())
        return sum(norms_after) / len(norms_after) if norms_after else 0.0
