import torch
from peft import (
    PeftModel,
    get_peft_config,
    get_peft_model,
)

from micm_nlp.models.xpe import (
    get_xpe_model,
    is_xpe_adapter_dir,
    is_xpe_config,
    load_xpe_pretrained,
)


class PEFT:
    prompt_encoder_key = 'default'

    @staticmethod
    def is_peft(base):
        peft = getattr(base._config, 'peft', None)
        pretrained = base._config.model.pretrained
        return (peft is not None and getattr(peft, 'peft_type', None) is not None) or (
            pretrained is not None and getattr(pretrained, 'adapter', None) is not None
        )

    @staticmethod
    def from_pretrained(base_model, path):
        if is_xpe_adapter_dir(path):
            return load_xpe_pretrained(base_model, path)
        return PeftModel.from_pretrained(base_model, path)

    @staticmethod
    def get_total_virtual_tokens(base):
        """Total virtual-token count injected by the active prompt encoder, or
        None if no prompt encoder is attached. Handles the ModuleDict-wrapped
        form that PEFT uses after adapter load.
        """
        prompt_encoder = getattr(base._model, 'prompt_encoder', None)
        if isinstance(prompt_encoder, torch.nn.ModuleDict) and PEFT.prompt_encoder_key in prompt_encoder:
            prompt_encoder = prompt_encoder[PEFT.prompt_encoder_key]
        return getattr(prompt_encoder, 'total_virtual_tokens', None) if prompt_encoder is not None else None

    @staticmethod
    def setup_model(base):

        pret = base._config.model.pretrained
        peft = getattr(base._config, 'peft', None)

        if pret and getattr(pret, 'adapter', None) is not None:
            print('Load Pretrained PEFT Model...')
            pret_adapter_path = base._get_pret_path(pret.adapter)
            base._model = PEFT.from_pretrained(base._model, pret_adapter_path)

        elif peft:
            # Load Cross Prompt Encoder
            if is_xpe_config(peft):
                print('Setup XPE PEFT Model...')
                base._model = get_xpe_model(base._model, peft)

            else:
                print('Setup PEFT Model...')
                peft_config = get_peft_config(dict(peft))
                base._model = get_peft_model(base._model, peft_config)

                # peft.mapping.get_peft_model
                #     peft.peft_model.PeftModelForSequenceClassification
                #         peft.peft_model.PeftModel
                #             peft.peft_model.PeftModel.add_adapter
                #                 peft.utils.others._prepare_prompt_learning_config
                #                 peft.peft_model.PeftModel._setup_prompt_encoder
                #                     peft.tuners.p_tuning.model.PromptEncoder(PromptEncoderConfig)
                #                 peft.peft_model.PeftModel.set_additional_trainable_modules(peft_config, adapter_name)
