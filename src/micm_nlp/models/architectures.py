# Standard Libraries

# PyTorch

# Transformers
from transformers import T5Config, T5ForConditionalGeneration

# from transformers.models.t5.modeling_t5 import T5_INPUTS_DOCSTRING, _CONFIG_FOR_DOC  # These are required in the forward docstring
# T5_INPUTS_DOCSTRING = T5_INPUTS_DOCSTRING or ""
# _CONFIG_FOR_DOC = _CONFIG_FOR_DOC or ""

# FlashAttention (optional)
try:
    from flash_attn.modules.mha import FlashSelfAttention
except ImportError:
    FlashSelfAttention = None


class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, flash_attn: bool = False):
        super().__init__(config)

        # Adjust Flash Attention
        if flash_attn:
            if FlashSelfAttention is None:
                raise RuntimeError(
                    'flash_attn requested but not installed. Install flash-attn or run with flash_attn=False.'
                )

            # Replace layer self-attention
            def replace_self_atten(layer, causal):
                original_self_attn = layer.layer[0].SelfAttention
                flash_self_attn = FlashSelfAttention(
                    embed_dim=original_self_attn.d_model,
                    num_heads=original_self_attn.n_heads,
                    causal=causal,
                    dropout=original_self_attn.dropout,
                )
                # Copy pretrained weights (q, k, v matrices and biases)
                flash_self_attn.Wq.weight.data = original_self_attn.q.weight.data.clone()
                flash_self_attn.Wk.weight.data = original_self_attn.k.weight.data.clone()
                flash_self_attn.Wv.weight.data = original_self_attn.v.weight.data.clone()
                flash_self_attn.Wq.bias.data = original_self_attn.q.bias.data.clone()
                flash_self_attn.Wk.bias.data = original_self_attn.k.bias.data.clone()
                flash_self_attn.Wv.bias.data = original_self_attn.v.bias.data.clone()

                layer.layer[0].SelfAttention = flash_self_attn

            # Replace encoder self-attention
            for layer in self.encoder.block:
                replace_self_atten(layer, causal=False)

            # Replace decoder self-attention (causal)
            for layer in self.decoder.block:
                replace_self_atten(layer, causal=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        # utils.p('CustomT5ForConditionalGeneration.forward()')

        kwargs.pop('num_items_in_batch', None)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs,
        )

    # # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     decoder_input_ids: Optional[torch.LongTensor] = None,
    #     decoder_attention_mask: Optional[torch.BoolTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     decoder_head_mask: Optional[torch.FloatTensor] = None,
    #     cross_attn_head_mask: Optional[torch.Tensor] = None,
    #     encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     task_ids: Optional[torch.LongTensor] = None,
    # ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
    #     r"""
    #     labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    #         Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
    #         config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
    #         labels in `[0, ..., config.vocab_size]`

    #     Returns:

    #     Examples:

    #     ```python
    #     >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

    #     >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    #     >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    #     >>> # training
    #     >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    #     >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    #     >>> outputs = model(input_ids=input_ids, labels=labels)
    #     >>> loss = outputs.loss
    #     >>> logits = outputs.logits

    #     >>> # inference
    #     >>> input_ids = tokenizer(
    #     ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
    #     ... ).input_ids  # Batch size 1
    #     >>> outputs = model.generate(input_ids)
    #     >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    #     >>> # studies have shown that owning a dog is good for you.
    #     ```"""
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    #     if head_mask is not None and decoder_head_mask is None:
    #         if self.config.num_layers == self.config.num_decoder_layers:
    #             # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
    #             decoder_head_mask = head_mask

    #     # Encode if needed (training, first prediction pass)
    #     if encoder_outputs is None:
    #         # Convert encoder inputs in embeddings if needed
    #         encoder_outputs = self.encoder(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             inputs_embeds=inputs_embeds,
    #             head_mask=head_mask,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #     elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
    #         encoder_outputs = BaseModelOutput(
    #             last_hidden_state=encoder_outputs[0],
    #             hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
    #             attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
    #         )

    #     hidden_states = encoder_outputs[0]

    #     if self.model_parallel:
    #         torch.cuda.set_device(self.decoder.first_device)

    #     print("\n=== FINAL BATCH KEYS INTO FORWARD ===")
    #     print("Model class in use:", type(model))
    #     print(f"Batch keys: {list(locals().keys())}")
    #     print(f"input_ids is None? {input_ids is None}")
    #     print(f"inputs_embeds is None? {inputs_embeds is None}")
    #     print(f"labels is None? {labels is None}")

    #     # print(labels)
    #     # if labels is not None:
    #     #     for label in labels:
    #     #         print(label)
    #     # else:
    #     #     print('labels is None')

    #     if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
    #         # get decoder inputs from shifting lm labels to the right
    #         decoder_input_ids = self._shift_right(labels)

    #     # Set device for model parallelism
    #     if self.model_parallel:
    #         torch.cuda.set_device(self.decoder.first_device)
    #         hidden_states = hidden_states.to(self.decoder.first_device)
    #         if decoder_input_ids is not None:
    #             decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
    #         if attention_mask is not None:
    #             attention_mask = attention_mask.to(self.decoder.first_device)
    #         if decoder_attention_mask is not None:
    #             decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    #     # Decode
    #     decoder_outputs = self.decoder(
    #         input_ids=decoder_input_ids,
    #         attention_mask=decoder_attention_mask,
    #         inputs_embeds=decoder_inputs_embeds,
    #         past_key_values=past_key_values,
    #         encoder_hidden_states=hidden_states,
    #         encoder_attention_mask=attention_mask,
    #         head_mask=decoder_head_mask,
    #         cross_attn_head_mask=cross_attn_head_mask,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #     )
    #     # print(decoder_outputs)

    #     sequence_output = decoder_outputs[0]

    #     # Set device for model parallelism
    #     if self.model_parallel:
    #         torch.cuda.set_device(self.encoder.first_device)
    #         self.lm_head = self.lm_head.to(self.encoder.first_device)
    #         sequence_output = sequence_output.to(self.lm_head.weight.device)

    #     if self.config.tie_word_embeddings:
    #         # Rescale output before projecting on vocab
    #         # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    #         sequence_output = sequence_output * (self.model_dim**-0.5)

    #     lm_logits = self.lm_head(sequence_output)

    #     loss = None
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss(ignore_index=-100)
    #         # move labels to correct device to enable PP
    #         labels = labels.to(lm_logits.device)
    #         loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    #         # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    #     if not return_dict:
    #         output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
    #         return ((loss,) + output) if loss is not None else output

    #     return Seq2SeqLMOutput(
    #         loss=loss,
    #         logits=lm_logits,
    #         past_key_values=decoder_outputs.past_key_values,
    #         decoder_hidden_states=decoder_outputs.hidden_states,
    #         decoder_attentions=decoder_outputs.attentions,
    #         cross_attentions=decoder_outputs.cross_attentions,
    #         encoder_last_hidden_state=encoder_outputs.last_hidden_state,
    #         encoder_hidden_states=encoder_outputs.hidden_states,
    #         encoder_attentions=encoder_outputs.attentions,
    #     )

    #     print("\n=== FINAL BATCH KEYS INTO FORWARD ===")
    #     print("Model class in use:", type(model))
    #     print(f"Batch keys: {list(locals().keys())}")
    #     print(f"input_ids is None? {input_ids is None}")
    #     print(f"inputs_embeds is None? {inputs_embeds is None}")
    #     print(f"labels is None? {labels is None}")
