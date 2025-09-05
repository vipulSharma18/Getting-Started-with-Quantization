"""
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
"""
# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os
from typing import Any, Optional, Union
import torch
from .cache_optim import Cache
from .masking_optim import create_masks_for_generate
from transformers import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

ALL_STATIC_CACHE_IMPLEMENTATIONS = ("static", "offloaded_static")


class GenerationMixinCustom:
    def _cache_dependant_input_preparation(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Generic cache-dependent input preparation
        The code is put in a separate function to allow granular unit testing
        as it needs a different implementation to be exportable.
        """
        if (
            inputs_embeds is not None  # Exception 1
            or (cache_position[-1] >= input_ids.shape[1])  # Exception 3
        ):
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]
        return inputs_embeds, input_ids


    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. It includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        model_inputs["cache_position"] = cache_position

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        # 3. Prepare base model inputs
        input_ids_key = "input_ids"
        # `clone` calls in this function ensure a consistent stride. See #32227
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
        model_inputs["inputs_embeds"] = None

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = None
        attention_mask_key = "attention_mask"
        position_ids_key = "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

        # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
        # pass)
        if (
            isinstance(past_key_values, Cache)
            and past_key_values.is_compileable
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
            base_model = getattr(self, self.base_model_prefix, self)
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
            if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
                causal_mask_creation_function = getattr(
                    decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )

            # If it's not defined, it means the model uses the new general mask API
            if causal_mask_creation_function is None:  # can't be found
                token_type_ids = model_inputs.get("token_type_ids")
                position_ids = model_inputs.get(position_ids_key)
                # Some models may overwrite the general one
                causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
                attention_mask = causal_mask_creation_function(
                    config=self.config,
                    # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
                    input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs
        model_inputs.pop("labels", None)
        return model_inputs

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs = None,
    ) -> tuple[torch.Tensor, Optional[str], dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        input_name = self.main_input_name
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}
        # 2. model_input_name (input_ids) is passed as kwarg
        # and `inputs` is None, so use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        inputs = inputs_kwarg
        return inputs, input_name, model_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        if "past_key_values" in outputs:
            cache_name = "past_key_values"
            model_kwargs[cache_name] = getattr(outputs, "past_key_values")

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        return model_kwargs

    def _get_stopping_criteria(
        self,
        generation_config,
        tokenizer = None,
        **kwargs,
    ):
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
        return criteria

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        cache_position = torch.ones(seq_length, dtype=torch.int64, device=device).cumsum(0) - 1
        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            past_length = cache.get_seq_length()
            cache_position = cache_position[past_length:]
        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def _supports_logits_to_keep(self) -> bool:
        """
        Return True if the current model supports the keyword argument `logits_to_keep` in forward()
        to save memory. Checking it in this way allows to avoid using a new model attribute.
        """
        return "logits_to_keep" in set(inspect.signature(self.forward).parameters.keys())

    def _prepare_special_tokens(
        self,
        generation_config,
        kwargs_has_attention_mask: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._decoder_start_token_tensor = decoder_start_token_tensor

    def generate(self, inputs = None, generation_config = None, use_model_defaults = None, **kwargs):

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config = self.generation_config
        kwargs_has_attention_mask = kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs. We pass input_ids to generate instead of inputs = xyz, so need this.
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, kwargs
        )
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = inputs_tensor.shape[1]
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # keep logits_to_keep as 1 to only keep the last logit indx and not all of them.
        # set it to 1 to avoid computing the whole logit matrix. This can save a lot of memory during the first forward pass.
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache
        # 11. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            inputs_tensor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            **model_kwargs,
        )
        return result

    def _sample(
        self,
        input_ids,
        stopping_criteria,
        generation_config,
        **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        is_prefill = True

        compiled_forward_decode = self.get_compiled_call(self.forward, dynamic = False)
        compiled_forward_prefill = self.get_compiled_call(self.forward, dynamic = True)
        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill: # the shape of prefill stage is unknown as the prompt can be of any length, so we can't use compile well.
                outputs = compiled_forward_prefill(**model_inputs, return_dict=True)
                is_prefill = False
            else: # decode only happens one token at a time, so we can use compile well.
                outputs = compiled_forward_decode(**model_inputs, return_dict=True)

            # don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float16, device=input_ids.device)

            # token selection
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, next_token_logits)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        return input_ids
