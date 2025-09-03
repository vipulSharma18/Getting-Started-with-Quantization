# PreTrainedModel
"""
https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
"""

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import functools
import importlib.metadata
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from functools import wraps
from typing import Callable, Optional, Union, get_type_hints

import torch
from packaging import version
from torch import nn

from transformers import PretrainedConfig, GenerationConfig
from ..optims.attention_optim import sdpa_attention_forward

ALL_ATTENTION_FUNCTIONS = {"sdpa": sdpa_attention_forward}


class EmbeddingAccessMixin:
    """
    Base utilities to regroup getters and setters for embeddings.
    Introduces the `input_layer_embed` attribute, which indicates
    where the input embeddings come from and where they
    should be set.
    """

    _input_embed_layer = "embed_tokens"  # default layer that holds input embeddings.

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """

        # 1) Check if the model has an attribute named 'embed_tokens' (the standard input embedding layer
        #  for most NLP models), and if so, return it.

        name = getattr(self, "_input_embed_layer", "embed_tokens")

        if (default_embedding := getattr(self, name, None)) is not None:
            return default_embedding
        # 2) encoder/decoder and VLMs like `Gemma3nForConditionalGeneration`

        if hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens

        # 3) vanilla decoder‑only architectures
        elif hasattr(self, "embed_tokens"):
            return self.embed_tokens
        else:
            base_model = getattr(self, "base_model_prefix", None)
            if base_model is not None:
                base_model = getattr(self, base_model, None)
                if base_model is not None and base_model is not self:
                    return base_model.get_input_embeddings()
            raise NotImplementedError(
                f"`get_input_embeddings` not auto‑handled for {self.__class__.__name__}; "
                "please override in the subclass."
            )

    def get_output_embeddings(self):
        if not hasattr(self, "lm_head"):
            return None
        try:
            # Speech / vision backbones raise here, so we return None.
            # Legit use of get_input_embs?
            self.get_input_embeddings()
        except NotImplementedError:
            return None
        return self.lm_head


class PreTrainedModel(nn.Module, EmbeddingAccessMixin):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None
    _checkpoint_conversion_mapping = {}  # used for BC support in VLMs, not meant to be used by new models
    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None
    # the _keep_in_fp32_modules will avoid casting to anything other than float32, except bfloat16
    # to also prevent bfloat16 casting, use the _keep_in_fp32_modules_strict flag
    _keep_in_fp32_modules_strict = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None
    is_parallelizable = False
    supports_gradient_checkpointing = False
    _is_stateful = False
    # Flash Attention support
    _supports_flash_attn = False
    # SDPA support
    _supports_sdpa = True
    # Flex Attention support
    _supports_flex_attn = False
    _can_compile_fullgraph = True
    _tp_plan = None
    _tp_size = None
    _pp_plan = None
    _supports_attention_backend = True
    _can_record_outputs = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # For BC we keep the original `config_class` definition in case
        # there is a `config_class` attribute (e.g. remote code models),
        # otherwise we derive it from the annotated `config` attribute.

        # defined in this particular subclass
        child_annotation = cls.__dict__.get("__annotations__", {}).get("config", None)
        child_attribute = cls.__dict__.get("config_class", None)

        # defined in the class (this subclass or any parent class)
        full_annotation = get_type_hints(cls).get("config", None)
        full_attribute = cls.config_class

        # priority (child class_config -> child annotation -> global class_config -> global annotation)
        if child_attribute is not None:
            cls.config_class = child_attribute
        elif child_annotation is not None:
            cls.config_class = child_annotation
        elif full_attribute is not None:
            cls.config_class = full_attribute
        elif full_annotation is not None:
            cls.config_class = full_annotation

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

        # Check the attention implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._attn_implementation_internal = self._check_and_adjust_attn_implementation(
            self.config._attn_implementation, is_init_check=True
        )

        self.loss_type = None

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)
        self._keep_in_fp32_modules_strict = copy.copy(self.__class__._keep_in_fp32_modules_strict)

        self._no_split_modules = self._no_split_modules or []

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).

        This is also used when the user is running distributed code. We add hooks to the modules here, according to
        the model's tp_plan!
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

        # Make sure the modules correctly exist if the flag is active
        if self._keep_in_fp32_modules is not None or self._keep_in_fp32_modules_strict is not None:
            all_parameters = {name for name, _ in self.named_parameters() if len(name) > 0}
            unique_module_names = set()
            # Get all unique module names in the module graph, without the prefixes
            for param in all_parameters:
                unique_module_names.update(
                    [name for name in param.split(".") if not name.isnumeric() and name not in ["weight", "bias"]]
                )
            # Check that every module in the keep_in_fp32 list is part of the module graph
            if self._keep_in_fp32_modules is not None:
                for module in self._keep_in_fp32_modules:
                    if module not in unique_module_names:
                        raise ValueError(
                            f"{module} was specified in the `_keep_in_fp32_modules` list, but is not part of the modules in"
                            f" {self.__class__.__name__}"
                        )

            if self._keep_in_fp32_modules_strict is not None:
                for module in self._keep_in_fp32_modules_strict:
                    if module not in unique_module_names:
                        raise ValueError(
                            f"{module} was specified in the `_keep_in_fp32_modules_strict` list, but is not part of the modules in"
                            f" {self.__class__.__name__}"
                        )

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], is_init_check: bool = False
    ) -> str:
        # we'll always use sdpa so no point of checking kernels and what not
        return attn_implementation

    def get_correct_attn_implementation(self, requested_attention: Optional[str], is_init_check: bool = False) -> str:
        applicable_attention = "sdpa" if requested_attention is None else requested_attention
        return applicable_attention

    @classmethod
    def _can_set_attn_implementation(cls) -> bool:
        """Detect whether the class supports setting its attention implementation dynamically. It is an ugly check based on
        opening the file, but avoids maintaining yet another property flag.
        """
        class_file = sys.modules[cls.__module__].__file__
        with open(class_file, "r") as f:
            code = f.read()
        # heuristic -> if we find those patterns, the model uses the correct interface
        if re.search(r"class \w+Attention\(nn.Module\)", code):
            return (
                "eager_attention_forward" in code
                and "ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]" in code
            )
        else:
            # If no attention layer, assume `True`. Most probably a multimodal model or inherits from existing models
            return True

    def set_attn_implementation(self, attn_implementation: Union[str, dict]):
        """
        Set the requested `attn_implementation` for this model.

        Args:
            attn_implementation (`str` or `dict`):
                The attention implementation to set for this model. It can be either a `str`, in which case it will be
                dispatched to all submodels if relevant, or a `dict` where keys are the sub_configs name, in which case each
                submodel will dispatch the corresponding value.
        """
        requested_implementation = (
            attn_implementation
            if not isinstance(attn_implementation, dict)
            else attn_implementation.get("", self.config._attn_implementation)
        )

        # At this point, the model was already instantiated, so instead of crashing on bad value, let's simply
        # warn the user that the requested value is not working
        if requested_implementation != self.config._attn_implementation:
            # In this case, raise
            if not self._can_set_attn_implementation():
                print(
                    f"{self.__class__.__name__} does not support setting its attention implementation dynamically, because it "
                    "does not follow the functional approach based on AttentionInterface "
                    "(see https://huggingface.co/docs/transformers/en/attention_interface)"
                )
            else:
                requested_implementation = self._check_and_adjust_attn_implementation(
                    requested_implementation, is_init_check=False
                )
                # Apply the change (on the internal attr, to avoid setting it recursively)
                self.config._attn_implementation_internal = requested_implementation

        # Apply it to all submodels as well
        for submodule in self.modules():
            # We found a submodel (which is not self) with a different config (otherwise, it may be the same "actual model",
            # e.g. ForCausalLM has a Model inside, but no need to check it again)
            if (
                submodule is not self
                and isinstance(submodule, PreTrainedModel)
                and submodule.config.__class__ != self.config.__class__
                # If it was already changed, no need to do it again
                and not hasattr(submodule.config, "_attn_was_changed")
            ):
                # In this case, warn and skip
                if not submodule._can_set_attn_implementation():
                    print(
                        f"{submodule.__class__.__name__} does not support setting its attention implementation dynamically, because it "
                        "does not follow the functional approach based on AttentionInterface "
                        "(see https://huggingface.co/docs/transformers/en/attention_interface)"
                    )
                # Set the attn on the submodule
                else:
                    sub_implementation = requested_implementation
                    if isinstance(attn_implementation, dict):
                        for subconfig_key in self.config.sub_configs:
                            # We need to check for exact object match here, with `is`
                            if getattr(self.config, subconfig_key) is submodule.config:
                                sub_implementation = attn_implementation.get(
                                    subconfig_key, submodule.config._attn_implementation
                                )
                                break
                    # Check the module can use correctly, otherwise we raise an error if requested attention can't be set for submodule
                    sub_implementation = submodule.get_correct_attn_implementation(sub_implementation)
                    submodule.config._attn_implementation_internal = sub_implementation

                # Still add it as "changed" even if it was skipped, as we would otherwise try to set it in the dark afterwards
                # We need to set it on the config itself, to differentiate 2 subconfigs of the same __class__ potentially
                submodule.config._attn_was_changed = True

        # We need this as some old and badly designed models use subconfigs without declaring the corresponding modules as PreTrainedModel
        for subconfig_key in self.config.sub_configs:
            subconfig = getattr(self.config, subconfig_key)
            sub_implementation = (
                requested_implementation
                if not isinstance(attn_implementation, dict)
                else attn_implementation.get(subconfig_key, subconfig._attn_implementation)
            )
            # This means we did not perform any check above for this particular subconfig -> set it in the dark if it is registered
            if (
                not hasattr(subconfig, "_attn_was_changed")
                # If it's already the same, then no need to enter here and raise warnings
                and sub_implementation != subconfig._attn_implementation
            ):
                if sub_implementation not in ["eager"] + list(ALL_ATTENTION_FUNCTIONS.keys()):
                    raise ValueError(
                        f'Specified `attn_implementation="{sub_implementation}"` is not supported for {subconfig_key}. '
                        'The only possible arguments are "eager" (manual attention implementation)'
                        f"or one of the following: {list(ALL_ATTENTION_FUNCTIONS.keys())}"
                    )
                subconfig._attn_implementation_internal = sub_implementation
                print(
                    f"We set the attention implementation for the sub-config `{subconfig_key}` to `{sub_implementation}` "
                    "without finding the associated sub-model. For this reason we could not check if the model supports it. "
                    "You may encounter undefined behavior."
                )
            # Unset the attribute in this case, to avoid issues in the future
            else:
                if hasattr(subconfig, "_attn_was_changed"):
                    del subconfig._attn_was_changed

    def get_decoder(self):
        """
        Best-effort lookup of the *decoder* module.

        Order of attempts (covers ~85 % of current usages):

        1. `self.decoder`
        2. `self.model`                       (many wrappers store the decoder here)
        3. `self.model.get_decoder()`         (nested wrappers)
        4. fallback: raise for the few exotic models that need a bespoke rule
        """
        if hasattr(self, "decoder"):
            return self.decoder

        if hasattr(self, "model"):
            inner = self.model
            if hasattr(inner, "get_decoder"):
                return inner.get_decoder()
            return inner

        return None  # raise AttributeError(f"{self.__class__.__name__} has no decoder; override `get_decoder()` if needed.")

    def set_decoder(self, decoder):
        """
        Symmetric setter. Mirrors the lookup logic used in `get_decoder`.
        """

        if hasattr(self, "decoder"):
            self.decoder = decoder
            return

        if hasattr(self, "model"):
            inner = self.model
            if hasattr(inner, "set_decoder"):
                inner.set_decoder(decoder)
            else:
                self.model = decoder
            return

        return  # raise AttributeError(f"{self.__class__.__name__} cannot accept a decoder; override `set_decoder()`.")

    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute,
            # it's necessary to manually call the `cuda` method on HQQLinear layers.
            super().cuda(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if len(args) > 0:
                        device = args[0]
                    else:
                        device = kwargs.get("device", "cuda")
                    module.cuda(device)
            return self

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "Calling `cuda()` is not supported for `8-bit` quantized models. "
                    " Please use the model as it is, since the model has already been set to the correct devices."
                )
            elif version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.2"):
                raise ValueError(
                    "Calling `cuda()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        return super().cuda(*args, **kwargs)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        # For BNB/GPTQ models, we prevent users from casting the model to another dtype to restrict unwanted behaviours.
        # the correct API should be to load the model with the desired dtype directly through `from_pretrained`.
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute, we must
            # explicitly move the parameters to the target device for each HQQLinear layer after `to`.
            super().to(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if "device" in kwargs:
                        device = kwargs["device"]
                    else:
                        device = args[0]
                    if "dtype" in kwargs:
                        dtype = kwargs["dtype"]
                    elif dtype_present_in_args:
                        dtype = arg
                    else:
                        dtype = None
                    # Due to the current messy implementation of HQQLinear, updating `compute_dtype`
                    # followed by calling the `cuda` method achieves the intended behavior of `to`,
                    # even when the target device is CPU.
                    if dtype is not None:
                        module.compute_dtype = dtype
                    module.cuda(device)
            return self

        if dtype_present_in_args and getattr(self, "quantization_method", None) == QuantizationMethod.QUARK:
            raise ValueError("Casting a Quark quantized model to a new `dtype` is not supported.")

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a bitsandbytes model in a new `dtype`. Make sure to load the model using `from_pretrained` using the"
                    " desired `dtype` by passing the correct `dtype` argument."
                )

            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "`.to` is not supported for `8-bit` bitsandbytes models. Please use the model as it is, since the"
                    " model has already been set to the correct devices and casted to the correct `dtype`."
                )
            elif version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.2"):
                raise ValueError(
                    "Calling `to()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        elif getattr(self, "quantization_method", None) == QuantizationMethod.GPTQ:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired"
                    " `dtype` by passing the correct `dtype` argument."
                )
        return super().to(*args, **kwargs)

    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def get_init_context(cls, is_quantized: bool, _is_ds_init_called: bool):
        init_contexts = [no_init_weights(), init_empty_weights()]
        return init_contexts

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)), or `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

                Accept HF kernel references in the form:
                  <namespace>/<repo_name>[@<revision>][:<kernel_name>]

                - <namespace> and <repo_name> are any non-"/" and non-":" sequences.
                - "@<revision>" is optional (branch, tag, or commit-ish), e.g. "@main", "@v1.2.0", "@abc123".
                - ":<kernel_name>" is optional and selects a function inside the kernel repo.
                - Both options can appear together and in this order only: @revision first, then :kernel_name.
                - We intentionally allow a leading "<wrapper>|" prefix (e.g., "flash|...") because the code
                  strips it before loading; '|' is not excluded in the character classes here.

                Examples that match:
                  "org/model"
                  "org/model@main"
                  "org/model:custom_kernel"
                  "org/model@v1.2.3:custom_kernel"

            > Parameters for big model inference

            dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch_dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `dtype` or `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `dtype` or `torch_dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            tp_plan (`str`, *optional*):
                A torch tensor parallel plan, see [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html). Currently, it only accepts
                `tp_plan="auto"` to use predefined plan based on the model. Note that if you use it, you should launch your script accordingly with
                `torchrun [args] script.py`. This will be much faster than using a `device_map`, but has limitations.
            tp_size (`str`, *optional*):
                A torch tensor parallel degree. If not provided would default to world size.
            device_mesh (`torch.distributed.DeviceMesh`, *optional*):
                A torch device mesh. If not provided would default to world size. Used only for tensor parallel for now.
                If provided, it has to contain dimension named `"tp"` in case it's > 1 dimensional, this dimension will be used for tensor parallelism
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_state_dict (`bool`, *optional*):
                If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
                RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
                `True` when there is some disk offload.
            offload_buffers (`bool`, *optional*):
                Whether or not to offload the buffers with the model parameters.
            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
                bitsandbytes, gptq). There may be other quantization-related kwargs, including `load_in_4bit` and
                `load_in_8bit`, which are parsed by QuantizationConfigParser. Supported only for bitsandbytes
                quantizations and not preferred. consider inserting all such arguments into quantization_config
                instead.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_tf` or `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.
            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.
            key_mapping (`dict[str, str], *optional*):
                A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
                architecture, but was not converted accordingly.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
        ```
        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        offload_buffers = kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")
        generation_config = kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)
        tp_plan = kwargs.pop("tp_plan", None)
        tp_size = kwargs.pop("tp_size", None)
        distributed_config: DistributedConfig = kwargs.pop("distributed_config", None)
        device_mesh = kwargs.pop("device_mesh", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        use_kernels = kwargs.pop("use_kernels", False)

        key_mapping = kwargs.pop("key_mapping", None)
        # Load models with hardcoded key mapping on class for VLMs only, to keep BC and standardize model
        if key_mapping is None and any(
            allowed_name in class_name.__name__.lower() for class_name in cls.__mro__[:-1] for allowed_name in VLMS
        ):
            key_mapping = cls._checkpoint_conversion_mapping

        if distributed_config is not None:
            tp_plan = "auto"

        # Not used anymore -- remove them from the kwargs
        _ = kwargs.pop("resume_download", None)
        _ = kwargs.pop("mirror", None)
        _ = kwargs.pop("_fast_init", True)
        _ = kwargs.pop("low_cpu_mem_usage", None)

        # For BC on torch_dtype argument
        if torch_dtype is not None:
            print("`torch_dtype` is deprecated! Use `dtype` instead!")
            # If both kwargs are provided, use `dtype`
            dtype = dtype if dtype is not None else torch_dtype

        if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
            raise ValueError(
                "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
            )
        if tp_size is not None and tp_plan is None:
            raise ValueError("tp_plan has to be set when tp_size is passed.")
        if tp_plan is not None and tp_plan != "auto":
            # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
            raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")
        if tp_plan is not None and device_map is not None:
            raise ValueError(
                "`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization."
            )

        if device_map == "auto" and int(os.environ.get("WORLD_SIZE", "0")):
            logger.info(
                "You've set device_map=`auto` while triggering a distributed run with torchrun. This might lead to unexpected behavior. "
                "If your plan is to load the model on each device, you should set device_map={"
                ": PartialState().process_index} where PartialState comes from accelerate library"
            )

        # We need to correctly dispatch the model on the current process device. The easiest way for this is to use a simple
        # `device_map` pointing to the correct device
        if tp_plan is not None:
            if device_mesh is None:
                tp_plan, device_map, device_mesh, tp_size = initialize_tensor_parallelism(tp_plan, tp_size=tp_size)
            else:
                if device_mesh.ndim > 1:
                    if "tp" not in device_mesh.mesh_dim_names:
                        raise ValueError(
                            "When using `tp_plan` and n-d `device_mesh`, it must contain a 'tp' dimension. "
                            "Please provide a valid `device_mesh`."
                        )
                    device_mesh = device_mesh["tp"]
                tp_size = device_mesh.size()
                device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")

            if tp_size is None:
                tp_size = torch.distributed.get_world_size()

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

            if _adapter_model_path is None:
                _adapter_model_path = find_adapter_config_file(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                    **adapter_kwargs,
                )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
        else:
            _adapter_model_path = None

        # Potentially detect context manager or global device, and use it (only if no device_map was provided)
        if device_map is None and not is_deepspeed_zero3_enabled():
            device_in_context = get_torch_context_manager_or_global_device()
            if device_in_context == torch.device("meta"):
                # TODO Cyril: raise an error instead of the warning in v4.53 (and change the test to check for raise instead of success)
                print(
                    "We detected that you are using `from_pretrained` with a meta device context manager or `torch.set_default_device('meta')`\n"
                    "This is an anti-pattern and will raise an Error in version v4.53\nIf you want to initialize a model on the meta device, use "
                    "the context manager or global device with `from_config`, or `ModelClass(config)`"
                )
            device_map = device_in_context

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed Zero-3 is not compatible with passing a `device_map`.")
            if not is_accelerate_available():
                raise ValueError(
                    "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` "
                    "requires `accelerate`. You can install it with `pip install accelerate`"
                )

        # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
        if load_in_4bit or load_in_8bit:
            if quantization_config is not None:
                raise ValueError(
                    "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
                    "`quantization_config` argument at the same time."
                )

            # preparing BitsAndBytesConfig from kwargs
            config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
            config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
            quantization_config, kwargs = BitsAndBytesConfig.from_dict(
                config_dict=config_dict, return_unused_kwargs=True, **kwargs
            )
            print(
                "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
                "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
            )

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
            if "gguf_file" in model_kwargs:
                model_kwargs.pop("gguf_file")
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs

        # Because some composite configs call super().__init__ before instantiating the sub-configs, we need this call
        # to correctly redispatch recursively if the kwarg is provided
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        transformers_explicit_filename = getattr(config, "transformers_weights", None)

        if transformers_explicit_filename is not None:
            if not transformers_explicit_filename.endswith(
                ".safetensors"
            ) and not transformers_explicit_filename.endswith(".safetensors.index.json"):
                raise ValueError(
                    "The transformers file in the config seems to be incorrect: it is neither a safetensors file "
                    "(*.safetensors) nor a safetensors index file (*.safetensors.index.json): "
                    f"{transformers_explicit_filename}"
                )

        hf_quantizer, config, dtype, device_map = get_hf_quantizer(
            config, quantization_config, dtype, from_tf, from_flax, device_map, weights_only, user_agent
        )

        if gguf_file is not None and hf_quantizer is not None:
            raise ValueError(
                "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
            )

        if (
            gguf_file
            and device_map is not None
            and ((isinstance(device_map, dict) and "disk" in device_map.values()) or "disk" in device_map)
        ):
            raise RuntimeError(
                "One or more modules is configured to be mapped to disk. Disk offload is not supported for models "
                "loaded from GGUF files."
            )

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            variant=variant,
            gguf_file=gguf_file,
            from_tf=from_tf,
            from_flax=from_flax,
            use_safetensors=use_safetensors,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            commit_hash=commit_hash,
            is_remote_code=cls._auto_class is not None,
            transformers_explicit_filename=transformers_explicit_filename,
        )

        is_sharded = sharded_metadata is not None
        is_quantized = hf_quantizer is not None
        is_from_file = pretrained_model_name_or_path is not None or gguf_file is not None

        if (
            is_safetensors_available()
            and is_from_file
            and not is_sharded
            and checkpoint_files[0].endswith(".safetensors")
        ):
            with safe_open(checkpoint_files[0], framework="pt") as f:
                metadata = f.metadata()

            if metadata is None:
                # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
                pass
            elif metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)

        if from_pt:
            if gguf_file:
                from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

                # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
                # passed directly as a kwarg from now on
                with torch.device("meta"):
                    dummy_model = cls(config)
                state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
                    "tensors"
                ]

            # Find the correct dtype based on current state
            config, dtype, dtype_orig = _get_dtype(
                cls, dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
            )

        config.name_or_path = pretrained_model_name_or_path
        model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # Make sure to tie the weights correctly
        model.tie_weights()

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Find fp32 modules if needed
        keep_in_fp32_modules = []
        # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
        # in case of force loading a model that should stay bf16 in fp16 (which includes a few quantizers as this is a pre-processing
        # step for e.g. bitsandbytes). See https://github.com/huggingface/transformers/issues/20287 for details.
        if model._keep_in_fp32_modules is not None and (
            dtype == torch.float16 or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
        ):
            keep_in_fp32_modules.extend(model._keep_in_fp32_modules)

        if model._keep_in_fp32_modules_strict is not None and (dtype == torch.float16 or dtype == torch.bfloat16):
            keep_in_fp32_modules.extend(model._keep_in_fp32_modules_strict)

        keep_in_fp32_regex = None
        if keep_in_fp32_modules:
            # We need to match exact layers, so we add either `.` on each side, or start/end of string
            keep_in_fp32_regex = re.compile("|".join([rf"((^|\.){module}($|\.))" for module in keep_in_fp32_modules]))

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model,
                device_map=device_map,
                keep_in_fp32_modules=model._keep_in_fp32_modules,
                config=config,
                use_kernels=use_kernels,
            )
            # We store the original dtype for quantized models as we cannot easily retrieve it
            # once the weights have been quantized
            # Note that once you have loaded a quantized model, you can't change its dtype so this will
            # remain a single source of truth
            original_dtype = dtype if dtype is not None else torch.get_default_dtype()

            def _assign_original_dtype(module):
                for child in module.children():
                    if isinstance(child, PreTrainedModel):
                        child.config._pre_quantization_dtype = original_dtype
                    _assign_original_dtype(child)

            config._pre_quantization_dtype = original_dtype
            _assign_original_dtype(model)

        if _torch_distributed_available and device_mesh is not None:
            model = distribute_model(model, distributed_config, device_mesh, tp_size)

        # Prepare the full device map
        if device_map is not None:
            device_map = _get_device_map(model, device_map, max_memory, hf_quantizer, dtype, keep_in_fp32_regex)

        # Finalize model weight initialization
        if from_tf:
            model, loading_info = cls._load_from_tf(model, config, checkpoint_files)
        elif from_flax:
            model = cls._load_from_flax(model, checkpoint_files)
        elif from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                checkpoint_files,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                device_map=device_map,
                disk_offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_regex=keep_in_fp32_regex,
                device_mesh=device_mesh,
                key_mapping=key_mapping,
                weights_only=weights_only,
            )
        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # check if using kernels
        if use_kernels:
            if not is_kernels_available():
                raise ValueError(
                    "Kernels are not available. To use kernels, please install kernels using `pip install kernels`"
                )

            from kernels import Device, kernelize

            kernelize(model, device=Device(type=model.device.type))

        # If it is a model with generation capabilities, attempt to load generation files (generation config,
        # custom generate function)
        if model.can_generate() and generation_config is not None:
            logger.info("The user-defined `generation_config` will be used to override the default generation config.")
            model.generation_config = model.generation_config.from_dict(generation_config.to_dict())
        elif model.can_generate() and pretrained_model_name_or_path is not None:
            repo_loading_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                "subfolder": subfolder,
                **kwargs,
            }
            # Load generation config
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **repo_loading_kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass
            # Load custom generate function if `pretrained_model_name_or_path` defines it (and override `generate`)
            if hasattr(model, "load_custom_generate"):
                try:
                    custom_generate = model.load_custom_generate(
                        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **repo_loading_kwargs
                    )
                    model.generate = functools.partial(custom_generate, model=model)
                except OSError:  # there is no custom generate function
                    pass

        # Dispatch model with hooks on all devices if necessary (not needed with a tp_plan, so we skip it as it slightly
        # harm performances)
        if device_map is not None and device_mesh is None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
                "offload_buffers": offload_buffers,
            }
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
            # For HQQ method we force-set the hooks for single GPU envs
            if (
                "force_hooks" in inspect.signature(dispatch_model).parameters
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
            ):
                device_map_kwargs["force_hooks"] = True
            if (
                hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                device_map_kwargs["offload_buffers"] = True

            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            model.hf_quantizer = hf_quantizer
            hf_quantizer.postprocess_model(model, config=config)

        if _adapter_model_path is not None:
            adapter_kwargs["key_mapping"] = key_mapping
            model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                token=token,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            if from_pt:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            elif from_flax:
                loading_info = None
            return model, loading_info
        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model: "PreTrainedModel",
        state_dict: Optional[dict],
        checkpoint_files: Optional[list[str]],
        pretrained_model_name_or_path: Optional[str],
        ignore_mismatched_sizes: bool = False,
        sharded_metadata: Optional[dict] = None,
        device_map: Optional[dict] = None,
        disk_offload_folder: Optional[str] = None,
        offload_state_dict: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        hf_quantizer: Optional[HfQuantizer] = None,
        keep_in_fp32_regex: Optional[re.Pattern] = None,
        device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
        key_mapping: Optional[dict[str, str]] = None,
        weights_only: bool = True,
    ):
        # TODO: we should only be calling hf_quantizer.skip_placement or something like that
        is_quantized = hf_quantizer is not None
        is_hqq_or_quark = is_quantized and hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.QUARK,
        }
        is_hqq_or_bnb = is_quantized and hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.BITS_AND_BYTES,
        }

        # Get all the keys of the state dicts that we have to initialize the model
        if sharded_metadata is not None:
            original_checkpoint_keys = sharded_metadata["all_checkpoint_keys"]
        elif state_dict is not None:
            original_checkpoint_keys = list(state_dict.keys())
        else:
            original_checkpoint_keys = list(
                load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
            )

        # Check if we are in a special state, i.e. loading from a state dict coming from a different architecture
        prefix = model.base_model_prefix
        _prefix = f"{prefix}."
        has_prefix_module = any(s.startswith(prefix) for s in original_checkpoint_keys) if len(prefix) > 0 else False
        expects_prefix_module = hasattr(model, prefix) if len(prefix) > 0 else False
        loading_task_model_from_base_state_dict = not has_prefix_module and expects_prefix_module
        loading_base_model_from_task_state_dict = has_prefix_module and not expects_prefix_module

        # Find the key names that the model expects from the serialized keys
        key_renaming_mapping = model._get_key_renaming_mapping(
            original_checkpoint_keys,
            key_mapping,
            loading_base_model_from_task_state_dict,
            loading_task_model_from_base_state_dict,
        )
        checkpoint_keys = list(key_renaming_mapping.values())

        # Find missing and unexpected keys from the state dict
        missing_keys, unexpected_keys = _find_missing_and_unexpected_keys(
            cls,
            model,
            original_checkpoint_keys,
            checkpoint_keys,
            loading_base_model_from_task_state_dict,
            hf_quantizer,
            device_map,
        )
        # Find all the keys with shape mismatch (if we ignore the mismatch, the weights need to be newly initialized the
        # same way as missing keys)
        mismatched_keys, mismatched_shapes = _find_mismatched_keys(
            model,
            state_dict,
            checkpoint_files,
            ignore_mismatched_sizes,
            key_renaming_mapping,
            is_quantized,
            weights_only,
        )

        # We need to update both the mapping and the list of checkpoint keys to remove the mismatched ones
        key_renaming_mapping = {k: v for k, v in key_renaming_mapping.items() if v not in mismatched_keys}
        checkpoint_keys = list(key_renaming_mapping.values())

        # Move missing (and potentially mismatched) keys back to cpu from meta device (because they won't be moved when
        # loading the weights as they are not in the loaded state dict)
        model._move_missing_keys_from_meta_to_cpu(missing_keys + mismatched_keys, unexpected_keys, dtype, hf_quantizer)

        # correctly initialize the missing (and potentially mismatched) keys
        model._initialize_missing_keys(checkpoint_keys, ignore_mismatched_sizes, is_quantized)

        # Set some modules to fp32 if needed
        if keep_in_fp32_regex is not None:
            for name, param in model.named_parameters():
                if keep_in_fp32_regex.search(name):
                    # param = param.to(torch.float32) does not work here as only in the local scope.
                    param.data = param.data.to(torch.float32)

        # Make sure we are able to load base models as well as derived models (specific task models, with heads)
        model_to_load = model
        # In this case, we load a ForTaskModel with keys from a BaseModel -> only load keys to the BaseModel
        if loading_task_model_from_base_state_dict:
            model_to_load = getattr(model, prefix)
            # Here we need to remove the prefix we added to correctly find missing/unexpected keys, as we will load
            # in the submodule
            key_renaming_mapping = {k: v[len(_prefix) :] for k, v in key_renaming_mapping.items()}
            checkpoint_keys = list(key_renaming_mapping.values())
            # We need to update the device map as well
            if device_map is not None:
                device_map = {k[len(_prefix) :] if k.startswith(_prefix) else k: v for k, v in device_map.items()}
            # small sanity check: the base model should not contain task-specific head keys
            task_specific_expected_keys = [s for s in model.state_dict() if not s.startswith(_prefix)]
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(
                key in task_specific_expected_keys and key not in base_model_expected_keys for key in checkpoint_keys
            ):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )

        # Get reverse key mapping
        reverse_key_renaming_mapping = {v: k for k, v in key_renaming_mapping.items()}

        is_offloaded_safetensors = False
        # This offload index if for params explicitly on the "disk" in the device_map
        disk_offload_index = None
        disk_only_shard_files = []
        # Prepare parameters offloading if needed
        if device_map is not None and "disk" in device_map.values():
            if offload_state_dict is None:
                offload_state_dict = True
            if disk_offload_folder is not None:
                os.makedirs(disk_offload_folder, exist_ok=True)
            is_offloaded_safetensors = checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors")
            if disk_offload_folder is None and not is_offloaded_safetensors:
                raise ValueError(
                    "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                    " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                    " offers the weights in this format."
                )
            if is_offloaded_safetensors:
                param_device_map = expand_device_map(device_map, checkpoint_keys)
                str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
                if sharded_metadata is None:
                    weight_map = dict.fromkeys(checkpoint_keys, checkpoint_files[0])
                else:
                    folder = os.path.sep.join(checkpoint_files[0].split(os.path.sep)[:-1])
                    # Fix the weight map keys according to the key mapping
                    weight_map = {
                        key_renaming_mapping[k]: v
                        for k, v in sharded_metadata["weight_map"].items()
                        if k in key_renaming_mapping
                    }
                    weight_map = {k: os.path.join(folder, v) for k, v in weight_map.items()}
                    # Find potential checkpoints containing only offloaded weights
                    disk_only_shard_files = get_disk_only_shard_files(device_map, weight_map)
                disk_offload_index = {
                    name: {
                        "safetensors_file": file,
                        "weight_name": reverse_key_renaming_mapping[name],
                        "dtype": str_dtype,
                    }
                    for name, file in weight_map.items()
                    if param_device_map[name] == "disk"
                }
            else:
                disk_offload_index = {}

        # This offload index if for params that are supposed to be on the "cpu", either with or without a device_map
        # It allows to load parameters one-by-one from the state dict, avoiding a memory peak of 2 x state_dict_size,
        # i.e. 1x to load it, and 1x to copy it to model
        cpu_offload_folder = None
        cpu_offload_index = None
        if offload_state_dict:
            cpu_offload_folder = tempfile.mkdtemp()
            cpu_offload_index = {}

        # To be able to iterate, even if we don't use it if the state_dict is already provided
        elif state_dict is not None:
            checkpoint_files = [""]

        # Compute expected model keys
        expected_keys = list(model_to_load.state_dict().keys())
        if hf_quantizer is not None:
            expected_keys = hf_quantizer.update_expected_keys(model_to_load, expected_keys, checkpoint_keys)

        if logger.level >= logging.WARNING:
            verify_tp_plan(expected_keys, getattr(model_to_load, "_tp_plan", None))

        # Warmup cuda to load the weights much faster on devices
        if device_map is not None and not is_hqq_or_quark:
            expanded_device_map = expand_device_map(device_map, expected_keys)
            caching_allocator_warmup(model_to_load, expanded_device_map, hf_quantizer)

        # Prepare and compatabilize arguments for serial and parallel shard loading
        args_list = [
            (
                shard_file,
                state_dict,
                disk_only_shard_files,
                is_hqq_or_bnb,
                is_quantized,
                device_map,
                hf_quantizer,
                key_renaming_mapping,
                weights_only,
                model_to_load,
                expected_keys,
                reverse_key_renaming_mapping,
                disk_offload_folder,
                disk_offload_index,
                cpu_offload_folder,
                cpu_offload_index,
                is_offloaded_safetensors,
                keep_in_fp32_regex,
                unexpected_keys,
                device_mesh,
            )
            for shard_file in checkpoint_files
        ]

        error_msgs = []

        if (
            os.environ.get("HF_ENABLE_PARALLEL_LOADING", "").upper() in ENV_VARS_TRUE_VALUES
            and not is_deepspeed_zero3_enabled()
        ):
            _error_msgs, disk_offload_index, cpu_offload_index = load_shard_files_with_threadpool(args_list)
            error_msgs += _error_msgs
        else:
            if len(args_list) > 1:
                args_list = logging.tqdm(args_list, desc="Loading checkpoint shards")

            for args in args_list:
                _error_msgs, disk_offload_index, cpu_offload_index = load_shard_file(args)
                error_msgs += _error_msgs

        # Adjust offloaded weights name and save if needed
        if disk_offload_index is not None and len(disk_offload_index) > 0:
            if loading_task_model_from_base_state_dict:
                # We need to add the prefix of the base model
                prefix = cls.base_model_prefix
                if not is_offloaded_safetensors:
                    for weight_name in disk_offload_index:
                        shutil.move(
                            os.path.join(disk_offload_folder, f"{weight_name}.dat"),
                            os.path.join(disk_offload_folder, f"{prefix}.{weight_name}.dat"),
                        )
                disk_offload_index = {f"{prefix}.{key}": value for key, value in disk_offload_index.items()}
            if not is_offloaded_safetensors:
                save_offload_index(disk_offload_index, disk_offload_folder)
                disk_offload_index = None

        # one-at-a-time param loading for the cpu offloaded params
        if offload_state_dict:
            # Load back temporarily offloaded state dict
            load_offloaded_weights(model_to_load, cpu_offload_index, cpu_offload_folder)
            shutil.rmtree(cpu_offload_folder)

        if hf_quantizer is not None:
            missing_keys = hf_quantizer.update_missing_keys_after_loading(model_to_load, missing_keys, prefix)

        # Post-processing for tensor parallelism
        if device_mesh is not None:
            # When using TP, the device map is a single device for all parameters
            tp_device = list(device_map.values())[0]
            # This is needed for the RotaryEmbedding, which was not initialized on the correct device as it is
            # not part of the state_dict (persistent=False)
            for buffer in model.buffers():
                if buffer.device != tp_device:
                    buffer.data = buffer.to(tp_device)

            # In this case, the top-most task module weights were not moved to device and parallelized as they
            # were not part of the loaded weights: do it now
            if loading_task_model_from_base_state_dict:
                parameters_to_initialize = {
                    name: param for name, param in model.named_parameters() if not name.startswith(prefix)
                }
                for name, param in parameters_to_initialize.items():
                    # If it is still on meta here, it means that it's a tied weight that will be tied later anyway -> skip it
                    if param.device.type == "meta":
                        continue
                    # Shard the param
                    to_contiguous, casting_dtype = _infer_parameter_dtype(model, name, param, keep_in_fp32_regex)
                    shard_and_distribute_module(
                        model,
                        param.to(tp_device),
                        param,
                        name,
                        casting_dtype,
                        to_contiguous,
                        device_mesh.get_local_rank(),
                        device_mesh,
                    )

        # All potential warnings/infos
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = print if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            print(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, (shape1, shape2) in zip(mismatched_keys, mismatched_shapes)
                ]
            )
            print(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, disk_offload_index, error_msgs

    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        loss_type = getattr(self, "loss_type", None)

        if loss_type is None or loss_type not in LOSS_MAPPING:
            print(
                f"`loss_type={loss_type}` was set in the config but it is unrecognized. "
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    def get_compiled_call(self, compile_config: Optional[CompileConfig]) -> Callable:
        """Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
        non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
        want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
        (where we want the speed-ups of compiled version with static shapes)."""
        # Only reset it if not present or different from previous config
        if "llama4" in self.config.model_type:  # TODO try to enable for FULL COMPILE HYBRID CACHE SUPPORT
            return self.__call__
        compile_config = compile_config or CompileConfig()
        default_config = getattr(self.generation_config, "compile_config", None) or CompileConfig()
        if (
            not hasattr(self, "_compiled_call")
            or getattr(self, "_last_compile_config", default_config) != compile_config
        ):
            self._last_compile_config = compile_config
            self._compiled_call = torch.compile(self.__call__, **compile_config.to_dict())
        return self._compiled_call
