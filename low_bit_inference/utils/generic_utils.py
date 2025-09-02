"""
https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py
"""
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from collections import OrderedDict
from collections.abc import  MutableMapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import fields, is_dataclass
from functools import partial
from typing import Any, Callable, Optional, TypedDict
import torch


_CAN_RECORD_REGISTRY = {}


def _model_output_flatten(output):
    return list(output.values()), list(output.keys())

def _model_output_unflatten(values, context, output_type=None):
    return output_type(**dict(zip(context, values)))


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
    
        from torch.utils._pytree import register_pytree_node

        register_pytree_node(
            cls,
            _model_output_flatten,
            partial(_model_output_unflatten, output_type=cls),
            serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not isinstance(first_field, torch.Tensor):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: list[AbstractContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


class TransformersKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the loss function

    Attributes:
        num_items_in_batch (`Optional[torch.Tensor]`, *optional*):
            Number of items in the batch. It is recommended to pass it when
            you are doing gradient accumulation.
        output_hidden_states (`Optional[bool]`, *optional*):
            Most of the models support outputting all hidden states computed during the forward pass.
        output_attentions (`Optional[bool]`, *optional*):
            Turn this on to return the intermediary attention scores.
        output_router_logits (`Optional[bool]`, *optional*):
            For MoE models, this allows returning the router logits to compute the loss.
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    num_items_in_batch: Optional["torch.Tensor"]
    output_hidden_states: Optional[bool]
    output_attentions: Optional[bool]
    output_router_logits: Optional[bool]
    cu_seq_lens_q: Optional["torch.LongTensor"]
    cu_seq_lens_k: Optional["torch.LongTensor"]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


class GeneralInterface(MutableMapping):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications though the class mapping, as well as local modifications in a single file with the local mapping.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())