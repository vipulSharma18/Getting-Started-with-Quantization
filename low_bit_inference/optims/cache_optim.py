"""
Note:
-----

1. Use static cache to ensure torch.compile can work on it (shape doesn't change so it can optimize it).
2. We can also quantize the KV Cache, or use techniques like latent attentition to downproject and upproject
the KV cache and save memory.

This file implements StaticCache and other variants like the quantized cache, we'll need to take code from here and optimize it.
https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys, self.values = None, None

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor): ...

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        self.keys.zero_()
        self.values.zero_()

    def destroy(self):
        if self.keys is not None:
            del self.keys
            self.keys = None
        if self.values is not None:
            del self.values
            self.values = None

class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len, head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, dummy_key_states: torch.Tensor, dummy_value_states: torch.Tensor):
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        key_batch, key_heads, _, key_dim = dummy_key_states.shape
        key_dtype, key_device = dummy_key_states.dtype, dummy_key_states.device

        value_batch, value_heads, _, value_dim = dummy_value_states.shape
        value_dtype, value_device = dummy_value_states.dtype, dummy_value_states.device

        assert (key_batch==value_batch) and (key_heads==value_heads) and (key_dim==value_dim) \
            and (key_dtype==value_dtype) and (key_device==value_device)

        self.max_batch_size, self.num_heads, self.head_dim = value_batch, value_heads, value_dim
        self.dtype, self.device = value_dtype, value_device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None or self.values is None:
            self.lazy_initialization(key_states, value_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        self.keys[:, :, cache_position] = key_states
        self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (self.keys[0, 0].any(dim=-1)).sum() if self.keys is not None else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len

class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`.
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
    ):
        self.layers = layers if layers is not None else []
        self.destroy_counter = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        return keys, values

    def early_initialization(
        self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
        
        # note: len passed here is 0, but during init it will use max cache len attribute of class and set it to constant size
        fake_keys_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
        # Init all layers
        for layer in self.layers:
            if layer.keys is None or layer.values is None:
                layer.lazy_initialization(fake_keys_tensor, fake_keys_tensor)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def destroy_for_cudagraph_setup(self):
        self.destroy_counter +=1 
        assert self.destroy_counter <=1, "Destroying repeatedly, doing something wrong, cudagraphs need fixed input address."
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].destroy()

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        return len(self.layers)

class StaticCache(Cache):
    is_compileable = True

    def __init__(
        self,
        config,
        max_cache_len: int,
        **kwargs,
    ):
        config = config.get_text_config()
        layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        layers = []
        for _ in range(len(layer_types)):
            layer = StaticLayer(max_cache_len=max_cache_len)
            layers.append(layer)

        super().__init__(layers=layers)