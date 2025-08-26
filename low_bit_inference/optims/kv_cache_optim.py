"""
Note:
-----

1. Use static cache to ensure torch.compile can work on it.
2. We can also quantize the KV Cache, or use techniques like latent attentition to downproject and upproject
the KV cache and save memory.
"""

from transformers import StaticCache
from ..utils.config_utils import to_torch_dtype


def setup_cache(cache_size, model_config, profile_config):
    past_key_values = StaticCache(
        config=model_config,
        max_batch_size=1,
        max_cache_len=cache_size,
        device=profile_config.device,
        dtype=to_torch_dtype(profile_config.compute_dtype)
    )
    return past_key_values