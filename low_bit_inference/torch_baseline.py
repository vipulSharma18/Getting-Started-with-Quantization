import torch
from omegaconf import OmegaConf
# utils
from low_bit_inference.hf_loader import load_model_tokenizer_prompt_cache
from low_bit_inference.utils.config_utils import get_config, to_torch_dtype
from low_bit_inference.utils.profile_utils import profile_model

config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
print(f"Model loaded {config.model_id}.")

## compilation and backends configs
torch.backends.fp32_precision = "tf32"

model = model.to(config.device)
print(f"Model moved to {config.device}, starting profiling.")

assert (not model.compile_decode) and (not model.compile_prefill) and (not model.quantize)
print(f"Compile config: decode {config.compile_decode}, \
    prefill {config.compile_prefill}. Quantize status: {config.quantize}")

def cache_init(past_key_values, model, config, *args, **kwargs):
    past_key_values.early_initialization(
        batch_size=1,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=to_torch_dtype(config.compute_dtype),
        device=config.device,
    )
    return past_key_values

profile_model(model, tokenizer, prompt, config, past_key_values, cache_init)