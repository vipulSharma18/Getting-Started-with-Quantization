import math
import torch
from omegaconf import OmegaConf
# utils
from .utils.hf_utils import load_model_tokenizer
from .utils.config_utils import get_config
from .utils.profile_utils import profile_model
# optims
from .optims.kv_cache_optim import setup_cache

config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer = load_model_tokenizer(config)
print(f"Model loaded {config.model_id}.")

prompt = config.prompt if isinstance(config.prompt, list) else [config.prompt]
print(f"Using prompt: {prompt}")

if config.use_cache:
    print("Setting up cache")
    model.config.use_cache = config.use_cache
    model.generation_config.cache_implementation = None  # remove the gen config var otherwise value error for using both ways
    cache_size = 2**math.ceil(math.log(len(prompt) + config.max_new_tokens, 2))
    past_key_values = setup_cache(cache_size, model.config, config)

## compile the model here if you want
# model.forward = torch.compile(model.forward)
torch.set_float32_matmul_precision('high')
model = model.to(config.device)
print(f"Model moved to {config.device}, starting profiling.")

profile_model(model, tokenizer, past_key_values, prompt, config)