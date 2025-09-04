import torch
from omegaconf import OmegaConf
# utils
from .hf_loader import load_model_tokenizer_prompt_cache
from .utils.config_utils import get_config
from .utils.profile_utils import profile_model

config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
print(f"Model loaded {config.model_id}.")

## compile the model here if you want
torch.set_float32_matmul_precision('high')

def get_compiled_call(model, dynamic = None):
    # compiled_call = torch.compile(model.__call__, fullgraph=True, dynamic=False, mode="max-autotune")
    return model.__call__

model.get_compiled_call = get_compiled_call
model = model.to(config.device)
print(f"Model moved to {config.device}, starting profiling.")

profile_model(model, tokenizer, past_key_values, prompt, config)