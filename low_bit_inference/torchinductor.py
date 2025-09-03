import os
import torch
from omegaconf import OmegaConf
# utils
from .hf_loader import load_model_tokenizer_prompt_cache
from .utils.config_utils import get_config
from .utils.profile_utils import profile_model
from .utils.compile_utils import compile_model


config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
print(f"Model loaded {config.model_id}.")

# compile the model here if you want
# basic cuda and cudnn configs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"

# Note for mode to option resolution:
# In [9]: torch._inductor.list_mode_options()
# Out[9]:
# {
#     'default': {},
#     'reduce-overhead': {
#         'triton.cudagraphs': True
#         },
#     'max-autotune-no-cudagraphs': {
#         'max_autotune': True,
#         'coordinate_descent_tuning': True
#         },
#     'max-autotune': {
#         'max_autotune': True,
#         'triton.cudagraphs': True,
#         'coordinate_descent_tuning': True
#         }
# }

model = model.to(config.device)
model, tokenizer = compile_model(model, tokenizer)
print("Model moved to GPU, starting profiling.")

profile_model(model, tokenizer, past_key_values, prompt, config)