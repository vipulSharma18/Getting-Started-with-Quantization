import os
import math
import gc
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Float8WeightOnlyConfig, Int4WeightOnlyConfig
from datetime import datetime
from omegaconf import OmegaConf
# utils
from .utils.hf_utils import load_model_tokenizer
from .utils.config_utils import get_config
from .utils.profile_utils import profile_model
from .utils.compile_utils import compile_model
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

# compile the model here if you want
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
os.environ["TORCHINDUCTOR_BENCHMARK_FUSION"] = "1"
os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
os.environ["TORCHINDUCTOR_FREEZING"] = "1" 

model = model.to(config.device)
model, tokenizer = compile_model(model, tokenizer)
print("Model moved to GPU, starting profiling.")

model = quantize_(
    model,
    Float8WeightOnlyConfig()
)

profile_model(model, tokenizer, past_key_values, prompt, config)