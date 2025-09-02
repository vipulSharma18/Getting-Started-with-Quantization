import os
import torch
import torchao
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
model = torchao.autoquant(model)

print("Model moved to GPU, starting profiling.")

profile_model(model, tokenizer, past_key_values, prompt, config)