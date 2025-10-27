"""
Activations unquantized in bf16, and weights quantized in int4 as a demo for gemlite.
TorchAO has the same configs as part of the weights only quantization runs,
so we can try to match/compare the TorchAO and GemLite performance and be aware of that confound.
"""

import torch

import gemlite
from .utils.gemlite_utils import get_default_cache_config
gemlite.reset_config()
gemlite.load_config(get_default_cache_config())

import gc
from omegaconf import OmegaConf
# utils
from .hf_loader import load_model_tokenizer_prompt_cache
from .utils.config_utils import get_config, to_torch_dtype
from .utils.profile_utils import profile_model
from .utils.gemlite_utils import patch_model, monkeypatch_gemlite
monkeypatch_gemlite()
from gemlite.helper import A16W4_HQQ_INT


config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
print(f"Model loaded {config.model_id}.")

# torch backends and compiler configs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch._inductor.config.benchmark_kernel = True
torch._inductor.config.benchmark_fusion = True
torch._inductor.config.freezing = True

model = model.to(config.device)
print("Model moved to GPU, starting profiling.")

assert config.compile_decode and config.quantize
print(f"Compile config: decode {config.compile_decode}, \
    prefill {config.compile_prefill}. Quantize status: {config.quantize}")

def cache_init(past_key_values, model, config, kv_compiled=False):
    if not kv_compiled:
        # just doing this so that the key and vals are output of cudagraph and hence mutating them in update doesn't cause cudagraph skipping
        past_key_values.early_initialization = torch.compile(
            past_key_values.early_initialization,
            mode="reduce-overhead",
        )

    past_key_values.early_initialization(
        batch_size=1,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=to_torch_dtype(config.compute_dtype),
        device=config.device,
    )
    return past_key_values

def model_quantize(causal_model, quantized=False):
    if not quantized:
        processor = A16W4_HQQ_INT(device="cuda", dtype=torch.bfloat16)
        patch_model(causal_model.model, device="cuda", processor=processor, group_size=64)
        torch.cuda.empty_cache()
        gc.collect()
    else:
        pass

model.quantization_function = model_quantize

profile_model(model, tokenizer, prompt, config, past_key_values, cache_init)