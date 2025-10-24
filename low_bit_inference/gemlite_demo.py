import torch
from omegaconf import OmegaConf
from .hf_loader import load_model_tokenizer_prompt_cache
from .utils.config_utils import get_config, to_torch_dtype
from .utils.profile_utils import profile_model
import gemlite
from gemlite import DType, GemLiteLinear
from gemlite.core import TORCH_TO_DTYPE
from gemlite.helper import (
    cleanup_linear,
    patch_model,
    A16W4_HQQ_INT
)
 

gemlite.reset_cache()
gemlite.set_autotune("max")  # fast for fast startup, but slow perf. max for slow startup, but best perf.

def create_gemlite_processor(W_nbits=4, quantize_activations=False):
    assert W_nbits in [8, 4, 2, 1.58, 1], "Invalid W_nbits."
    nbits_to_max_val = {8:127, 4:7, 2:1, 1.58:1, 1:1}
    max_val = nbits_to_max_val[W_nbits]
    zeros   = max_val

    gemlite_linear = GemLiteLinear(round(W_nbits), 
                    group_size=in_features, 
                    in_features=in_features, 
                    out_features=out_features, 
                    input_dtype=input_dtype, 
                    output_dtype=TORCH_TO_DTYPE[compute_dtype], 
                    scaled_activations=scale_activations,
                    )


    gemlite_linear.pack(W_q, scales=scales, zeros=zeros, bias=bias, fma_mode=False)

    if(quantize_activations):
        gemlite_linear.W_group_mode       = 1 #shift only (Wq - zeros)
        gemlite_linear.channel_scale_mode = 3 #activations + weight
    else:
        gemlite_linear.W_group_mode       = 3 #(Wq - zeros) * scales
        gemlite_linear.channel_scale_mode = 0 #no post-scaling

    torch.cuda.empty_cache()

    return gemlite_linear

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

assert config.compile_decode and (not config.quantize)
print(f"Compile config: decode {config.compile_decode}, \
    prefill {config.compile_prefill}. Quantize status: {config.quantize}")
model = model.to(config.device)
print("Model moved to GPU, starting profiling.")

def cache_init(past_key_values, model, config, kv_compiled=False):
    if not kv_compiled:
        # just doing this so that the key and vals are output of cudagraph and hence mutating them in update doesn't cause cudagraph skipping
        past_key_values.early_initialization = torch.compile(past_key_values.early_initialization, mode="reduce-overhead")
    
    past_key_values.early_initialization(
        batch_size=1,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=to_torch_dtype(config.compute_dtype),
        device=config.device,
    )
    return past_key_values

def model_quantize(causal_model, quantized=False):
    quantize_activations = False
    W_nbits = 1
    # gemlite linear layer replacement with GemLite.linear object
    patch_linearlayers(model, patch_linear_to_gemlite)

    if not quantized:
        quantize_(causal_model.model, Int8WeightOnlyConfig())
        quantize_(causal_model.lm_head, Int8WeightOnlyConfig())
    else:
        pass

model.quantization_function = model_quantize

profile_model(model, tokenizer, prompt, config, past_key_values, cache_init)