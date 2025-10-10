# Original Author and code: mobicham, https://gist.github.com/mobicham/8e9d009a1ad7fd138e1df849a326a6ed
#pip install torch torchvision --upgrade;
#pip install transformers==4.53.2;
#DISABLE_CUDA=1 pip install git+https://github.com/mobiusml/hqq/;
#pip install git+https://github.com/mobiusml/gemlite/;
#TRITON_PRINT_AUTOTUNING=1 ipython ...
################################################################################################################
import torch
from omegaconf import OmegaConf
from .hf_loader import load_model_tokenizer_prompt_cache
from .utils.config_utils import get_config, to_torch_dtype
from .utils.profile_utils import profile_model

import gemlite
from gemlite.core import TORCH_TO_DTYPE
from gemlite.helper import (
    cleanup_linear,
    patch_model,
    warmup
)

from gemlite.helper import (
    A16W8,
    A16Wn,
    A16W8_INT,
    A16Wn_HQQ_INT,
    A16W8_HQQ_INT,
    A16W4_HQQ_INT,
    A16W2_HQQ_INT,
    A16W1_HQQ_INT,
    A16Wn_MXFP,
    A16W8_MXFP,
    A16W4_MXFP
)

from gemlite.helper import (
    A8W8_dynamic,
    A8W8_int8_dynamic,
    A8W8_INT8_dynamic,
    A8W8_fp8_dynamic,
    A8W8_FP8_dynamic,
    A8Wn_HQQ_INT_dynamic,
    A8W4_HQQ_INT_dynamic,
    A8W2_HQQ_INT_dynamic,
    A8W8_MXFP_dynamic,
    A8Wn_MXFP_dynamic,
    A8W8_MXFP_dynamic,
    A8W4_MXFP_dynamic
)

from gemlite.helper import (
    A4W4_MXFP_dynamic,
    A4W4_NVFP_dynamic
)

from gemlite.helper import (
    A16W158_INT,
    A8W158_INT_dynamic
)

from gemlite import DType, GemLiteLinear

from hqq.utils.generation_hf import HFGenerator

gemlite.set_autotune("max")  # fast for fast startup, but slow perf. max for slow startup, but best perf.

device        = 'cuda:0'
compute_dtype = torch.float16
model_id      = "unsloth/Meta-Llama-3.1-8B-Instruct"
cache_dir     = "/root/.cache/huggingface/hub"

print(f"Loading pretrained model and tokenizer: {model_id}.")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model     = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=compute_dtype,
    attn_implementation="sdpa",
    cache_dir=cache_dir,
    device_map="cpu"
)

params = sum(p.numel() for p in model.parameters())

print(f"Model loaded {model_id}. Number of parameters in model: {params}")

quantize_activations = False
W_nbits = 1

#4090 RTX - batch_size = 1
#A16W16: 60 tokens/sec
#A16W8 : 101 tokens/sec
#A16W4 : 186 tokens/sec
#A16W2 : 296 tokens/sec
#A16W1 : 344 tokens/sec
################################################################################################################
def autoname_modules(m):
    for name, module in m.named_modules():
        module.name = name  

def patch_linearlayers(model, fct):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.Linear):
            setattr(model, name, fct(layer, name))
        else:
            patch_linearlayers(layer, fct)

def patch_linear_to_gemlite(layer, name):
    global W_nbits, quantize_activations
    layer = layer.to(device, non_blocking=True)
    #if('lm_head' in name): return layer

    weight, bias = layer.weight, layer.bias

    if(isinstance(weight, torch.nn.Parameter)):
        weight = weight.data
    if(isinstance(bias, torch.nn.Parameter)):
        bias = bias.data

    out_features  = weight.shape[0]
    in_features   = weight.shape[1]

    ############################################################################
    #Quantize: A8Wn only works with symmetric quant with single shift
    assert W_nbits in [8, 4, 2, 1.58, 1], "Invalid W_nbits."
    nbits_to_max_val = {8:127, 4:7, 2:1, 1.58:1, 1:1}
    weight  = weight.to(device=device, dtype=compute_dtype)
    
    #weight ~ (W_q.float() - zeros) * scales
    if W_nbits == 1: #BUG in W_nbits==1. with A8 activations add shift only x 2 in new W_group_mode.
        W_q = weight.sign() #[-1, 1]
        W_q[W_q == 0] = 1
        zeros  = 0.50
        scales = torch.ones((out_features, 1), dtype=compute_dtype, device=device) * weight.abs().max() * 2
        W_q    = ((W_q + zeros) / 2).to(torch.uint8) #[0, 1]
    elif W_nbits == 1.58:
        W_q     = weight.sign() #[-1, 0, 1]
        zeros   = 1 
        scales  = torch.ones((out_features, 1), dtype=compute_dtype, device=device) * weight.abs().max()
        W_q     = (W_q + zeros).to(torch.uint8) #[0, 1, 2]
    else:
        max_val = nbits_to_max_val[W_nbits]
        zeros   = max_val
        scales  = weight.float().abs().amax(axis=1, keepdim=True) / max_val
        W_q     = (((weight / scales) + zeros).round_()).to(torch.uint8)
    ############################################################################


    if quantize_activations:
        input_dtype = DType.INT8
        scale_activations = True
    else:
        input_dtype = TORCH_TO_DTYPE[compute_dtype]
        scale_activations = False

    gemlite_linear = GemLiteLinear(round(W_nbits), 
                    group_size=in_features, 
                    in_features=in_features, 
                    out_features=out_features, 
                    input_dtype=input_dtype, 
                    output_dtype=TORCH_TO_DTYPE[compute_dtype], 
                    scaled_activations=scale_activations,
                    )

    bias = bias.clone().to(device=self.device, dtype=compute_dtype) if (bias is not None) else None

    gemlite_linear.pack(W_q, scales=scales, zeros=zeros, bias=bias, fma_mode=False)

    if(quantize_activations):
        gemlite_linear.W_group_mode       = 1 #shift only (Wq - zeros)
        gemlite_linear.channel_scale_mode = 3 #activations + weight
    else:
        gemlite_linear.W_group_mode       = 3 #(Wq - zeros) * scales
        gemlite_linear.channel_scale_mode = 0 #no post-scaling

    del weight, W_q, bias
    torch.cuda.empty_cache()

    return gemlite_linear

autoname_modules(model)
patch_linearlayers(model, patch_linear_to_gemlite)
model = model.to(device)
print("Model moved to GPU, starting profiling.")
torch.cuda.synchronize()

################################################################################################################

# use CUDA_BLOCKING_MODE=1 to do sync after each kernel dispatch and to know precisely what part of code is slow.

skip_first = 0
wait = 0
warmup = 0
active = 1
repeat = 1

profiling_schedule = torch.profiler.schedule(
    skip_first = skip_first,
    wait = wait,
    warmup = warmup,
    active = active,
    repeat = repeat
)

for i in range(skip_first + repeat*(wait + warmup + active)):
    print(f"Profiling Iteration {i}.")
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule = profiling_schedule,
        on_trace_ready = torch.profiler.tensorboard_trace_handler('./log/gemlite'),
        record_shapes = True,
        profile_memory = True,
        with_stack = True,  # this will add overhead, set it to False for benchmarking.
        with_flops = True,
    ) as prof:
        with torch.inference_mode():
            gen = HFGenerator(
                model,
                tokenizer,
                max_new_tokens=1024,
                do_sample=False,
                compile="partial",
                compile_options={"mode": "max-autotune-no-cudagraphs", "fullgraph": True},
            ).enable_cuda_graph().warmup()

            out = gen.generate("Write an essay about large language models.", print_tokens=False)
        prof.step()
print("profiling complete")

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
print("-"*20)
print(prof.key_averages(group_by_stack_n=8).table(sort_by="cpu_time_total", row_limit=100))

prof.export_chrome_trace("trace.json")