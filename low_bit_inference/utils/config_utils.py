import os
import sys
from datetime import datetime
from enum import Enum
from typing import Optional
import torch
from omegaconf import OmegaConf
from dataclasses import dataclass


class DType(Enum):
    fp32 = torch.float32
    fp16 = torch.float16
    bf16 = torch.bfloat16
    uint8 = torch.uint8
    int8 = torch.int8
    #  from here onwards, maybe use torchao dtypes
    fp8_e4m3 = torch.float8_e4m3fn  # these are mx fp8 i believe, confirm and support both mx and nvfp8
    fp8_e5m2 = torch.float8_e5m2
    int4 = torch.float16
    mxfp4 = torch.float16
    nvfp4 = torch.float16

@dataclass
class ProfileConfig:
    # hf model args
    device: str = "cuda:0"
    compute_dtype: DType = DType.fp16
    model_id: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    cache_dir: str = "/root/.cache/huggingface/hub"

    # profiling args
    profiling_dir: str = "log/"
    skip_first: int = 0
    wait: int = 0
    warmup: int = 0
    active: int = 1
    repeat: int = 1
    tps_only: bool = False

    # inference args
    prompt: str = "Write an essay about large language models."
    max_new_tokens: int = 1024
    chat_template: str = ""
    do_sample: bool = False
    top_k: Optional[int] = 5
    top_p: Optional[float] = 0.5
    temperature: Optional[float] = 0.6
    use_cache: bool = True
    cache_implementation: str = "static"
    attn_implementation: str = "sdpa"

    # tokenizer args
    padding_side: str = "left"

def to_torch_dtype(x):
    unsupported = ["fp8", "int4", "mxfp4", "nvfp4"]
    if isinstance(x, DType):
        if x.name in unsupported:
            raise ValueError(f"DType {x.name} not supported yet.")
        return x.value
    elif isinstance(x, str):
        if x in unsupported:
            raise ValueError(f"DType {x} not supported yet.")
        return DType[x].value
    return x

def get_config():
    if len(sys.argv) > 1 \
        and not sys.argv[1].startswith('--') \
        and (sys.argv[1].endswith('yaml') or sys.argv[1].endswith('yml')):
        # First argument is a YAML file path
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"The config file passed {config_file} doesn't exist.")
            raise ValueError(f"current dir: {os.getcwd()}, requested file: {config_file}")
        yml_conf = OmegaConf.load(config_file)
        sys.argv = ([sys.argv[0]] + sys.argv[2:])
    else:
        # No YAML file provided, use CLI arguments only
        yml_conf = OmegaConf.create({})
    
    schema = OmegaConf.structured(ProfileConfig)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(schema, yml_conf, cli_conf)

    if not conf.do_sample:
        conf.top_k = None
        conf.temperature = None
        conf.top_p = None
    m_d_hh_mm_ss = datetime.now().strftime("%m_%d_%H_%M_%S")
    conf.profiling_dir = os.path.join(conf.profiling_dir, m_d_hh_mm_ss)
    os.makedirs(conf.profiling_dir, exist_ok=True)
    return conf