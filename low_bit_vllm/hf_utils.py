from transformers import AutoModelForCausalLM, AutoTokenizer
from config_utils import to_torch_dtype

def autoname_modules(m):
    for name, module in m.named_modules():
        module.name = name  

def prep_for_inference(model, tokenizer):
    # ref: https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py#L238
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})
    tokenizer.padding_side = "right"
    model.eval()
    model.generation_config.cache_implementation = "static"
    model.config.use_cache = True
    return model, tokenizer

def load_model_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=config.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=to_torch_dtype(config.compute_dtype),
        attn_implementation="sdpa",
        cache_dir=config.cache_dir,
        device_map="cpu"
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {params}")
    model, tokenizer = prep_for_inference(model, tokenizer)
    autoname_modules(model)
    return model, tokenizer