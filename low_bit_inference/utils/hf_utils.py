from transformers import AutoModelForCausalLM, AutoTokenizer
from .config_utils import to_torch_dtype


def autoname_modules(m):
    for name, module in m.named_modules():
        module.name = name

def prep_for_inference(model, tokenizer, config):
    # ref: https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py#L238
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})
    tokenizer.padding_side = config.padding_side
    model.eval()
    return model, tokenizer

def load_model_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=config.cache_dir)  # if a rust based tokenizer is not avialable, this falls back to Python implementation which is slower.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=to_torch_dtype(config.compute_dtype),
        attn_implementation=config.attn_implementation,
        cache_dir=config.cache_dir,
        device_map=config.device,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {params}")
    model, tokenizer = prep_for_inference(model, tokenizer, config)
    autoname_modules(model)
    return model, tokenizer