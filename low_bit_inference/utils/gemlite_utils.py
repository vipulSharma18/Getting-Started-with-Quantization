"""
Source/ref: https://github.com/dropbox/gemlite/blob/master/gemlite/helper.py
"""
import os
import torch
import gemlite
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig


#Replaces all linear layers with the corresponding processor
def patch_model(model, device, processor, skip_modules=['lm_head', 'vision', 'visual', 'embed_tokens'], group_size=64):
    """
    Helper function to quantize the whole model from cpu device. 
    The group_size parameter is only used in HQQ for W_nbits <=4, all the other configs do not use this parameter.
    """
    #Name modules
    for name, module in model.named_modules():
        module.name = name

    #Patching fct
    def _patching_fct(layer, device, skip_modules, group_size):
        layer = layer.to(device, non_blocking=True)
        if(any(s in layer.name for s in skip_modules)):
            return layer
        else:
            if(isinstance(layer, torch.nn.Linear)):
                if hasattr(processor, 'from_hqqlinear'):
                    return processor.from_linear(layer, group_size)
                else:
                    return processor.from_linear(layer)
            elif(isinstance(layer, HQQLinear)):
                return processor.from_hqqlinear(layer)
            else:
                return layer

    #Replaces linear layers
    def _patch_linearlayers(model, fct, device, skip_modules, group_size):
        for name, layer in model.named_children():
            if isinstance(layer, (torch.nn.Linear, HQQLinear)):
                setattr(model, name, fct(layer, device, skip_modules, group_size))
            else:
                _patch_linearlayers(layer, fct, device, skip_modules, group_size)

    #Apply patch
    _patch_linearlayers(model, _patching_fct, device, skip_modules, group_size)

# add from_linear method to classes that only have from_hqqlinear for uniform method specification
def monkeypatch_gemlite():
    """
    Notes:
    -----
    A16W8 has from_linear. A16W8_INT8, A16W8_FP8 subclass A16W8 so have it as well.
    A16Wn has from_hqqlinear and mxfp_from_linear.
        Subclasses HQQ_INT: missing from_linear A16Wn_HQQ_INT.
        Subclasses MXFP: has from_linear A16Wn_MXFP.
        Subclasses HQQ_INT_dynamic: missing from_linear A8Wn_HQQ_INT_dynamic.

    A8W8_dynamic: has from_linear.
    A8W8_MXFP_dynamic: has from_linear.
    A8Wn_MXFP_dynamic: has from_linear.
    A4W4_MXFP_dynamic: has from_linear.
    A4W4_NVFP_dynamic: has from_linear.
    A16W158_INT, A8W158_INT_dynamic: missing from_linear but has from_bitlinear.

    Overrides:
    ---------
    1. A16Wn_HQQ_INT
    2. A8Wn_HQQ_INT_dynamic
    3. A16W158_INT, A8W158_INT_dynamic
    """

    # simple aliasing of methods for bitlinear
    gemlite.helper.A16W158_INT.from_linear = gemlite.helper.A16W158_INT.from_bitlinear
    gemlite.helper.A8W158_INT_dynamic.from_linear = gemlite.helper.A8W158_INT_dynamic.from_bitlinear

    # for int hqq: first need to convert layer to hqqlinear before calling from_hqqlinear on it.
    def from_linear(self, layer, group_size=64):
        dtype = layer.weight.dtype if hasattr(layer.weight, 'dtype') else torch.bfloat16
        config_group_size = group_size if self.W_nbits<=4 else None
        quant_config   = BaseQuantizeConfig(nbits=self.W_nbits, group_size=config_group_size, axis=1)
        linear         = HQQLinear(layer, quant_config=quant_config, compute_dtype=dtype, device=self.device)
        return self.from_hqqlinear(linear)

    gemlite.helper.A16Wn_HQQ_INT.from_linear = from_linear
    gemlite.helper.A8Wn_HQQ_INT_dynamic.from_linear = from_linear

def get_default_cache_config(root_path = None):
    """
    Gemlite comes with gpu model specific optimal configs for triton. We try to load them to avoid autotune time.
    """
    if root_path is None:
      try:
        root_path = os.path.join(gemlite.__path__[0], "configs/")
      except ModuleNotFoundError as e:
        raise e
    
    def get_tags(path):
        return [f.split('.')[0] for f in os.listdir(path)]

    name = torch.cuda.get_device_properties(0).name.lower().replace(' ', '_')
    tags = get_tags(root_path)
    tags.sort(key=len, reverse=True)
    print("Picking from configs:", tags)

    selected_tag = None
    for tag in tags:
        if(tag in name):
            selected_tag = os.path.join(root_path, tag + '.json')
            break
    if selected_tag is None:
      print("Warning: Do manual autotune, no cached config found.")

    return selected_tag