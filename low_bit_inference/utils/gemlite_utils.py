import torch
import gemlite
from hqq.core.quantize import HQQLinear


#Replaces all linear layers with the corresponding processor
def patch_model(model, device, processor, skip_modules=[]):
    #Name modules
    for name, module in model.named_modules():
        module.name = name

    #Patching fct
    def _patching_fct(layer, device, skip_modules):
        layer = layer.to(device, non_blocking=True)
        if(any(s in layer.name for s in skip_modules)):
            return layer
        else:
            if(isinstance(layer, torch.nn.Linear)):
                return processor(device=device).from_linear(layer)
            elif(isinstance(layer, HQQLinear)):
                return processor(device=device).from_hqqlinear(layer)
            else:
                return layer

    #Replaces linear layers
    def _patch_linearlayers(model, fct, device, skip_modules):
        for name, layer in model.named_children():
            if isinstance(layer, (torch.nn.Linear, HQQLinear)):
                setattr(model, name, fct(layer, device, skip_modules))
            else:
                _patch_linearlayers(layer, fct, device, skip_modules)

    #Apply patch
    _patch_linearlayers(model, _patching_fct, device, skip_modules)

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
    import gemlite.helper

    gemlite.helper.A16Wn_HQQ_INT.from_linear = gemlite.helper.A16Wn_HQQ_INT.from_hqqlinear
    gemlite.helper.A8Wn_HQQ_INT_dynamic.from_linear = gemlite.helper.A16Wn_HQQ_INT.from_hqqlinear
    gemlite.helper.A16W158_INT.from_linear = gemlite.helper.A16W158_INT.from_bitlinear
    gemlite.helper.A8W158_INT_dynamic.from_linear = gemlite.helper.A8W158_INT_dynamic.from_bitlinear

    assert id(gemlite.helper.A16Wn_HQQ_INT.from_linear) == id(gemlite.helper.A16Wn_HQQ_INT.from_hqqlinear)
    assert id(gemlite.helper.A8Wn_HQQ_INT_dynamic.from_linear) == id(gemlite.helper.A16Wn_HQQ_INT.from_hqqlinear)
    assert id(gemlite.helper.A16W158_INT.from_linear) == id(gemlite.helper.A16W158_INT.from_bitlinear)
    assert id(gemlite.helper.A8W158_INT_dynamic.from_linear) == id(gemlite.helper.A8W158_INT_dynamic.from_bitlinear)