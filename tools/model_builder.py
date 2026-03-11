from models.action_converter import ActionConverter, EightAngularActionConverter
import config

import timm

import timm.utils.cuda
from transformers import MambaConfig as HFMambaConfig, MambaModel as HFMamba
from models.custom_hf_mamba import MambaModel as HFCustomMamba

import torch.nn as nn
from models.backbones import SimpleJoiner, SimpleBackbone, ConvBackbone
from models.foveater import FoveationProperties

from models.foveated_mamba_6 import FoveatedMamba6
from models.foveated_mamba_7 import FoveatedMamba7





from models.vit_recons import TransformerBaseline, TransformerSequential, TransformerEncoderDecoder

import models.vit_recons as vit_module
import models.heads as mamba_heads_module



def create_model(model_name, *args, **kwargs):
    if model_name == 'Mamba':
        return create_mamba(*args, **kwargs)
    elif model_name == 'FoveatedMamba6':
        return create_foveated_mamba_6(*args, **kwargs)
    elif model_name == 'FoveatedMamba7':
        return create_foveated_mamba_7(*args, **kwargs)
    elif model_name == 'TransformerBaseline':
        return create_transformer_baseline(*args, **kwargs)
    elif model_name == 'TransformerSequential':
        return create_transformer_sequential(*args, **kwargs)
    elif model_name == 'TransformerEncoderDecoder':
        return create_transformer_encoder_decoder(*args, **kwargs)
    else:
        return timm.create_model(model_name=model_name, *args, **kwargs)


def create_transformer_baseline(foveation_properties_cfg, decoder_cfg=None, *args, **kwargs):
    foveation_properties = FoveationProperties(**foveation_properties_cfg)
    decoder = create_transformer_decoder(**decoder_cfg) if decoder_cfg is not None else None
    return TransformerBaseline(foveation_properties=foveation_properties, decoder=decoder, *args, **kwargs)

def create_transformer_sequential(foveation_properties_cfg, decoder_cfg=None, *args, **kwargs):
    foveation_properties = FoveationProperties(**foveation_properties_cfg)
    decoder = create_transformer_decoder(**decoder_cfg) if decoder_cfg is not None else None
    return TransformerSequential(foveation_properties=foveation_properties, decoder=decoder, *args, **kwargs)

def create_transformer_encoder_decoder(foveation_properties_cfg, decoder_cfg=None, *args, **kwargs):
    foveation_properties = FoveationProperties(**foveation_properties_cfg)
    decoder = create_transformer_decoder(**decoder_cfg) if decoder_cfg is not None else None
    return TransformerEncoderDecoder(foveation_properties=foveation_properties, decoder=decoder, *args, **kwargs)


def create_transformer_decoder(model_name, *args, **kwargs):
    return getattr(vit_module, model_name)(*args, **kwargs)


def create_mamba(mamba_name, drop_prob=0.0, *args, **kwargs):
    if mamba_name == 'HFMamba':
        mamba = HFMamba(HFMambaConfig(*args, **kwargs)).to(config.DEVICE)
    elif mamba_name == 'HFCustomMamba':
        mamba = HFCustomMamba(HFMambaConfig(*args, **kwargs), drop_prob=drop_prob).to(config.DEVICE)
    else:
        raise NameError(f'Unknown mamba name {mamba_name}.')
    return mamba






def create_foveated_mamba_7(mamba_cfg, backbones_cfg, foveation_properties_cfg, joiner_cfg,
                            heads_cfg, action_converter_cfg, *args, **kwargs):
    mamba = create_mamba(**mamba_cfg)
    foveation_properties = FoveationProperties(**foveation_properties_cfg)
    joiner = create_joiner(**joiner_cfg)
    heads = create_module_list(heads_cfg, create_head)
    backbones = create_module_list(backbones_cfg, create_foveated_mamba_backbone)
    action_converter = create_action_converter(**action_converter_cfg)
    foveated_mamba2 = FoveatedMamba7(mamba=mamba, heads=heads, action_converter=action_converter, joiner=joiner,
                                     foveation_properties=foveation_properties, backbones=backbones, *args, **kwargs)
    return foveated_mamba2


def create_foveated_mamba_6(mamba_cfg, backbones_cfg,  joiner_cfg,
                            heads_cfg, action_converter_cfg, foveation_properties_cfg=None, *args, **kwargs):
    mamba = create_mamba(**mamba_cfg)
    if foveation_properties_cfg is None:
        foveation_properties_cfg = {'patch_sizes': (4, ), 'resize_sizes': (4, )}
    foveation_properties = FoveationProperties(**foveation_properties_cfg)
    joiner = create_joiner(**joiner_cfg)
    heads = create_module_list(heads_cfg, create_head)
    backbones = create_module_list(backbones_cfg, create_foveated_mamba_backbone)
    action_converter = create_action_converter(**action_converter_cfg)
    foveated_mamba2 = FoveatedMamba6(mamba=mamba, heads=heads, action_converter=action_converter, joiner=joiner,
                                     foveation_properties=foveation_properties, backbones=backbones, *args, **kwargs)
    return foveated_mamba2



def create_module_list(modules_cfg, create_module_fct):
    modules = []
    module_keys = sorted(list(modules_cfg.keys()), key=int)
    for key in module_keys:
        modules.append(create_module_fct(**modules_cfg[key]))
    return nn.ModuleList(modules)


def create_head(head_name, *args, **kwargs):
    return getattr(mamba_heads_module, head_name)(*args, **kwargs)


def create_backbones(backbones_cfg):
    backbones = []
    backbone_keys = sorted(list(backbones_cfg.keys()), key=int)
    for key in backbone_keys:
        backbones.append(create_foveated_mamba_backbone(**backbones_cfg[key]))
    return nn.ModuleList(backbones)


def create_action_converter(action_converter_name, *args, **kwargs):
    if action_converter_name == "ActionConverter":
        return ActionConverter(*args, **kwargs)
    if action_converter_name == "EightAngularActionConverter":
        return EightAngularActionConverter(*args, **kwargs)
    else:
        raise NameError(f'Unknown name {action_converter_name}')


def create_joiner(joiner_name, *args, **kwargs):
    if joiner_name == 'SimpleJoiner':
        return SimpleJoiner(*args, **kwargs)
    else:
        raise NameError(f'Unknown name {joiner_name}')


def create_foveated_mamba_backbone(backbone_name, *args, **kwargs):
    if backbone_name == 'SimpleBackbone':
        backbone = SimpleBackbone()
    elif backbone_name == 'Backbone':
        raise NotImplementedError()
    elif backbone_name == 'ConvBackbone':
        backbone = ConvBackbone(*args, **kwargs)

    else:
        raise NameError(f'Unknown backbone {backbone_name}')
    return backbone
