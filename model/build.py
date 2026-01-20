# import timm
# import os
# import torch
# from .eva_text import eva02_large_patch14_clip_224_text, eva02_base_patch16_clip_224_text
# from .eva_original import (
#     eva02_large_patch14_clip_224,
#     eva02_large_patch14_224,
#     eva02_base_patch14_224,
# )

# __factory = {
#     'eva02_l': eva02_large_patch14_clip_224,
#     'eva02_l_text': eva02_large_patch14_clip_224_text,

#     # 纯 EVA02
#     'eva02_b14': eva02_base_patch14_224,
#     'eva02_l14': eva02_large_patch14_224,

#     # 纯EVA + 文本分支
#     # 'eva02_b14_text': eva02_base_patch14_224_text,
# }

# # def build_model(config, num_classes):
# #     ckpt = "/home/jin/code/MADE/weights/eva02_base_patch14_224/model.safetensors"

# #     model_name = "eva02_base_patch14_224.mim_in22k"

# #     model = timm.create_model(
# #         model_name,
# #         pretrained=True,  # 让它拿到 pretrained_cfg（预处理配置等）
# #         pretrained_cfg_overlay=dict(file=ckpt),  # 关键：权重改成本地文件
# #         num_classes=0,    # 做特征抽取/REID 一般设 0（去掉分类头）
# #     )
# #     return model
# #     # name = config.MODEL.NAME
# #     # # eva02_meta
# #     # if config.MODEL.TYPE == 'eva02_meta':
# #     #     return __factory[name](pretrained=True, num_classes=num_classes, meta_dims=config.MODEL.META_DIMS)

# #     # # 只有 text 模型需要 cfg
# #     # if str(name).endswith("_text") or bool(getattr(config.MODEL, "ADD_TEXT", False)):
# #     #     return __factory[name](pretrained=True, num_classes=num_classes, cfg=config)

# #     # return __factory[name](pretrained=True, num_classes=num_classes)


# try:
#     from safetensors.torch import load_file as safe_load_file
# except Exception:
#     safe_load_file = None

# def _load_local_pretrain(model, ckpt_path: str):
#     if ckpt_path is None or str(ckpt_path).strip() == "":
#         return model
#     ckpt_path = os.path.expanduser(ckpt_path)
#     if not os.path.isfile(ckpt_path):
#         raise FileNotFoundError(f"PRETRAIN_PATH not found: {ckpt_path}")

#     if ckpt_path.endswith(".safetensors"):
#         if safe_load_file is None:
#             raise RuntimeError("safetensors not installed, but ckpt is .safetensors")
#         state = safe_load_file(ckpt_path)
#     else:
#         obj = torch.load(ckpt_path, map_location="cpu")
#         state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

#     # 去掉分类 head（ImageNet=1000 vs ReID=num_classes）
#     for k in list(state.keys()):
#         if k.startswith("head.") or k.startswith("classifier."):
#             state.pop(k, None)

#     msg = model.load_state_dict(state, strict=False)
#     print(f"[Pretrain] load from {ckpt_path}")
#     print(f"[Pretrain] missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")
#     return model

# # def build_model(config, num_classes):
# #     name = config.MODEL.NAME
# #     ckpt_path = getattr(config.MODEL, "PRETRAIN_PATH", "")

# #     # 关键：有本地权重就不要走 timm 在线 pretrained
# #     use_online_pretrained = (ckpt_path is None) or (str(ckpt_path).strip() == "")

# #     if config.MODEL.TYPE == "eva02_meta":
# #         model = __factory[name](pretrained=use_online_pretrained, num_classes=num_classes, meta_dims=config.MODEL.META_DIMS)
# #     else:
# #         model = __factory[name](pretrained=use_online_pretrained, num_classes=num_classes)

# #     if not use_online_pretrained:
# #         model = _load_local_pretrain(model, ckpt_path)

# #     return model
# def build_model(config, num_classes):
#     name = config.MODEL.NAME
#     ckpt_path = getattr(config.MODEL, "PRETRAIN_PATH", "")
#     use_online_pretrained = (ckpt_path is None) or (str(ckpt_path).strip() == "")

#     is_text_model = str(name).endswith("_text") or bool(getattr(config.MODEL, "ADD_TEXT", False))

#     if is_text_model:
#         text_dims = getattr(config.MODEL, "TEXT_DIMS", [512])
#         model = __factory[name](pretrained=use_online_pretrained, num_classes=num_classes, text_dims=text_dims)
#     else:
#         model = __factory[name](pretrained=use_online_pretrained, num_classes=num_classes)

#     if not use_online_pretrained:
#         model = _load_local_pretrain(model, ckpt_path)

#     return model

import os
import torch

from .eva_text import (
    eva02_large_patch14_clip_224_text,
    eva02_base_patch16_clip_224_text,
)

from .eva_original import (
    eva02_large_patch14_clip_224,
    eva02_large_patch14_224,
    eva02_base_patch14_224,
)

# 关键：导入两个版本的 checkpoint_filter_fn
# text 分支模型用 eva_text 的 filter（里面的命名替换与你的修改更一致）
from .eva_text import checkpoint_filter_fn as checkpoint_filter_fn_text
from .eva_original import checkpoint_filter_fn as checkpoint_filter_fn_original

__factory = {
    'eva02_l': eva02_large_patch14_clip_224,
    'eva02_l_text': eva02_large_patch14_clip_224_text,

    # 纯 EVA02
    'eva02_b14': eva02_base_patch14_224,
    'eva02_l14': eva02_large_patch14_224,
}

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


def _load_local_pretrain(model, ckpt_path: str, use_text_filter: bool = False):
    """
    修复 open_clip_model.safetensors / open_clip_pytorch_model.bin 等权重与 EVA key 不匹配：
    - 必须先 checkpoint_filter_fn 过滤/去前缀/改名
    - 自动忽略 text.* / logit_scale
    """
    if ckpt_path is None or str(ckpt_path).strip() == "":
        return model

    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"PRETRAIN_PATH not found: {ckpt_path}")

    # 1) load raw state dict
    if ckpt_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise RuntimeError("safetensors not installed, but ckpt is .safetensors")
        state = safe_load_file(ckpt_path)
    else:
        obj = torch.load(ckpt_path, map_location="cpu")
        # 兼容 {"state_dict": ...} / {"model": ...} / 直接就是 state_dict 的情况
        if isinstance(obj, dict):
            state = obj.get("state_dict", obj.get("model", obj))
        else:
            state = obj

    # 2) 清理分类头（有些 ckpt 里会带）
    for k in list(state.keys()):
        if k.startswith("head.") or k.startswith("classifier."):
            state.pop(k, None)

    # 3) 关键：用 EVA 自带的 filter_fn 做 key 映射（open_clip -> EVA）
    filter_fn = checkpoint_filter_fn_text if use_text_filter else checkpoint_filter_fn_original
    filtered = filter_fn(state, model)

    for k in list(filtered.keys()):
        if k.startswith("head.") or k.startswith("classifier."):
            filtered.pop(k, None)
            
    # 4) load
    msg = model.load_state_dict(filtered, strict=False)
    print(f"[Pretrain] load from {ckpt_path}")
    print(f"[Pretrain] filtered keys: {len(filtered)}")
    print(f"[Pretrain] missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")
    
    if len(msg.unexpected_keys) > 0:
        print("[Pretrain] unexpected example:", msg.unexpected_keys[:30])
    if len(msg.missing_keys) > 0:
        print("[Pretrain] missing example:", msg.missing_keys[:30])

    return model


def build_model(config, num_classes):
    name = config.MODEL.NAME
    ckpt_path = getattr(config.MODEL, "PRETRAIN_PATH", "")
    use_online_pretrained = (ckpt_path is None) or (str(ckpt_path).strip() == "")

    # 是否 text 模型（你现在 eva02_l_text 就是）
    is_text_model = str(name).endswith("_text") or bool(getattr(config.MODEL, "ADD_TEXT", False))

    if is_text_model:
        text_dims = getattr(config.MODEL, "TEXT_DIMS", [512])
        # ⚠️ 这里要把 cfg 也传进去（你 eva_text 里 PromptText 读取 cfg.MODEL.TEXT.*）
        model = __factory[name](
            pretrained=use_online_pretrained,
            num_classes=num_classes,
            text_dims=text_dims,
            cfg=config,
        )
    else:
        model = __factory[name](
            pretrained=use_online_pretrained,
            num_classes=num_classes,
        )

    # 本地权重：一定走 filter_fn
    if not use_online_pretrained:
        model = _load_local_pretrain(model, ckpt_path, use_text_filter=is_text_model)

    return model
