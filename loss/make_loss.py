# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch.nn as nn
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .proxy_loss import ProxyAnchorLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATA.SAMPLER
    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

        # ---------------- Proxy-Anchor cached config (minimal-invasion) ----------------
    text_cfg = getattr(cfg.MODEL, "TEXT", None)
    pa_on = False
    pa_w = 0.0
    pa_loss_fn = None
    if (text_cfg is not None) and bool(getattr(text_cfg, "PROXY_ANCHOR_ON", False)):
        pa_w = float(getattr(text_cfg, "PROXY_ANCHOR_W", 0.0))
        if pa_w > 0.0:
            pa_on = True
            pa_alpha = float(getattr(text_cfg, "PROXY_ANCHOR_ALPHA", 32.0))
            pa_delta = float(getattr(text_cfg, "PROXY_ANCHOR_DELTA", 0.1))
            pa_loss_fn = ProxyAnchorLoss(alpha=pa_alpha, delta=pa_delta)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATA.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, model=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # ---------------- CE / ID loss (保持你原逻辑不动) ----------------
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                # ---------------- Triplet loss (保持你原逻辑不动) ----------------
                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                    TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    feat_main = feat[0]
                else:
                    TRI_LOSS = triplet(feat, target)[0]
                    feat_main = feat

                total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                # ---------------- Proxy-Anchor（最小侵入新增项） ----------------
                # 只从 cfg.MODEL.TEXT 读取，命名与 PROXY_GATE_* 完全一致
                text_cfg = getattr(cfg.MODEL, "TEXT", None)
                # if (text_cfg is not None) and bool(getattr(text_cfg, "PROXY_ANCHOR_ON", False)):
                #     pa_w = float(getattr(text_cfg, "PROXY_ANCHOR_W", 0.0))
                #     if pa_w > 0.0 and (model is not None):
                #         # 统一 proxy 系统：proxy 来自分类头 head.weight
                #         m = model.module if hasattr(model, "module") else model
                #         if not hasattr(m, "head") or not hasattr(m.head, "weight"):
                #             raise AttributeError("Proxy-Anchor expects model.head.weight to exist (shared proxy system).")

                #         proxies = m.head.weight  # [num_classes, feat_dim]
                #         pa_alpha = float(getattr(text_cfg, "PROXY_ANCHOR_ALPHA", 32.0))
                #         pa_delta = float(getattr(text_cfg, "PROXY_ANCHOR_DELTA", 0.1))

                #         # 复用你已有实现 proxy_loss.py 里的 ProxyAnchorLoss :contentReference[oaicite:1]{index=1}
                #         pa_loss_fn = ProxyAnchorLoss(alpha=pa_alpha, delta=pa_delta)
                #         PA_LOSS = pa_loss_fn(feat_main, target, proxies.to(feat_main.device))
                #         print("PA_w:", pa_w, "PA:", float(PA_LOSS.detach().cpu()))
                #         total = total + pa_w * PA_LOSS
                                # ---------------- Proxy-Anchor（最小侵入新增项） ----------------
                PA_LOSS = None
                if pa_on and (model is not None):
                    # 统一 proxy 系统：proxy 来自分类头 head.weight
                    m = model.module if hasattr(model, "module") else model
                    if not hasattr(m, "head") or not hasattr(m.head, "weight"):
                        raise AttributeError("Proxy-Anchor expects model.head.weight to exist (shared proxy system).")

                    proxies = m.head.weight  # [num_classes, feat_dim]
                    PA_LOSS = pa_loss_fn(feat_main, target, proxies)
                    total = total + pa_w * PA_LOSS

                # ---- stash breakdown for logger (no behavior change) ----
                try:
                    loss_func._last = {
                        "id": float(ID_LOSS.detach().item()),
                        "tri": float(TRI_LOSS.detach().item()),
                        "pa": float(PA_LOSS.detach().item()) if PA_LOSS is not None else 0.0,
                        "pa_w": float(pa_w),
                    }
                except Exception:
                    pass

                return total
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                    'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    # elif cfg.DATA.SAMPLER == 'softmax_triplet':
    #     def loss_func(score, feat, target, target_cam):
    #         if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
    #             if cfg.MODEL.IF_LABELSMOOTH == 'on':
    #                 if isinstance(score, list):
    #                     ID_LOSS = [xent(scor, target) for scor in score[1:]]
    #                     ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
    #                     ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
    #                 else:
    #                     ID_LOSS = xent(score, target)

    #                 if isinstance(feat, list):
    #                         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
    #                         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
    #                         TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
    #                 else:
    #                         TRI_LOSS = triplet(feat, target)[0]

    #                 return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
    #                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
    #             else:
    #                 if isinstance(score, list):
    #                     ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
    #                     ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
    #                     ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
    #                 else:
    #                     ID_LOSS = F.cross_entropy(score, target)

    #                 if isinstance(feat, list):
    #                         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
    #                         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
    #                         TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
    #                 else:
    #                         TRI_LOSS = triplet(feat, target)[0]

    #                 return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
    #                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
    #         else:
    #             print('expected METRIC_LOSS_TYPE should be triplet'
    #                   'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


