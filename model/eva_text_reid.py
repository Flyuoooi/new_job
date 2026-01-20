import torch
import torch.nn as nn
import torch.nn.functional as F

from .eva_original import eva02_base_patch14_224
from .text_branch import CoOpTextBranch


class EVAReIDWithText(nn.Module):
    """
    纯 EVA02 视觉主干 +（训练期）CoOp-Text 分支
    推理期：只返回视觉 feat（与 MADE 评测逻辑兼容）
    """
    def __init__(self, num_classes: int, pretrained: bool = True, cfg=None):
        super().__init__()
        assert cfg is not None, "cfg is required"

        self.cfg = cfg
        # 1) 纯 EVA02（timm-style）
        self.visual = eva02_base_patch14_224(pretrained=pretrained, num_classes=num_classes)

        # EVA 输出 feat_dim=768（base）
        self.vis_dim = 768
        self.txt_dim = 512  # CLIP text 默认 512

        # 2) 把视觉 feat 投影到 text dim，用于对齐/引导 mask
        self.img_to_txt = nn.Linear(self.vis_dim, self.txt_dim)

        # 3) 文本分支（CoOp + attn-mask + residual）
        self.text_branch = CoOpTextBranch(
            num_classes=num_classes,
            cfg=cfg,
            txt_dim=self.txt_dim,
        )

        # 温度
        self.itc_temp = float(getattr(cfg.MODEL.TEXT, "TEMP", 0.07))

    @torch.no_grad()
    def _estimate_cloth_dir(self, img_txt: torch.Tensor, clothes_id: torch.Tensor):
        """
        用 batch 内 clothes_id 的均值方向近似 cloth direction（轻量，不依赖属性/SOLIDER）
        img_txt: [B, 512]
        clothes_id: [B]
        return cloth_dir_per_sample: [B, 512]
        """
        B, D = img_txt.shape
        device = img_txt.device
        clothes_id = clothes_id.long()

        uniq = torch.unique(clothes_id)
        # 每个 clothes 的均值
        means = []
        for c in uniq.tolist():
            m = img_txt[clothes_id == c].mean(dim=0, keepdim=True)
            means.append(m)
        means = torch.cat(means, dim=0)  # [C, D]
        global_mean = img_txt.mean(dim=0, keepdim=True)

        dirs = F.normalize(means - global_mean, dim=1, eps=1e-6)  # [C, D]

        # 给每个样本分配对应 clothes 的 dir
        c2idx = {c: i for i, c in enumerate(uniq.tolist())}
        idx = torch.tensor([c2idx[int(c)] for c in clothes_id.tolist()], device=device)
        return dirs[idx]  # [B, D]

    def forward(self, images, meta=None, targets=None, clothes_id=None):
        """
        - 兼容原 MADE：processor 里可能传 meta（我们忽略）
        - 训练时若 cfg.MODEL.ADD_TEXT=True，需要 processor 传 targets/clothes_id
        """
        # 视觉特征（统一用 forward_features + forward_head，避免 self.visual.forward 的 train/eval 分支差异）
        x = self.visual.forward_features(images)
        feat = self.visual.forward_head(x, pre_logits=True)  # [B, 768]

        if not self.training:
            return feat

        cls_score = self.visual.head(feat)

        aux = {}
        if bool(getattr(self.cfg.MODEL, "ADD_TEXT", True)):
            assert targets is not None, "targets required when ADD_TEXT=True"
            assert clothes_id is not None, "clothes_id required when ADD_TEXT=True"

            img_txt = self.img_to_txt(feat)  # [B, 512]
            img_txt_n = F.normalize(img_txt.float(), dim=1, eps=1e-6)

            cloth_dir = self._estimate_cloth_dir(img_txt_n, clothes_id)  # [B, 512]

            # 文本分支输出
            out = self.text_branch(img_txt_n, targets, cloth_dir=cloth_dir)
            # out: dict with keys: txt_full, txt_masked, txt_rec, resid_loss, orth_loss
            txt_rec_n = F.normalize(out["txt_rec"].float(), dim=1, eps=1e-6)

            # -------- ITC（监督：image -> unique pid text）--------
            uniq_pids, inv = torch.unique(targets.long(), return_inverse=True)
            # 取 unique pid 的文本向量（同一 pid 共享 prompt）
            txt_u = txt_rec_n.new_empty((uniq_pids.numel(), txt_rec_n.shape[1]))
            for i, pid in enumerate(uniq_pids.tolist()):
                # 从 batch 里随便取一个该 pid 的文本即可（它们相同）
                txt_u[i] = txt_rec_n[(targets == pid).nonzero(as_tuple=True)[0][0]]

            logits = (img_txt_n @ txt_u.t()) / self.itc_temp  # [B, U]
            itc_loss = F.cross_entropy(logits, inv)

            aux["itc_loss"] = itc_loss
            aux["resid_loss"] = out["resid_loss"]
            aux["orth_loss"] = out["orth_loss"]

        return cls_score, feat, aux


# 给 build.py 注册用的工厂函数
def eva02_base_patch14_224_text(pretrained=False, num_classes=0, **kwargs):
    cfg = kwargs.get("cfg", None)
    return EVAReIDWithText(num_classes=num_classes, pretrained=pretrained, cfg=cfg)
