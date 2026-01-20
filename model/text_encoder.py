import torch.nn as nn
class ResNormLayer(nn.Module):
    def __init__(self, linear_size,):
        super(ResNormLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.norm_fn1 = nn.LayerNorm(self.l_size)
        self.norm_fn2 = nn.LayerNorm(self.l_size)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        out = x + y
        return out

# from __future__ import annotations

# import torch
# import torch.nn as nn


# class ResNormLayer(nn.Module):
#     """
#     Original (strong) residual normalization layer used for low-quality meta.
#     """
#     def __init__(self, linear_size: int):
#         super().__init__()
#         self.nonlin1 = nn.ReLU(inplace=True)
#         self.nonlin2 = nn.ReLU(inplace=True)
#         self.norm_fn1 = nn.LayerNorm(linear_size)
#         self.norm_fn2 = nn.LayerNorm(linear_size)
#         self.w1 = nn.Linear(linear_size, linear_size)
#         self.w2 = nn.Linear(linear_size, linear_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.w1(x)
#         y = self.nonlin1(y)
#         y = self.norm_fn1(y)
#         y = self.w2(y)
#         y = self.nonlin2(y)
#         y = self.norm_fn2(y)
#         out = x + y
#         return out


# class LightResNormLayer(nn.Module):
#     """
#     Lighter residual block for high-quality semantics:
#       x -> LN -> Linear -> GELU -> Linear -> scaled residual
#     - smaller hidden ratio
#     - learnable residual scale (init small)
#     """
#     def __init__(self, dim: int, hidden_ratio: float = 0.5, init_scale: float = 0.1):
#         super().__init__()
#         hidden = max(1, int(dim * hidden_ratio))
#         self.ln = nn.LayerNorm(dim)
#         self.fc1 = nn.Linear(dim, hidden)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden, dim)
#         self.scale = nn.Parameter(torch.tensor(float(init_scale)))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.ln(x)
#         y = self.fc1(y)
#         y = self.act(y)
#         y = self.fc2(y)
#         return x + self.scale * y


# # def build_text_heads(
# #     in_dim: int,
# #     out_dim: int,
# #     head2_hidden_ratio: float = 0.5,
# #     head2_init_scale: float = 0.1,
# # ):
# #     """
# #     Returns (head1, head2) for injecting high-quality text semantics.

# #     head1: Linear -> ReLU -> LayerNorm (NO residual)
# #     head2: Linear -> ReLU -> LayerNorm -> LightResNormLayer (weak residual)
# #     """
# #     head1 = nn.Sequential(
# #         nn.Linear(in_dim, out_dim),
# #         nn.ReLU(inplace=True),
# #         nn.LayerNorm(out_dim),
# #     )

# #     head2 = nn.Sequential(
# #         nn.Linear(in_dim, out_dim),
# #         nn.ReLU(inplace=True),
# #         nn.LayerNorm(out_dim),
# #         LightResNormLayer(out_dim, hidden_ratio=head2_hidden_ratio, init_scale=head2_init_scale),
# #     )

# #     return head1, head2
