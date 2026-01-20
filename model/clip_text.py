# model/clip_text.py
"""
Lightweight CLIP text encoder loader (OpenAI CLIP code vendored in model/CLIP).
API:
    clip_model, tokenizer, text_encoder, dtype = load_clip_text_encoder(...)
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
from .CLIP import clip as openai_clip

class CLIPTextEncoder(nn.Module):
    """CoOp-style text encoder wrapper."""
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts_emb: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts_emb + self.positional_embedding.to(prompts_emb.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(prompts_emb.dtype)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection
        return x

def load_clip_text_encoder(
    clip_name: str = "ViT-B/16",
    device: str = "cuda",
    download_root: Optional[str] = None,
    jit: bool = False,
) -> Tuple[nn.Module, Callable[[str], torch.Tensor], CLIPTextEncoder, torch.dtype]:
    clip_model, _ = openai_clip.load(
        clip_name, device=device, jit=jit, download_root=download_root
    )
    tokenizer = openai_clip.tokenize
    text_encoder = CLIPTextEncoder(clip_model)
    dtype = clip_model.dtype
    return clip_model, tokenizer, text_encoder, dtype
