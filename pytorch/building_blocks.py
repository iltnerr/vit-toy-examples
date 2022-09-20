import torch
import torch.nn.functional as F

from torch import nn, Tensor

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):

        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # Use conv layer to directly extract patches and to linearly project them to the hidden dimension (latent query & key vector space)
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))


    def forward(self, x: Tensor) -> Tensor:

        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)

        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    """
    PyTorch already has a nn.MultiheadAttention implementation.
    This class serves as an example for self-attention in multiple "heads".
    """
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3) # fuse the queries, keys and values in one matrix
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:

        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # sum up products of query and key vectors over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # attention weights
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # sum up products of attention weights and value vectors over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        # linear embedding
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    """
    Residual connection between layers.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    """
    2-layer MLP with GELU (Gaussian Error Linear Units) activation function.
    Subclass nn.Sequential to avoid writing the forward method for each layer.
    """
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """
    Encoder Block of a Transformer composed by a Multi-Head Self-Attention layer, an MLP layer, and additionally Residual Connections and Normalization Layers.
    """
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            # MHSA
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            # MLP
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    """
    In ViT only the Encoder part of the original transformer is used. The encoder is L blocks of TransformerBlock.
    """
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    """
    The last layer is a normal fully connected layer that gives the class probability. It first performs a basic mean over the whole sequence.
    """
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)) # Softmax/Sigmoid activation missing? nn.Linear layers do not output probabilities, but logits. Maybe it is used at some other point.


class ViT(nn.Sequential):
    """
    Vision Transformer Model:
    Input Image -> Patch Extraction & Linear Embedding -> Multi-Head Attention & MLP -> Classification Head
    """
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )