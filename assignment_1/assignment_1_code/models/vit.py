


## TODO implement your own ViT in this file
# You can take any existing code from any repository or blog post - it doesn't have to be a huge model
# specify from where you got the code and integrate it into this code repository so that 
# you can run the model with this code

# Some parts of code are taken from the excercise of another lecture

import math
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.lin = nn.Linear(dim, dim)
        self.dim = dim // self.num_heads
        self.scale = qk_scale or 1. / math.sqrt(self.dim)

    def forward(self, x):
        """
        Forward pass of the MultiHeadAttention layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, C), where
                B is the batch size, N is the number of elements (e.g., sequence length),
                and C is the feature dimension.

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
            torch.Tensor: Attention weights. (For visualization purposes)
        """
        B, N, C = x.shape
        
        # Calculate query, key and value and transpose to correct dimensions
        assert C == self.dim * self.num_heads
        q = self.q(x).view(B,N,self.num_heads, self.dim).transpose(1,2)
        k = self.k(x).view(B,N,self.num_heads, self.dim).transpose(1,2)
        v = self.v(x).view(B,N,self.num_heads, self.dim).transpose(1,2)        
        k_t = k.transpose(-1, -2)
        
        # Get scaled attention scores 
        attn_scores = torch.matmul(q, k_t) * self.scale # (B,H,N,N)
        
        # Apply softmax
        attn_weights = attn_scores.softmax(dim=-1)
        
        # Multiply attention scores with values
        attn = torch.matmul(attn_weights, v) # (B,H,N,D)
        
        # Convert to correct format and pass through a Linear layer
        attn = attn.transpose(1, 2).contiguous().view(B, N, C)
        x = self.lin(attn)
        
        attn = attn_weights
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer
    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        pool="cls",
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = dim

        # Apply patch embed, 3 because of RGB
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, self.embed_dim)
        
        # Randomly initialize class token and positonal embedding
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, self.embed_dim)), requires_grad=True)
        self.pos_embed = nn.Parameter(torch.randn(size=(1, self.patch_embed.num_patches+1, self.embed_dim)), requires_grad=True)

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.pool = pool

        # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(dim)

        # Classifier head
        self.mlp_head = nn.Linear(dim, num_classes)

    def prepare_tokens(self, x):
        """
        Prepares input tokens for a transformer model.

        Args:
            x (torch.Tensor): Input tensor with shape (B, nc, w, h).

        Returns:
            torch.Tensor: Processed tokens with positional encoding and [CLS] token.

        Note:
            Assumes the presence of attributes: patch_embed, cls_token, pos_embed.
        """
        B, nc, w, h = x.shape

        # apply patch linear embedding
        x = self.patch_embed(x) # Returns (B, N, C) where N is the number of patches and C is the embedding dimension
        
        # add the [CLS] token to the embed patch tokens
        temp = self.cls_token.expand(B, -1, -1)
        x = torch.cat((temp, x), dim=1)
        
        # add positional encoding to each token
        x += self.pos_embed

        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.norm(x)
        x = self.mlp_head(x)
        out = torch.sigmoid(x)
        return out
