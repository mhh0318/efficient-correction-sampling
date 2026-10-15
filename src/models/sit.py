"""Adapted from willisma/SiT, cbde832a40b153ccc79603412409da9c9b0c568c.
Label dropout, prediction conversion and CFG belong to the common denoiser.
"""
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from .layers import TimestepEmbedder, LabelEmbedder, get_2d_sincos_pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, d//self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))


class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size*mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6*hidden_size))

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate1[:, None]*self.attn(modulate(self.norm1(x), shift1, scale1))
        return x + gate2[:, None]*self.mlp(modulate(self.norm2(x), shift2, scale2))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**2*channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2*hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152,
                 depth=28, num_heads=16, mlp_ratio=4., num_classes=1000):
        super().__init__()
        if input_size % patch_size:
            raise ValueError('State size must be divisible by patch size')
        self.in_channels, self.input_size, self.patch_size = in_channels, input_size, patch_size
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_size//patch_size)**2, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.input_size//self.patch_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        w = self.x_embedder.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=.02)
        for i in (0, 2): nn.init.normal_(self.t_embedder.mlp[i].weight, std=.02)
        for block in [*self.blocks, self.final_layer]:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x, t, labels):
        if x.shape[1:] != (self.in_channels, self.input_size, self.input_size):
            raise ValueError('Input shape differs from model configuration')
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(labels)
        for block in self.blocks: x = block(x, c)
        x = self.final_layer(x, c)
        p, h = self.patch_size, self.input_size//self.patch_size
        x = x.reshape(x.shape[0], h, h, p, p, self.in_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], self.in_channels, h*p, h*p)


SiT_models = {
    f'SiT-{name}/{patch}': partial(SiT, depth=depth, hidden_size=width, num_heads=heads, patch_size=patch)
    for name, depth, width, heads in [('S', 12, 384, 6), ('B', 12, 768, 12), ('L', 24, 1024, 16), ('XL', 28, 1152, 16)]
    for patch in (2, 4, 8)
}
