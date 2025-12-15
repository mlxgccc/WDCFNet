from einops import rearrange
from net.transformer_utils import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
def rotate_every_two(x):
    """旋转位置编码辅助函数"""
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    """应用旋转位置编码"""
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Module):
    """深度可分离卷积"""

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''x: (b h w c)'''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class RotaryPositionEmbedding2D(nn.Module):
    """2D旋转位置编码"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.num_heads = num_heads
        self.register_buffer('angle', angle)

    def forward(self, slen):
        '''slen: (h, w)'''
        index = torch.arange(slen[0] * slen[1]).to(self.angle)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        sin = sin.reshape(slen[0], slen[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(slen[0], slen[1], -1)
        return (sin, cos)


class PathEmbeding2D(nn.Module):
    """路径嵌入模块 - 生成结构化掩码"""

    def __init__(self, embed_dim, nheads, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 A_init_range=(1, 1.1), bias=False):
        super().__init__()
        self.nheads = nheads
        self.dt_linear = nn.Linear(int(embed_dim // nheads), 2, bias=bias)

        # 初始化 dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A 参数
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

    def generate_structed_mask_1d(self, x: torch.Tensor):
        chunk_size = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum_tril = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
        x_segsum_tril = x_segsum_tril.masked_fill(~mask, 0.0)

        x_segsum_triu = x_cumsum[..., None, :] - x_cumsum[..., :, None]
        mask = torch.triu(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=1)
        x_segsum_triu = x_segsum_triu.masked_fill(~mask, 0.0)

        x_segsum = x_segsum_tril + x_segsum_triu
        return x_segsum

    def forward(self, x: torch.Tensor):
        """x: (b h w c)"""
        batch, height, width, dim = x.shape
        seqlen = height * width
        x = x.view(batch, seqlen, dim)
        headdim = int(dim // self.nheads)
        x = x.view(batch, seqlen, self.nheads, headdim)

        dt = self.dt_linear(x)
        dt_alpha, dt_beta = dt[:, :, :, 0], dt[:, :, :, 1]

        A = -torch.exp(self.A_log)
        dt_alpha = F.softplus(dt_alpha + self.dt_bias) * A
        dt_beta = F.softplus(dt_beta + self.dt_bias) * A

        dt_alpha = dt_alpha.view(batch, height, width, self.nheads).contiguous()
        dt_alpha = dt_alpha.permute(0, 1, 3, 2).contiguous()
        structed_mask_w = self.generate_structed_mask_1d(dt_alpha)

        dt_beta = dt_beta.view(batch, height, width, self.nheads).contiguous()
        dt_beta = dt_beta.permute(0, 2, 1, 3).contiguous()
        dt_beta = dt_beta.permute(0, 1, 3, 2).contiguous()
        structed_mask_h = self.generate_structed_mask_1d(dt_beta)

        return structed_mask_w, structed_mask_h


class DBCAFM(nn.Module):


    def __init__(self, dim, num_heads=4, bias=True, use_pos_enc=True, value_factor=1):
        super(DBCAFM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_pos_enc = use_pos_enc
        self.factor = value_factor
        self.head_dim = dim // num_heads
        self.key_dim = dim // num_heads
        self.scaling = self.key_dim ** -0.5

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # 位置编码
        if use_pos_enc:
            self.rope = RotaryPositionEmbedding2D(dim, num_heads)

        # 动态权重融合模块 (保留原始CAB的优点)
        self.dynamic_weight = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 2, kernel_size=1, bias=bias),
            nn.Softmax(dim=1)
        )

        # Q, K, V 投影 (PPMA风格)
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim * self.factor, bias=bias)

        # 局部位置编码增强 (LEPE)
        self.lepe = DWConv2d(dim, 5, 1, 2)

        # 输出投影
        self.out_proj = nn.Linear(dim * self.factor, dim, bias=bias)

        # 路径嵌入 - 生成结构化注意力掩码
        self.path_embed = PathEmbeding2D(dim, num_heads)

        # 归一化和FFN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim, dim * 4),
        #     nn.GELU(),
        #     nn.Linear(dim * 4, dim)
        # )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, y, th):
        """
        x: query特征 [B, C, H, W]
        y: key/value特征1 [B, C, H, W]
        th: key/value特征2 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 动态加权融合 y 和 th (保留原CAB的融合策略)
        fusion_input = torch.cat([y, th], dim=1)
        weights = self.dynamic_weight(fusion_input)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        fused_kv = y * w1 + th * w2  # [B, C, H, W]

        # 转换为 (B, H, W, C) 格式供PPMA处理
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        fused_kv_bhwc = fused_kv.permute(0, 2, 3, 1).contiguous()

        # 生成旋转位置编码
        if self.use_pos_enc:
            sin, cos = self.rope((H, W))
        else:
            sin = cos = None

        # Q, K, V 投影
        q = self.q_proj(x_bhwc)  # (B, H, W, C)
        k = self.k_proj(fused_kv_bhwc)
        v = self.v_proj(fused_kv_bhwc)

        # LEPE增强
        lepe = self.lepe(v)

        # 缩放
        k = k * self.scaling

        # 重塑为多头格式
        q = q.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (B, heads, H, W, dim)
        k = k.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        # 应用旋转位置编码
        if self.use_pos_enc:
            qr = theta_shift(q, sin, cos)
            kr = theta_shift(k, sin, cos)
        else:
            qr, kr = q, k

        # 生成路径结构化掩码
        structed_mask_w, structed_mask_h = self.path_embed(x_bhwc)

        # 水平方向注意力
        qr_w = qr.transpose(1, 2)  # (B, H, heads, W, dim)
        kr_w = kr.transpose(1, 2)
        v_w = v.transpose(1, 2)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (B, H, heads, W, W)
        qk_mat_w = qk_mat_w + structed_mask_w
        qk_mat_w = torch.softmax(qk_mat_w, -1)

        # 垂直方向注意力
        qr_h = qr.permute(0, 3, 1, 2, 4)  # (B, W, heads, H, dim)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v_h = v.permute(0, 3, 1, 2, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (B, W, heads, H, H)
        qk_mat_h = qk_mat_h + structed_mask_h
        qk_mat_h = torch.softmax(qk_mat_h, -1)

        # 双路径注意力计算
        v_w = torch.matmul(qk_mat_w, v_w)  # (B, H, heads, W, dim)
        v_w = v_w.permute(0, 3, 2, 1, 4)  # (B, W, heads, H, dim)
        output1 = torch.matmul(qk_mat_h, v_w)  # (B, W, heads, H, dim)
        output1 = output1.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (B, H, W, C)

        v_h = torch.matmul(qk_mat_h, v_h)  # (B, W, heads, H, dim)
        v_h = v_h.permute(0, 3, 2, 1, 4)  # (B, H, heads, W, dim)
        output2 = torch.matmul(qk_mat_w, v_h)  # (B, H, heads, W, dim)
        output2 = output2.permute(0, 1, 3, 2, 4).flatten(-2, -1)  # (B, H, W, C)

        # 双路径融合
        output = 0.5 * output1 + 0.5 * output2

        # 添加LEPE
        output = output + lepe

        # 输出投影
        output = self.out_proj(output)

        # 第一次残差 + LayerNorm
        output = self.norm1(x_bhwc + output)

        # FFN + 第二次残差 + LayerNorm
        ffn_out = self.ffn(output)
        output = self.norm2(output + ffn_out)

        # 转回 (B, C, H, W)
        output = output.permute(0, 3, 1, 2).contiguous()

        return output



# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = DBCAFM(dim, num_heads, bias)

    def forward(self, x, y, th):
        x = x + self.ffn(self.norm(x), self.norm(y), self.norm(th))
        x = self.gdfn(self.norm(x))
        return x


class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = DBCAFM(dim, num_heads, bias=bias)

    def forward(self, x, y, th):
        x = x + self.ffn(self.norm(x), self.norm(y), self.norm(th))
        x = x + self.gdfn(self.norm(x))
        return x
