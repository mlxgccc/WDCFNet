import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import os

from torch import nn


def dwt2_gpu(x, save=False, folder="wavelet_output", prefix="img"):

    B, C, H, W = x.shape

    # Haar 小波核
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

    # 卷积得到四个子带
    LL = F.conv2d(x, ll.repeat(C, 1, 1, 1), stride=2, groups=C)
    LH = F.conv2d(x, lh.repeat(C, 1, 1, 1), stride=2, groups=C)
    HL = F.conv2d(x, hl.repeat(C, 1, 1, 1), stride=2, groups=C)
    HH = F.conv2d(x, hh.repeat(C, 1, 1, 1), stride=2, groups=C)

    # 上采样 LL 到原始尺寸
    LL_up = F.interpolate(LL, size=(H, W), mode='bilinear', align_corners=False)
    alpha, beta, gamma = 0.33, 0.33, 0.34
    HF = alpha * LH + beta * HL + gamma * HH
    HF_fused = F.interpolate(HF, size=(H, W), mode='bilinear', align_corners=False)



    # 内置保存功能
    if save:
        os.makedirs(folder, exist_ok=True)
        for subband, name in zip([LL, LH, HL, HH], ["LL", "LH", "HL", "HH"]):
            img = subband[0].detach().cpu()  # [B,C,H/2,W/2] -> [C,H/2,W/2]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 归一化
            vutils.save_image(img, os.path.join(folder, f"{prefix}_{name}.png"))
            print(f"✅ Saved: {os.path.join(folder, f'{prefix}_{name}.png')}")

        # 保存上采样后的 LL
        img_up = LL_up[0].detach().cpu()
        img_up = (img_up - img_up.min()) / (img_up.max() - img_up.min() + 1e-8)
        vutils.save_image(img_up, os.path.join(folder, f"{prefix}_LL_up.png"))
        print(f"✅ Saved: {os.path.join(folder, f'{prefix}_LL_up.png')}")
        imghigh_up = HF_fused[0].detach().cpu()
        imghigh_up = (imghigh_up - imghigh_up.min()) / (imghigh_up.max() - imghigh_up.min() + 1e-8)
        vutils.save_image(imghigh_up, os.path.join(folder, f"{prefix}_imghigh_up.png"))
        print(f"✅ Saved: {os.path.join(folder, f'{prefix}_imghigh_up.png')}")
    return LL_up, HF_fused
