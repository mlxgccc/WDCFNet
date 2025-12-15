import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as T
import os
from glob import glob

def save_img(tensor, name, folder):
    os.makedirs(folder, exist_ok=True)
    x = tensor[0].detach().cpu()  # [B,C,H,W] -> [C,H,W]

    # 归一化到0-1
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    vutils.save_image(x, os.path.join(folder, f"{name}.png"))
    print(f"✅ Saved: {os.path.join(folder, f'{name}.png')}")

def haar_dwt(img_tensor, save=True, folder="wavelet_output", prefix="img"):
    """
    输入: img_tensor = [B,3,H,W] (RGB)
    输出: LL, LH, HL, HH (均为 [B,3,H/2,W/2])
    """
    B, C, H, W = img_tensor.shape
    assert H % 2 == 0 and W % 2 == 0, "Height/Width must be even."

    # Haar 小波核
    ll = torch.tensor([[0.5, 0.5],[0.5, 0.5]], dtype=img_tensor.dtype, device=img_tensor.device).view(1,1,2,2)
    lh = torch.tensor([[-0.5, -0.5],[ 0.5,  0.5]], dtype=img_tensor.dtype, device=img_tensor.device).view(1,1,2,2)
    hl = torch.tensor([[-0.5,  0.5],[-0.5,  0.5]], dtype=img_tensor.dtype, device=img_tensor.device).view(1,1,2,2)
    hh = torch.tensor([[ 0.5, -0.5],[-0.5,  0.5]], dtype=img_tensor.dtype, device=img_tensor.device).view(1,1,2,2)

    # 对每个通道分别卷积
    LL = F.conv2d(img_tensor, ll.repeat(C,1,1,1), stride=2, groups=C)
    LH = F.conv2d(img_tensor, lh.repeat(C,1,1,1), stride=2, groups=C)
    HL = F.conv2d(img_tensor, hl.repeat(C,1,1,1), stride=2, groups=C)
    HH = F.conv2d(img_tensor, hh.repeat(C,1,1,1), stride=2, groups=C)

    if save:
        save_img(LL, f"{prefix}_LL", folder)
        save_img(LH, f"{prefix}_LH", folder)
        save_img(HL, f"{prefix}_HL", folder)
        save_img(HH, f"{prefix}_HH", folder)

    return LL, LH, HL, HH

def wavelet_from_dir(img_dir, output_folder="wavelet_output"):
    os.makedirs(output_folder, exist_ok=True)
    img_paths = glob(os.path.join(img_dir, "*.*"))  # 读取所有图片文件
    transform = T.ToTensor()

    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).cuda()  # [1,3,H,W]
        haar_dwt(tensor, save=True, folder=output_folder, prefix=img_name)

# 使用方法
wavelet_from_dir("E:\LKY\lol", output_folder="wavelet_output")
