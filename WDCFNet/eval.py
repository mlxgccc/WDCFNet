import os

from torch.cuda import device

from net.DWT import dwt2_gpu

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.WDCFNet import WDCFNet

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--llvip', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')
eval_parser.add_argument('--huawei', action='store_true', help='output Huawei dataset')
eval_parser.add_argument('--nikon', action='store_true', help='output Nikon dataset')

eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/HuaweiDICM.pth')

ep = eval_parser.parse_args()
# def eval(model, testing_data_loader, model_path, output_folder,
#          norm_size=True, LOL=False, v2=False, huawei=False,
#          unpaired=False, alpha=1.0, gamma=1.0):
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.set_grad_enabled(False)
#
#     # Load pretrained weights
#     model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
#     print('Pre-trained model is loaded.')
#     model.eval()
#     print('Evaluation:')
#
#     # Switches for different dataset modes
#     if LOL:
#         model.trans.gated = True
#     elif v2:
#         model.trans.gated2 = True
#         model.trans.alpha = alpha
#     elif huawei:
#         model.trans.gated2 = True
#     elif unpaired:
#         model.trans.gated2 = True
#         model.trans.alpha = alpha
#
#     for batch in tqdm(testing_data_loader):
#         with torch.no_grad():
#             # Parse inputs
#             if norm_size:
#                 input, name = batch[0], batch[1]
#             else:
#                 input, name, h, w = batch[0], batch[1], batch[2], batch[3]
#
#             # 1. Send input to GPU
#             input = input.to(device)
#
#             # 2. Extract wavelet features on GPU
#             freq_features = dwt2_gpu(input)
#
#             # 3. Forward pass on GPU
#             output = model((input ** gamma, freq_features[0], freq_features[1]))
#
#         # Make output folder
#         if not os.path.exists(output_folder):
#             os.mkdir(output_folder)
#
#         # Clamp and crop
#         output = torch.clamp(output.cuda(), 0, 1).cuda()
#         if not norm_size:
#             output = output[:, :, :h, :w]
#
#         # Save result
#         output_img = transforms.ToPILImage()(output.squeeze(0))
#         output_img.save(output_folder + name[0])
#
#         torch.cuda.empty_cache()
#
#     print('===> End evaluation')
#
#     # Reset flags
#     if LOL:
#         model.trans.gated = False
#     elif v2:
#         model.trans.gated2 = False
#
#     torch.set_grad_enabled(True)

def split_image(img_tensor, tile_size=256, overlap=32):
    """将大图切块，返回 tiles 和它们在原图中的坐标"""
    _, C, H, W = img_tensor.shape
    stride = tile_size - overlap
    tiles = []
    positions = []
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            h_end = min(i + tile_size, H)
            w_end = min(j + tile_size, W)
            tile = img_tensor[:, :, i:h_end, j:w_end]
            tiles.append(tile)
            positions.append((i, h_end, j, w_end))
    return tiles, positions


def merge_tiles(tiles, positions, full_size):
    """将 tiles 拼接回原图大小，overlap 区域平均融合"""
    _, C, H, W = full_size
    canvas = torch.zeros((1, C, H, W), device=tiles[0].device)
    weight = torch.zeros((1, C, H, W), device=tiles[0].device)
    for tile, (i0, i1, j0, j1) in zip(tiles, positions):
        canvas[:, :, i0:i1, j0:j1] += tile
        weight[:, :, i0:i1, j0:j1] += 1.0
    canvas /= weight
    return canvas


def eval(model, testing_data_loader, model_path, output_folder,
         norm_size=True, LOL=False, v2=False, huawei=False, unpaired=False,
         alpha=1.0, gamma=1.0, tile_size=512, overlap=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')

    # 设置模型开关
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif huawei:
        model.trans.gated2 = True
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
                _, _, H, W = input.shape
            else:
                input, name, H, W = batch[0], batch[1], batch[2], batch[3]

            input = input.to(device)

            # 大图分块推理
            tiles, positions = split_image(input, tile_size=tile_size, overlap=overlap)
            output_tiles = []
            for tile in tiles:
                # 小波特征
                freq_features = dwt2_gpu(tile)
                # 模型前向传播
                tile_out = model((tile ** gamma, freq_features[0], freq_features[1]))
                tile_out = torch.clamp(tile_out, 0, 1)
                output_tiles.append(tile_out)

            # 拼接回原图
            output = merge_tiles(output_tiles, positions, input.shape)

            if not norm_size:
                output = output[:, :, :H, :W]

            output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
            output_img.save(os.path.join(output_folder, name[0]))

            torch.cuda.empty_cache()

    print('===> End evaluation')

    # 复位模型开关
    if LOL:
        model.trans.gated = False
    elif v2 or huawei or unpaired:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)

if __name__ == '__main__':

    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")

    if not os.path.exists('./output'):
            os.mkdir('./output')

    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set(r".\datasets\LOLv1\Test\low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = r'.\results\LOLv1/'
        if ep.perc:
            weight_path = r'.\weights\LOL.pth'
        else:
            weight_path = r'.\weights\LOL.pth'

    elif ep.llvip:
        eval_data = DataLoader(dataset=get_eval_set(r".\datasets\LOLv1\Test\low"), num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = '.\output/'
        weight_path = r'.\weights\train\epoch.pth'

    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/input"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = r'.\weights\lol_v2_real.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = r'.\weights\lol_v2_real.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = r'.\weights\lol_v2_real.pth'
            alpha = 0.82

    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './results/LOLv2_syn/'
        if ep.perc:
            weight_path = r'.\weights\lol_v2_syn.pth'
        else:
            weight_path = r'.\weights\lol_v2_syn.pth'

    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = r'.\weights\epoch.pth'
        norm_size = False

    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = r'.\weights\epoch.pth'
        norm_size = False

    elif ep.huawei:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/Huawei/test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './results/huawei/'
        weight_path = r'.\weights\huawei.pth'
    elif ep.nikon:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/Nikon/test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './results/nikon/'
        weight_path = r'.\weights\huawei.pth'

    elif ep.DICM:
        eval_data = DataLoader(dataset=get_SICE_eval_set(r".\dataset\DICM\DICM"),
                               num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = './results/DICM/'
        weight_path = r'.\weights\DICM.pth'

    elif ep.LIME:
        eval_data = DataLoader(dataset=get_SICE_eval_set(r".\dataset\LIME\LIME"),
                               num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = './results/LIME/'
        weight_path = r'.\weights\LIME.pth'

    elif ep.MEF:
        eval_data = DataLoader(dataset=get_SICE_eval_set(r".\dataset\MEF\MEF"),
                               num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = './results/MEF/'
        weight_path = r'.\weights\Huaweimef8.pth'
    elif ep.NPE:
        eval_data = DataLoader(dataset=get_SICE_eval_set(r".\dataset\NPE"),
                               num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = './results/NPE/'
        weight_path = r'.\weights\NPE.pth'
    elif ep.VV:
        eval_data = DataLoader(dataset=get_SICE_eval_set(r".\dataset\VV\VV"),
                               num_workers=num_workers,
                               batch_size=1, shuffle=False)
        output_folder = './results/VV/'
        weight_path = r'.\weights\HuaweiVV.pth'


    eval_net = WDCFNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,huawei=ep.huawei,unpaired=ep.unpaired,alpha=alpha,gamma=ep.gamma)

