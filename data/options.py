import argparse


def option():
    # Training settings
    parser = argparse.ArgumentParser(description='WDCFNet')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=128, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to Train for end')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=1, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='wLearning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False)
    parser.add_argument('--cos_restart', type=bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to Train without warmup')

    # Train datasets
    parser.add_argument('--data_train_lol_blur', type=str, default='./datasets/LOL_blur/Train')
    parser.add_argument('--data_train_llvip', type=str, default='./datasets/LLVIP/Train')
    parser.add_argument('--data_train_lol_v1', type=str, default='./datasets/LOLv1/Train')
    parser.add_argument('--data_train_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID', type=str, default='./datasets/Sony_total_dark/Train')
    parser.add_argument('--data_train_SICE', type=str, default='./datasets/SICE/Dataset/Train')
    parser.add_argument('--data_train_huawei', type=str, default='./datasets/Huawei/Train')
    parser.add_argument('--data_train_nikon', type=str, default='./datasets/Nikon/Train')


    # validation low
    parser.add_argument('--data_val_lol_blur', type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_llvip', type=str, default='./datasets/LLVIP/Test/low')
    parser.add_argument('--data_val_lol_v1', type=str, default='./datasets/LOLv1/Test/low')
    parser.add_argument('--data_val_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Test/input')
    parser.add_argument('--data_val_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Test/input')
    parser.add_argument('--data_val_SID', type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix', type=str, default='./datasets/SICE/Dataset/eval/Test')
    parser.add_argument('--data_val_SICE_grad', type=str, default='./datasets/Huawei/Test/low')
    parser.add_argument('--data_val_huawei_grad', type=str, default='./datasets/Huawei/Test/low')
    parser.add_argument('--data_val_nikon_grad', type=str, default='./datasets/Nikon/Test/low')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur', type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_llvip', type=str, default='./datasets/LLVIP/Test/high/')
    parser.add_argument('--data_valgt_lol_v1', type=str, default='./datasets/LOLv1/Test/high/')
    parser.add_argument('--data_valgt_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Test/groundtruth/')
    parser.add_argument('--data_valgt_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Test/groundtruth/')
    parser.add_argument('--data_valgt_SID', type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix', type=str, default='./datasets/SICE/Dataset/eval/high/')
    parser.add_argument('--data_valgt_SICE_grad', type=str, default='./datasets/SICE/Dataset/eval/high/')
    parser.add_argument('--data_valgt_huawei', type=str, default='./datasets/Huawei/Test/high/')
    parser.add_argument('--data_valgt_nikon', type=str, default='./datasets/Nikon/Test/high/')


    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1)
    parser.add_argument('--D_weight', type=float, default=0.5)
    parser.add_argument('--E_weight', type=float, default=50.0)
    parser.add_argument('--P_weight', type=float, default=4e-3)

    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=50)
    parser.add_argument('--end_gamma', type=int, default=180)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=bool, default=True, help='if gradient fluctuates too much, turn-on it')

    # choose which dataset you want to Train, please only set one "True"
    parser.add_argument('--llvip', type=bool, default=False)
    parser.add_argument('--lol_v1', type=bool, default=True)
    parser.add_argument('--lolv2_real', type=bool, default=False)
    parser.add_argument('--lolv2_syn', type=bool, default=False)
    parser.add_argument('--lol_blur', type=bool, default=False)
    parser.add_argument('--SID', type=bool, default=False)
    parser.add_argument('--SICE_mix', type=bool, default=False)
    parser.add_argument('--SICE_grad', type=bool, default=False)
    parser.add_argument('--huawei', type=bool, default=False)
    parser.add_argument('--nikon', type=bool, default=False)

    return parser
