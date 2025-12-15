from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *

def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0),
        ToTensor(),
    ])

def transform2():
    return Compose([ToTensor()])



def get_lol_training_set(data_dir,size):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))
def get_lol_eval_set(data_dir):
    return LOLDatasetFromFolder(data_dir, transform=transform2())

def get_lol_v2_training_set(data_dir,size):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir,size):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_huawei_training_set(data_dir,size):
    return huaweiDatasetFromFolder(data_dir, transform=transform1(size))

def get_huawei_eval_set(data_dir):
    return huaweiDatasetFromFolder(data_dir, transform=transform2())

def get_nikon_training_set(data_dir,size):
    return NikonDatasetFromFolder(data_dir, transform=transform1(size))

def get_nikon_eval_set(data_dir):
    return NikonDatasetFromFolder(data_dir, transform=transform2())