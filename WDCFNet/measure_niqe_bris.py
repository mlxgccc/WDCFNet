import glob
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque
from loss.niqe_utils import *
import argparse

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
ep = eval_parser.parse_args()


def metrics(im_dir):
    avg_niqe = 0
    n = 0
    avg_brisque = 0

    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1

        im1 = Image.open(item).convert('RGB')
        # score_brisque = brisque.score(im1)
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)


        # avg_brisque += score_brisque
        avg_niqe += score_niqe

        torch.cuda.empty_cache()

    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque

if __name__ == '__main__':

    if ep.DICM:
        im_dir = './results/DICM/*.jpg'

    elif ep.LIME:
        im_dir = './results/LIME/*.bmp'

    elif ep.MEF:
        im_dir = './results/MEF/*.png'

    elif ep.NPE:
        im_dir = './results/NPE/*.jpg'

    elif ep.VV:
        im_dir = './results/VV/*.jpg'


    avg_niqe, avg_brisque = metrics(im_dir)
    print(avg_niqe)
    print(avg_brisque)
