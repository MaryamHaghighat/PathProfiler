import sys
sys.path.extend(["../.", "."])
from common.wsi_reader import get_reader_impl
from common.tile_processing_parallel import process_tiles
import glob
import os
import numpy as np
import argparse
import timeit
import imageio


'''
:param slide_dir: dir to  whole slide images
:param save_folder: a folder to save the extracted tiles
:param tile_magnification: the magnification at which tiles are extracted. The default is 10
:param tile_size: pixel size of the tiles; the default is 256
:param stride: stride; the default is 256
:param mask_dir: dir to mask images
:param mask_magnification: the magnification at which masks are generated. The default is 1.25
:param mask_ratio: minimum acceptable ratio of masked area. The default is 0.2
:param mpp_level_0: True: downsamples tiles from level 0, False: downsamples tiles from the best level
:param ds_level_0: True: downsamples tiles from level 0, False: downsamples tiles from the best level
(closest available level in the slide)
'''

parser = argparse.ArgumentParser(description='Extract tiles from WSIs in parallel')
parser.add_argument('--slide_dir', default='WSIs', help='path to slides dir', type=str, required=True)
parser.add_argument('--slide_id', default='*', type=str, help='slide filename ("*" for all slides)')
parser.add_argument('--mpp_level_0', type=float, default=None, help='manually enter mpp at level 0 if not readable from slides')
parser.add_argument('--tile_size', type=int, default=256)
parser.add_argument('--stride', type=int, default=256)
parser.add_argument('--tile_magnification', type=float, default=10)
parser.add_argument('--mask_dir', default='tissue-masks', help='path to tissue mask folder')
parser.add_argument('--mask_magnification', type=float, default=1.25)
parser.add_argument('--mask_ratio', type=float, default=0.2)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--save_folder', default='tiles', type=str, help='path to the folder to save results')
parser.add_argument('--ds_level_0', default=True, help='True: downsample from level 0, False: downsamples from the best level')
args = parser.parse_args()


def save_tile(tile, x_y, wsi_highest_magnification, save_folder, slide_name, tile_magnification):
    x, y = x_y
    savename = os.path.join(save_folder,
     slide_name + f'_obj_{wsi_highest_magnification}x_{args.tile_magnification}x_x_{x}_y_{y}.png')
    if not os.path.exists(savename):
        imageio.imwrite(savename, np.array(tile))


def main():
    dir_list = glob.glob(os.path.join(args.slide_dir, args.slide_id))
    mpp2mag = {.2: 40, .3: 40, .5: 20, 1: 10}
    for filename in dir_list:
        start = timeit.default_timer()
        basename = os.path.basename(filename)
        slide_name = os.path.splitext(basename)[0]
        reader = get_reader_impl(filename)
        ######### sanity check ################
        try:
            slide = reader(filename)
        except IOError:
            print("skipped {filename} \n error " + str(IOError))
            continue
        mask_path = os.path.join(args.mask_dir, slide_name + '.jpg')
        if not os.path.exists(mask_path):
            print('NO TISSUE MASK FOUND at', mask_path)
            continue
        #######################################
        if args.mpp_level_0:
            print('slides mpp manually set to', args.mpp_level_0)
            mpp = args.mpp_level_0
        else:
            try:
                mpp = slide.mpp[0]
            except:
                print('slide mpp is not available as "slide.mpp"\n use --mpp_level_0 to enter mpp at level 0 manually.')
                continue
        print('Extracting tiles from', basename)
        save_folder = os.path.join(args.save_folder, slide_name, f'{args.tile_magnification}x')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        wsi_highest_magnification = mpp2mag[np.round(mpp, 1)]
        downsample = wsi_highest_magnification / args.tile_magnification
        process_tiles(filename, args.tile_size, args.tile_magnification, args.stride, downsample, args.ds_level_0,
                      mask_path, args.mask_magnification, args.mask_ratio,
                      save_tile, wsi_highest_magnification, save_folder, slide_name, args.tile_magnification,
                      n_workers=args.workers)
        stop = timeit.default_timer()
        print('extracted tiles from %s: %.3f s' % (basename,  stop - start))

    print('Done!')


if __name__ == "__main__":
    main()
