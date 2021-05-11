import os
import argparse
import imageio
import numpy as np
import time
import cv2
from wsi_reader import get_reader_impl
import glob

parser = argparse.ArgumentParser(description='Extract tiles from WSIs')

parser.add_argument('--slide_dir', default='/well/rittscher/projects/ProMPT_cases/selected_images_by_lisa/selected_from_IHC_study', type=str)
parser.add_argument('--slide_id', default='*', type=str, help='slide filename (or "*" for all slides)')
parser.add_argument('--save_folder', type=str, default='../tiles', help='folder ')
parser.add_argument('--tile_magnification', type=float, default=10)
parser.add_argument('--mask_magnification', type=float, default=2.5)
parser.add_argument('--tile_size', type=int, default=256)
parser.add_argument('--stride', type=int, default=256)
parser.add_argument('--mask_dir', default='path to tissue mask folder')
parser.add_argument('--mask_ratio', type=float, default=0.2)

args = parser.parse_args()


def tiling(wsi_filename,
               save_folder,
               tile_magnification,
               tile_size,
               stride,
               mask_dir,
               mask_magnification,
               mask_ratio):
    '''
    :param slide_dir: dir to  whole slide images
    :param save_folder: a folder to save the extracted tiles
    :param tile_magnification: the magnification at which tiles are extracted. The default is 10
    :param tile_size: pixel size of the tiles; the default is 256
    :param stride: stride; the default is 256
    :param mask_dir: dir to tissue mask images
    :param mask_magnification: the magnification at which masks are generated. The default is 2.5
    :param mask_ratio: minimum acceptable ratio of masked area. The default is 0.2
    :return:
    '''

    t = time.time()

    #############################################################
    # sanity check
    assert os.path.exists(wsi_filename), "%s doesn't exist" % wsi_filename
    assert tile_magnification in [40, 20, 10, 5, 2.5, 1.25], "tile_magnification should either be 40, 20, 10, 5, 2.5, 1.25"
    #############################################################

    # create a folder to save tile
    slideID = os.path.splitext(os.path.basename(wsi_filename))[0]
    save_location = os.path.join(save_folder, slideID)
    mask_filename = os.path.join(mask_dir, slideID + '.jpg')

    #############################################################
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    #############################################################
    # read mask
    if mask_dir:
        mask = cv2.imread(mask_filename, -1)/255
        mask_downsample = int(tile_magnification / mask_magnification)
        mask_tile_size = int(tile_size / mask_downsample)
    #############################################################
    # read slide
    mpp2mag = {.2: 40, .3: 40, .5: 20, 1: 10}
    reader = get_reader_impl(wsi_filename)
    slide = reader(wsi_filename)

    if slide.mpp is None:
        mpp = .2
    else:
        mpp = slide.mpp[0]

    wsi_highest_magnification = mpp2mag[np.round(mpp, 1)]

    #############################################################

    level = int(np.log2(wsi_highest_magnification / tile_magnification))
    ###########################################################################################
    ncol, nrow = slide.level_dimensions[level]

    for x in range(0, ncol, stride):
        for y in range(0, nrow, stride):

            if mask_dir:
                crop_mask = mask[int(y / mask_downsample): mask_tile_size + int(y / mask_downsample),
                int(x / mask_downsample): mask_tile_size + int(x / mask_downsample)]
                if crop_mask.mean() < mask_ratio:
                    continue

            savename = os.path.join(save_location, slideID + f'_obj_{wsi_highest_magnification}x_{tile_magnification}x_x_{x}_y_{y}.jpg')
            if not os.path.exists(savename):
                try:
                    crop_img, _ = slide.read_region((x, y), level, (tile_size, tile_size), normalize=False)
                except:
                    slide = reader(wsi_filename)
                    continue
                crop_img = np.array(crop_img)
                imageio.imwrite(savename, crop_img)

    print('extracted tiles from %s: %.3f s' % (slideID, time.time() - t))


if __name__ == "__main__":

    wsi_filenames = glob.glob(os.path.join(args.slide_dir, args.slide_id))

    for filename in wsi_filenames:
        tiling(wsi_filename=filename,
               save_folder=args.save_folder,
               tile_magnification=args.tile_magnification,
               tile_size=args.tile_size,
               stride=args.stride,
               mask_dir=args.mask_dir,
               mask_magnification=args.mask_magnification,
               mask_ratio=args.mask_ratio
               )
