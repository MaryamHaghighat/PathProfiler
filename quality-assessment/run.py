import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tile_processing_parallel import process_tiles
import argparse
import cv2
import csv
import timeit
import os.path as path
from tempfile import mkdtemp
from torchvision import transforms
import torch
from models_embedding import ResNet18
from wsi_reader import get_reader_impl
import math

'''
Tile-level quality assessment of whole slides using pretrained model: "checkpoint_106.pth"
output: saves quality overlays and the processed region in a .npy file as:
slide_info = {'focus_artfcts': focus_artfcts, 'stain_artfcts': stain_artfcts,
'normal': normal, 'other_artfcts': other_artfcts, 'folding_artfcts': folding_artfcts,
'usblty': usblty, 'processed_region': processed_region}

Note: The pixel size of quality overlays is slide_size_at_5X / tile_size
Heatmaps can be saved at a given magnification
'''

parser = argparse.ArgumentParser()
parser.add_argument('--slide_dir', default='path to slide dir', type=str)
parser.add_argument('--slide_id', default='*', type=str, help='slide filename ("*" for all slides)')
parser.add_argument('--mpp_level0', type=float, default=None, help='manually enter mpp at level 0 if not readable from slides')
parser.add_argument('--mask_dir', default='../tissue-masks', help='path to tissue mask folder')
parser.add_argument('--tile_size', type=int, default=256, help='our model is trained with images of size 256)
parser.add_argument('--mask_magnification', type=float, default=2.5)
parser.add_argument('--mask_ratio', type=float, default=0.2)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--model', default='checkpoint_106.pth', type=str)
parser.add_argument('--save_folder', default='quality-overlays', type=str, help='path to the folder to save results')
args = parser.parse_args()

      
def eval_quality(tile, x_y, QA_model, usblty, normal, focus_artfcts, stain_artfcts, other_artfcts, folding_artfcts,
                 processed_region):
    ncol, nrow = x_y
    ncol, nrow = int(ncol/tile.shape[1]), int(nrow/tile.shape[0])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    totensor = transforms.ToTensor()
    tile = cv2.resize(np.array(tile), (224, 224), interpolation=cv2.INTER_CUBIC)
    tile = normalize(totensor(tile)).type(torch.FloatTensor)
    with torch.no_grad():
        QA_model.eval()
    out = QA_model.embedding(tile.unsqueeze(0)).squeeze(0).cpu()
    out[out > 1] = 1
    out[out < 0] = 0
    out = out.data.numpy()
    usblty[nrow, ncol] = out[0]
    normal[nrow, ncol] = out[1]
    stain_artfcts[nrow, ncol] = out[2]
    focus_artfcts[nrow, ncol] = out[3]
    folding_artfcts[nrow, ncol] = out[4]
    other_artfcts[nrow, ncol] = out[5]
    processed_region[nrow, ncol] = 1


def generate_heatmap(overlay, processed_region, heatmap_mag):
    cmap = cm.get_cmap('bwr')
    heatmap = cmap(overlay)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap[processed_region==0]=128
    heatmap=heatmap.repeat(int(heatmap_mag*256/5.), axis=0).repeat(int(heatmap_mag*256/5.), axis=1)
    return heatmap


def main():
    # define network
    QA_model = ResNet18()
    print("=> loading checkpoint '{}'".format(args.model))
    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    QA_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.model, checkpoint['epoch']))
    if not path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    dir_list = glob.glob(path.join(args.slide_dir, args.slide_id))

    output_dtype = 'float16'
    tmp_focus_artfcts_file = path.join(mkdtemp(), 'tmp_focus_artfcts_file.csv')
    tmp_stain_artfcts_file = path.join(mkdtemp(), 'tmp_stain_artfcts_file.csv')
    tmp_normal_file = path.join(mkdtemp(), 'tmp_normal_file.csv')
    tmp_other_artfcts_file = path.join(mkdtemp(), 'tmp_other_artfcts_file.csv')
    tmp_folding_artfcts_file = path.join(mkdtemp(), 'tmp_folding_artfcts_file.csv')
    tmp_usblty_file = path.join(mkdtemp(), 'tmp_usblty_file.csv')
    tmp_processed_region_file = path.join(mkdtemp(), 'tmp_processed_region_file.csv')

    mpp2mag = {.2: 40, .3: 40, .5: 20, 1: 10}
    for filename in dir_list:
        start = timeit.default_timer()
        basename = path.basename(filename)
        print(basename)
        slide_name = path.splitext(basename)[0]
        reader = get_reader_impl(filename)
        if path.exists(path.join(args.save_folder,  slide_name + '_quality_overlays.npy')):
            continue
        ############### sanity check #########
        try:
            slide = reader(filename)
        except IOError:
            print("error " + str(IOError))
            with open('slides_IOError.csv', 'a+', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([filename])
            continue
        mask_path = path.join(args.mask_dir, slide_name + '.jpg')
        if not path.exists(mask_path):
            print('NO TISSUE MASK FOUND')
            continue
        #######################################
        if args.mpp_level0:
            print('slides mpp manually set to', args.mpp_level0)
            mpp = args.mpp_level0
        else:
            try:
                mpp = slide.mpp['MPP']
            except:
                print('slide mpp is not available as "slide.mpp"\n use --mpp_level0 to enter mpp at level 0 manually.')
                continue

        wsi_highest_magnification = mpp2mag[np.round(mpp, 1)]
        level = round(np.log2(wsi_highest_magnification / 5.))  # Model is trained for tiles at 5X
        w, h = int(math.ceil(slide.level_dimensions[0][0]/2**level)), int(math.ceil(slide.level_dimensions[0][1]/2**level))
        w_map, h_map = int(np.round(w/args.tile_size)), int(np.round(h/args.tile_size))
        focus_artfcts = np.memmap(tmp_focus_artfcts_file, dtype=output_dtype, mode='w+', shape=(h_map, w_map))
        stain_artfcts = np.memmap(tmp_stain_artfcts_file, dtype=output_dtype, mode='w+', shape=(h_map, w_map))
        normal = np.memmap(tmp_normal_file, dtype=output_dtype, mode='w+', shape=(h_map, w_map))
        other_artfcts = np.memmap(tmp_other_artfcts_file, dtype=output_dtype, mode='w+', shape=(h_map, w_map))
        folding_artfcts = np.memmap(tmp_folding_artfcts_file, dtype=output_dtype, mode='w+', shape=(h_map, w_map))
        usblty = np.memmap(tmp_usblty_file, dtype=output_dtype, mode='w+', shape=(h_map, w_map))
        processed_region = np.memmap(tmp_processed_region_file, dtype=output_dtype, mode='w+', shape=(h_map, w_map))
        process_tiles(filename, args.tile_size, level, mask_path, args.mask_magnification, args.mask_ratio,
                      eval_quality, QA_model, usblty, normal, focus_artfcts, stain_artfcts,
                      other_artfcts, folding_artfcts, processed_region, n_workers=args.workers)
        slide_info = {'focus_artfcts': focus_artfcts, 'stain_artfcts': stain_artfcts, 'normal': normal,
                      'other_artfcts': other_artfcts, 'folding_artfcts': folding_artfcts, 'usblty': usblty,
                      'processed_region': processed_region}
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        np.save(path.join(args.save_folder, slide_name + '_quality_overlays.npy'), slide_info)

        cv2.imwrite(path.join(args.save_folder, slide_name + '_usability.png'),
                    generate_heatmap(usblty[:], processed_region, args.heatmap_mag))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_normal.png'),
                    generate_heatmap(normal[:], processed_region, args.heatmap_mag))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_staining_artfcts.png'),
                    generate_heatmap(stain_artfcts[:], processed_region, args.heatmap_mag))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_focus_artfcts.png'),
                    generate_heatmap(focus_artfcts[:], processed_region, args.heatmap_mag))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_other_artfcts.png'),
                    generate_heatmap(other_artfcts[:], processed_region, args.heatmap_mag))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_folding_artfcts.png'),
                    generate_heatmap(folding_artfcts[:], processed_region, args.heatmap_mag))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_processed_region.png'),
                    generate_heatmap(processed_region[:], processed_region, args.heatmap_mag))

    print('Done!')


if __name__ == "__main__":
    main()
