import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from common.wsi_reader import get_reader_impl
from common.tile_processing_parallel import process_tiles
from common.models import ResNet18
import glob
import os
import numpy as np
import argparse
import cv2
import csv
import timeit
import os.path as path
from tempfile import mkdtemp
from torchvision import transforms
import torch
from matplotlib import cm
import zarr


'''
Tile-level quality assessment of whole slides using pretrained model: "checkpoint_106.pth"
output: saves quality overlays and the processed region in a npy file as:
slide_info = {'focus_artfcts': focus_artfcts, 'stain_artfcts': stain_artfcts,
'normal': normal, 'other_artfcts': other_artfcts, 'folding_artfcts': folding_artfcts,
'usblty': usblty, 'processed_region': processed_region}

Note: The pixel size of quality overlays is slide_size_at_5X / tile_size, however, they can be saved at 
a given magnification with param "overlay_magnification"
'''

parser = argparse.ArgumentParser()
parser.add_argument('--slide_dir', default='WSIs', help='path to slides dir', type=str, required=True)
parser.add_argument('--slide_id', default='*', type=str, help='slide filename ("*" for all slides)')
parser.add_argument('--mpp_level_0', type=float, default=None, help='manually enter mpp at level 0 if not readable from slides')
parser.add_argument('--mask_dir', default='tissue-masks', help='path to tissue mask folder')
parser.add_argument('--tile_size', type=int, default=256, help='our model is trained with images size 256')
parser.add_argument('--mask_magnification', type=float, default=1.25)
parser.add_argument('--mask_ratio', type=float, default=0.2)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--model', default='checkpoint_106.pth', type=str)
parser.add_argument('--overlay_magnification', default=1.25, type=float)
parser.add_argument('--save_folder', default='quality-overlays', type=str, help='path to the folder to save results')
args = parser.parse_args()

      
def eval_quality(tile, x_y, QA_model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    totensor = transforms.ToTensor()
    tile = cv2.resize(np.array(tile), (224, 224), interpolation=cv2.INTER_CUBIC)
    tile = normalize(totensor(tile)).type(torch.FloatTensor)
    with torch.no_grad():
        out = QA_model.embedding(tile.unsqueeze(0)).squeeze(0).cpu()
        out = out.data.numpy().clip(0, 1)
    return np.pad(out, (0, 1), mode='constant', constant_values=1)

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
    heatmap[processed_region == 0] = 128
    heatmap=heatmap.repeat(int(heatmap_mag*256/5.), axis=0).repeat(int(heatmap_mag*256/5.), axis=1)
    return heatmap


def main():

    #############################################################
    # sanity check
    assert path.isfile(args.model), "=> no checkpoint found at '{}'".format(args.model)
    #############################################################

    # define network
    QA_model = ResNet18(n_classes=6)
    print("=> loading checkpoint '{}'".format(args.model))
    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    QA_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.model, checkpoint['epoch']))
    QA_model.eval()
    QA_model.cpu()
    if not path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    dir_list = glob.glob(path.join(args.slide_dir, args.slide_id))

    mpp2mag = {.25: 40, .5: 20, 1: 10}
    for filename in dir_list:
        start = timeit.default_timer()
        basename = path.basename(filename)
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
        print('Processing', basename)
        wsi_highest_magnification = mpp2mag[.25 * round(float(mpp) / .25)]
        downsample = wsi_highest_magnification / 5.  # Model is trained for tiles at 5X
        w, h = int(np.round(slide.level_dimensions[0][0]/downsample)), int(np.round(slide.level_dimensions[0][1]/downsample))
        w_map, h_map = int(np.round(w/args.tile_size)), int(np.round(h/args.tile_size))
        store = zarr.TempStore()
        try:
            output_zarr = zarr.open(store, mode='w-', shape=(h_map, w_map, 7), chunks=(1, 1, 7), dtype=np.float16)
            output = process_tiles(filename, args.tile_size, 5.0, args.tile_size, downsample, True, output_zarr, mask_path, args.mask_magnification, args.mask_ratio,
                        eval_quality, QA_model, n_workers=args.workers, unpad=False)
            slide_info = {
                'usblty': output[:,:,0],
                'normal': output[:,:,1],
                'stain_artfcts': output[:,:,2],
                'focus_artfcts': output[:,:,3],
                'folding_artfcts': output[:,:,4], 
                'other_artfcts': output[:,:,5],
                'processed_region': output[:,:,6]
            }
        finally:
            store.rmdir()
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        np.save(path.join(args.save_folder, slide_name + '_quality_overlays.npy'), slide_info)

        cv2.imwrite(path.join(args.save_folder, slide_name + '_usability.png'),
                    generate_heatmap(slide_info['usblty'], slide_info['processed_region'], args.overlay_magnification))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_normal.png'),
                    generate_heatmap(slide_info['normal'], slide_info['processed_region'], args.overlay_magnification))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_staining_artfcts.png'),
                    generate_heatmap(slide_info['stain_artfcts'], slide_info['processed_region'], args.overlay_magnification))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_focus_artfcts.png'),
                    generate_heatmap(slide_info['focus_artfcts'], slide_info['processed_region'], args.overlay_magnification))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_other_artfcts.png'),
                    generate_heatmap(slide_info['other_artfcts'], slide_info['processed_region'], args.overlay_magnification))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_folding_artfcts.png'),
                    generate_heatmap(slide_info['folding_artfcts'], slide_info['processed_region'], args.overlay_magnification))
        cv2.imwrite(path.join(args.save_folder, slide_name + '_processed_region.png'),
                    generate_heatmap(slide_info['processed_region'], slide_info['processed_region'], args.overlay_magnification))

    print('Done!')


if __name__ == "__main__":
    main()
