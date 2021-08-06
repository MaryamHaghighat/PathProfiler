from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import to_tensor
from process_slide import SlideProcessor
import time
import cv2

class ColorInverter(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 1 - x
        
class Processor:
    def __init__(self, foo, bar, xyz=False):
        self.model = ColorInverter()
        self.foo = foo
        self.bar = bar
        self.xyz = xyz
            
    def __call__(self, tiles, tiles_masks, x_y):
        tiles = torch.stack([to_tensor(x) for x in tiles])
        with torch.no_grad():
            y = (self.model(tiles).numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
        return y * tiles_masks[..., None]

class OtsuTissueFilter:
    def __init__(self, reader, downsample, filter_colors, luminance_range, tissue_threshold=0.05, color_tolerance=5):
        self.reader = reader
        self.downsample = downsample
        self.filter_colors = [np.array(color) / 255 for color in filter_colors] if filter_colors else []
        self.luminance_range = luminance_range
        self.tissue_threshold = tissue_threshold
        self.color_tolerance = color_tolerance / 255
        
    def __call__(self, tile, alfa_mask, x_y):
        tile_mask = alfa_mask
        if len(tile.shape) == 3:
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            for color in self.filter_colors:
                color_mask = ~cv2.inRange(tile, color-self.color_tolerance, color+self.color_tolerance).astype(bool)
                tile_mask &= color_mask
        else:
            tile_gray = tile
            
        gray_mask = (tile_gray <= self.luminance_threshold) & \
                    (tile_gray >= self.luminance_range[0]) & \
                    (tile_gray <= self.luminance_range[1])
                    
        tile_mask &= gray_mask
        
        if tile_mask.sum() / tile_mask.size >= self.tissue_threshold:
            return tile_mask
        else:
            return None
        
    @property
    def luminance_threshold(self):
        if not hasattr(self, '_luminance_threshold'):
            level = self.reader.get_best_level_for_downsample(self.downsample)
            slide, alfa_mask = self.reader.get_downsampled_slide(self.reader.level_dimensions[level])
            slide_mask = alfa_mask
        
            if len(slide.shape) == 3:
                for color in self.filter_colors:
                    color_mask = ~cv2.inRange(slide, color-self.color_tolerance, color+self.color_tolerance).astype(bool)
                    slide_mask &= color_mask
                slide = cv2.cvtColor(slide, cv2.COLOR_RGB2GRAY)
            
            gray_mask = (slide >= self.luminance_range[0]) & \
                        (slide <= self.luminance_range[1])
                    
            slide_mask &= gray_mask
        
            slide = (slide * 255).astype(np.uint8)
            threshold, _ = cv2.threshold(slide[slide_mask].ravel(), 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            threshold /= 255
        
            self._luminance_threshold = threshold
            
        return self._luminance_threshold
        
def main(args):
    slide_processor = SlideProcessor(args.tile_size, args.stride, 3, np.uint8, 512, OtsuTissueFilter, ((8, [(250, 5, 5)], (0.05, 0.95)), dict(tissue_threshold=0.15)), Processor, ((0, 1), dict(xyz=True)), n_workers=args.n_workers)
    for input_file in args.input_files:
        slide_id = input_file.stem
        print('processing...', end=' ')
        start = int(time.time())
        output = slide_processor(input_file, args.level)
        end = int(time.time())
        print(f'Done ({end-start} s)')
        print('writing...', end=' ')
        start = int(time.time())
        SlideProcessor.write_to_tiff(output, slide_id + '_processed.tif', tile=True, tile_width=args.tile_size, tile_height=args.tile_size, squash=False, pyramid=True, bigtiff=False, compression='deflate')
        end = int(time.time())
        print(f'Done ({end-start} s)')
        
parser = ArgumentParser(description='tissue vs background segmentation example')
parser.add_argument('input_files', type=Path, nargs='+')
parser.add_argument('--level', type=int, default=0)
parser.add_argument('--tile-size', dest='tile_size', type=int, default=512)
parser.add_argument('--stride', type=int, default=256)
parser.add_argument('--n-workers', dest='n_workers', type=int, default=8)
args = parser.parse_args()

main(args)
