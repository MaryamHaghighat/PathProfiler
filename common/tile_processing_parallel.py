import sys
sys.path.extend(["../.", "."])
from common.wsi_reader import get_reader_impl
import numpy as np
import math
import cv2
from multiprocessing import Process, Queue
from tempfile import TemporaryFile
from PIL import Image
from pathlib import Path


class TileWorker(Process):
    def __init__(self, queue_in, slide_path, tile_size, downsample, ds_level_0, output, processing_fn, *processing_fn_args):
        Process.__init__(self, name='TileWorker')
        self.queue_in = queue_in
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.downsample = downsample
        self.ds_level_0 = ds_level_0
        self.slide = None
        self.processing_fn = processing_fn
        self.processing_fn_args = processing_fn_args
        self.output = output

    def run(self):
        reader = get_reader_impl(self.slide_path)
        self.slide = reader(self.slide_path)
        while True:
            data_in = self.queue_in.get()
            if data_in is None:
                break
            x_y = data_in
#            try:
            tile, _ = self.slide.read_region_ds(x_y, self.downsample, (self.tile_size, self.tile_size),
                                                normalize=False, downsample_level_0=self.ds_level_0)
            tile = self.processing_fn(tile, x_y, *self.processing_fn_args)
            if self.output is not None:
                x, y = x_y
                self.output[y:y+self.tile_size, x:x+self.tile_size] = tile
#            except:
                # handle tiles without data
            reader = get_reader_impl(self.slide_path)
            self.slide = reader(self.slide_path)


def _get_mask(mask_path):
    try:
        mask = cv2.imread(mask_path, -1)
        return mask/255
    except:
        print('NO TISSUE MASK FOUND')


def _get_padding(tile_size, width, height):
    pad_w = int((math.ceil(width / tile_size) - width / tile_size) * tile_size)
    pad_h = int((math.ceil(height / tile_size) - height / tile_size) * tile_size)
    return pad_w, pad_h


def process_tiles(slide_path, tile_size, tile_magnification, stride, downsample, ds_level_0, mask_path,
                  mask_magnification, mask_ratio, processing_fn,
                  *processing_fn_args, output_transform=lambda output: output, n_workers=8, output_dtype=None, unpad=True):
    reader = get_reader_impl(slide_path)
    slide = reader(slide_path)
    mask = _get_mask(mask_path)
    mask_ds = int(tile_magnification / mask_magnification)
    mask_tile_size = int(tile_size / mask_ds)
    width, height = int(round(slide.level_dimensions[0][0]/downsample)), int(round(slide.level_dimensions[0][1]/downsample))
    pad_w, pad_h = _get_padding(tile_size, width, height)
    width += pad_w
    height += pad_h
    width = int(min(width, mask.shape[1]*mask_ds))
    height = int(min(height, mask.shape[0]*mask_ds))
    queue_in = Queue()
    output = None
    if output_dtype is not None:
        tmp_file = TemporaryFile()
        output = np.memmap(tmp_file, dtype=output_dtype, mode='w+', shape=(height, width))

    workers = [TileWorker(queue_in, slide_path, tile_size, downsample, ds_level_0, output, processing_fn,
                          *processing_fn_args) for _ in range(n_workers)]
        
    for worker in workers:
        worker.start()

    for x in range(0, width, stride):
        for y in range(0, height, stride):
            tile_mask = mask[int(y / mask_ds): mask_tile_size + int(y / mask_ds),
                             int(x / mask_ds): mask_tile_size + int(x / mask_ds)]
            if tile_mask.mean() > mask_ratio:
                queue_in.put((x, y))

    for _ in range(n_workers):
        queue_in.put(None)
        
    for worker in workers:
        worker.join()
                
    if output is not None:
        output = output_transform(output)
        return output[:-pad_h, :-pad_w] if unpad else output
