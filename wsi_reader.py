try:
    import openslide
    import tifffile
except:
    pass

try:
    from pixelengine import PixelEngine
    from softwarerendercontext import SoftwareRenderContext
    from softwarerenderbackend import SoftwareRenderBackend
except:
    pass

import numpy as np
import os
import cv2
from pathlib import Path
from typing import Tuple

class WSIReader:
    def close(self):
        pass

    @property
    def tile_dimensions(self):
        pass

    def read_region(self, x_y, level, tile_size, normalize=True, downsample_level_0=False):
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        x, y = x_y
        if downsample_level_0 and level > 0:
            downsample = round(self.level_dimensions[0][0] / self.level_dimensions[level][0])
            x, y = x * downsample, y * downsample
            tile_w, tile_h = tile_size[0] * downsample, tile_size[1] * downsample
            width, height = self.level_dimensions[0]
        else:
            tile_w, tile_h = tile_size
            width, height = self.level_dimensions[level]

        tile_w = width - x if (x + tile_w > width) else tile_w
        tile_h = height - y if (y + tile_h > height) else tile_h
        tile, alfa_mask = self._read_region((x, y), 0 if downsample_level_0 else level, (tile_w, tile_h))
        if downsample_level_0 and level > 0:
            tile_w = tile_w // downsample
            tile_h = tile_h // downsample
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_CUBIC).astype(
                np.bool)

        if normalize:
            tile = self._normalize(tile)

        padding = [(0, tile_size[1] - tile_h), (0, tile_size[0] - tile_w)]
        tile = np.pad(tile, padding + [(0, 0)] * (len(tile.shape) - 2), 'constant', constant_values=0)
        alfa_mask = np.pad(alfa_mask, padding, 'constant', constant_values=0)

        return tile, alfa_mask

    def _read_region(self, x_y, level, tile_size):
        pass

    def get_best_level_for_downsample(self, downsample):
        if downsample < self.level_downsamples[0]:
            return 0

        for i in range(1, self.level_count):
            if downsample < self.level_downsamples[i]:
                return i - 1

        return self.level_count - 1

    def get_downsampled_slide(self, dims, normalize=True):
        downsample = min(a / b for a, b in zip(self.level_dimensions[0], dims))
        level = self.get_best_level_for_downsample(downsample)
        slide_downsampled, alfa_mask = self.read_region((0, 0), level, self.level_dimensions[level],
                                                        normalize=normalize)
        slide_downsampled = cv2.resize(slide_downsampled, dims, interpolation=cv2.INTER_CUBIC)
        alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), dims, interpolation=cv2.INTER_CUBIC).astype(np.bool)
        return slide_downsampled, alfa_mask

    @property
    def level_dimensions(self):
        pass

    @property
    def level_count(self):
        pass

    @property
    def mpp(self):
        pass

    @property
    def dtype(self):
        pass

    @property
    def n_channels(self):
        pass

    @property
    def level_downsamples(self):
        if not hasattr(self, '_level_downsamples'):
            self._level_downsamples = []
            width, height = self.level_dimensions[0]
            for level in range(self.level_count):
                w, h = self.level_dimensions[level]
                ds = round(width / w)
                self._level_downsamples.append(ds)
        return self._level_downsamples

    @staticmethod
    def _normalize(pixels):
        if np.issubdtype(pixels.dtype, np.integer):
            pixels = (pixels / 255).astype(np.float32)
        return pixels

    @staticmethod
    def _is_rgba(pixels):
        return np.issubdtype(pixels.dtype, np.integer) and len(pixels.shape) == 3 and pixels.shape[2] == 4

    @staticmethod
    def _round(x, base):
        return base * round(x / base)


class OpenSlideReader(WSIReader):
    def __init__(self, slide_path: str):
        # define slide
        assert os.path.exists(slide_path), 'slide_path does not exist'

        self.slide_path = slide_path
        self._slide = openslide.open_slide(self.slide_path)

    def close(self):
        self.slide_path = None
        self._slide.close()

    def _read_region(self, x_y: Tuple[int, int], magnification: float, tile_size: Tuple[int, int]):
        """
        :param x_y: (x, y) coordinate at the specified magnification
        :param magnification: magnification of the tile
        :param tile_size: (W, H) tile size
        :return: uint8 image
        """
        # make sure that the magnification is between (0, objective]
        assert np.logical_and(magnification > 0.0,
                              magnification <= self.objective), 'magnification > 0.0, magnification <= {}'.format(
            self.objective)

        X, Y = x_y
        W, H = tile_size

        assert X >= 0 and Y >= 0, "X and Y must be non-negative."
        assert W > 0 and H > 0, "tile size must be strictly positive."

        # rescale factors
        rescale_factor = magnification / self.objective
        best_level_downsample = self._slide.get_best_level_for_downsample(1 / rescale_factor)
        rescale_factor_from_best_level_downsample = magnification * (self._slide.level_downsamples[best_level_downsample] / self.objective)

        # read tile
        X0 = int(round(X / rescale_factor))
        Y0 = int(round(Y / rescale_factor))
        W = int(round(W / rescale_factor_from_best_level_downsample))
        H = int(round(H / rescale_factor_from_best_level_downsample))

        tile = np.array(self._slide.read_region((X0, Y0), best_level_downsample, (W, H)), dtype=np.uint8)
        tile = cv2.resize(tile, (0, 0), fx=rescale_factor_from_best_level_downsample,
                                        fy=rescale_factor_from_best_level_downsample)

        alfa_mask = tile[:, :, 3] > 0
        tile = tile[:, :, :3]
        return tile, alfa_mask

    def get_best_level_for_downsample(self, downsample: float):
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._slide.level_dimensions

    @property
    def level_count(self):
        return self._slide.level_count

    @property
    def mpp(self):
        return float(self._slide.properties['openslide.mpp-x']), float(self._slide.properties['openslide.mpp-y'])

    @property
    def dtype(self):
        return np.dtype(np.uint8)

    @property
    def n_channels(self):
        return 3

    @property
    def level_downsamples(self):
        return self._slide.level_downsamples

    @property
    def tile_dimensions(self):
        if not hasattr(self, '_tile_dimensions'):
            self._tile_dimensions = []
            for level in range(self.level_count):
                tile_width = self._slide.properties[f'openslide.level[{level}].tile-width']
                tile_height = self._slide.properties[f'openslide.level[{level}].tile-height']
                self._tile_dimensions.append((tile_width, tile_height))
        return self._tile_dimensions

    @property
    def objective(self):
        # objective
        field = [field for field in self._slide.properties.keys() if 'objective' in field][0]
        return float(self._slide.properties[field])


class TiffReader(WSIReader):
    def __init__(self, slide_path, series=0):
        # define slide
        assert os.path.exists(slide_path), 'slide_path does not exist'

        self.slide_path = slide_path
        self.series = series
        self._reader = tifffile.TiffFile(str(slide_path))

    def close(self):
        self.slide_path = None
        self._reader.close()

    @property
    def tile_dimensions(self):
        if not hasattr(self, '_tile_dimensions'):
            self._tile_dimensions = []
            for level in range(self.level_count):
                page = self._reader.series[self.series].levels[level].pages[0]
                self._tile_dimensions.append((page.tilewidth, page.tilelength))
        return self._tile_dimensions

    def _read_region(self, x_y: Tuple[int, int], magnification: float, tile_size: Tuple[int, int]):
        """
        :param x_y: (x, y) coordinate at the specified magnification
        :param magnification: magnification of the tile
        :param tile_size: (W, H) tile size
        :return: uint8 image
        """

        X, Y = x_y
        W, H = tile_size

        assert X >= 0 and Y >= 0, "x and y must be non-negative."
        assert W > 0 and H > 0, "tile_size must be strictly positive."

        # make sure that the magnification is between (0, objective]
        assert np.logical_and(magnification > 0.0,
                              magnification <= self.objective), 'magnification > 0.0, magnification <= {}.'.format(self.objective)

        assert self.objective is not None, 'could not retrieve metadata'

        # rescale factors
        rescale_factor = magnification / self.objective
        best_level_downsample = self.get_best_level_for_downsample(1 / rescale_factor)
        rescale_factor_from_best_level_downsample = magnification * (self.level_downsamples[best_level_downsample] / self.objective)

        page = self._reader.series[self.series].levels[best_level_downsample].pages[0]

        j0, i0 = X, Y
        j0 = int(round(j0 / rescale_factor_from_best_level_downsample))
        i0 = int(round(i0 / rescale_factor_from_best_level_downsample))
        w = int(round(W / rescale_factor_from_best_level_downsample))
        h = int(round(H / rescale_factor_from_best_level_downsample))

        i1, j1 = i0 + h, j0 + w

        if not page.is_tiled:
            tile = page.asarray()[i0:i1, j0:j1]
            if tile.shape[2] == 1:
                tile = np.squeeze(tile, axis=2)
            alfa_mask = np.ones(tile.shape, dtype=np.bool)
            return tile, alfa_mask

        im_width = page.imagewidth
        im_height = page.imagelength

        if h < 1 or w < 1:
            raise ValueError("h and w must be strictly positive.")

        if i0 < 0 or j0 < 0 or i1 > im_height or j1 > im_width:
            raise ValueError("Requested crop area is out of image bounds.")

        tile_width, tile_height = page.tilewidth, page.tilelength

        tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
        tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

        tile_per_line = int(np.ceil(im_width / tile_width))

        out = np.zeros((page.imagedepth,
                        (tile_i1 - tile_i0) * tile_height,
                        (tile_j1 - tile_j0) * tile_width,
                        page.samplesperpixel), dtype=page.dtype)

        fh = page.parent.filehandle

        jpegtables = page.tags.get('JPEGTables', None)
        if jpegtables is not None:
            jpegtables = jpegtables.value

        for i in range(tile_i0, tile_i1):
            for j in range(tile_j0, tile_j1):
                index = int(i * tile_per_line + j)

                offset = page.dataoffsets[index]
                bytecount = page.databytecounts[index]

                if bytecount == 0:
                    continue

                fh.seek(offset)
                data = fh.read(bytecount)

                tile, indices, shape = page.decode(data, index, jpegtables=jpegtables)
                im_i = (i - tile_i0) * tile_height
                im_j = (j - tile_j0) * tile_width
                out[:, im_i: im_i + tile_height, im_j: im_j + tile_width] = tile

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        tile = out[:, im_i0: im_i0 + h, im_j0: im_j0 + w]
        tile = tile[0, :, :]

        if tile.shape[2] == 1:
            tile = np.squeeze(tile, axis=2)

        tile = cv2.resize(tile, (0, 0), fx=rescale_factor_from_best_level_downsample,
                                        fy=rescale_factor_from_best_level_downsample)

        if self._is_rgba(page):
            alfa_mask = tile[:, :, 3] > 0
            tile = tile[:, :, :3]
        else:
            alfa_mask = np.ones(tile.shape[:2], dtype=np.bool)
        return tile, alfa_mask

    @property
    def level_dimensions(self):
        if not hasattr(self, '_level_dimensions'):
            self._level_dimensions = []
            for level in range(self.level_count):
                page = self._reader.series[self.series].levels[level].pages[0]
                self._level_dimensions.append((page.imagewidth, page.imagelength))
        return self._level_dimensions

    @property
    def level_count(self):
        return len(self._reader.series[self.series].levels)

    @property
    def mpp(self):
        page = self._reader.series[self.series].levels[0].pages[0]
        try:
            description = page.description
            return tifffile.tifffile.svs_description_metadata(description)
        except:
            return None

    @property
    def objective(self):
        metadata = self.mpp
        if metadata is None:
            return metadata

        mpp = metadata['MPP']
        if np.logical_and(0.2 <= mpp, mpp <= 0.3):
            objective = 40.0
        elif np.logical_and(0.45 <= mpp, mpp <= 0.55):
            objective = 20.0
        elif np.logical_and(0.9 <= mpp, mpp <= 1.1):
            objective = 10.0
        else:
            raise NotImplementedError

        return objective

    @property
    def dtype(self):
        return self._reader.series[self.series].levels[0].pages[0].dtype

    @property
    def n_channels(self):
        page = self._reader.series[self.series].levels[0].pages[0]
        channels = page.samplesperpixel
        return channels - 1 if self._is_rgba(page) else channels


class IsyntaxReader(WSIReader):
    def __init__(self, slide_path):
        self.slide_path = slide_path
        self._pe = PixelEngine(SoftwareRenderBackend(), SoftwareRenderContext())
        self._pe['in'].open(slide_path)
        self._view = self._pe['in']['WSI'].source_view
        trunc_bits = {0: [0, 0, 0]}
        self._view.truncation(False, False, trunc_bits)

    def close(self):
        self._pe['in'].close()

    @property
    def tile_dimensions(self):
        if not hasattr(self, '_tile_dimensions'):
            self._tile_dimensions = [tuple(self._pe['in']['WSI'].block_size()[:2])] * self.level_count
        return self._tile_dimensions

    def _read_region(self, x_y, level, tile_size):
        x_start, y_start = x_y
        ds = self.level_downsamples[level]
        x_start *= ds
        y_start *= ds
        tile_w, tile_h = tile_size
        x_end, y_end = x_start + (tile_w - 1) * ds, y_start + (tile_h - 1) * ds
        view_range = [x_start, x_end, y_start, y_end, level]
        regions = self._view.request_regions([view_range],
                                             self._view.data_envelopes(level),
                                             True, [255, 255, 255], self._pe.BufferType(1))
        region, = self._pe.wait_any(regions)
        tile = np.empty(np.prod(tile_size) * 4, dtype=np.uint8)
        region.get(tile)
        tile.shape = (tile_h, tile_w, 4)
        return tile[:, :, :3], tile[:, :, 3] > 0

    @property
    def level_dimensions(self):
        if not hasattr(self, '_level_dimensions'):
            self._level_dimensions = []
            for level in range(self.level_count):
                x_start = self._view.dimension_ranges(level)[0][0]
                x_step = self._view.dimension_ranges(level)[0][1]
                x_end = self._view.dimension_ranges(level)[0][2]
                y_start = self._view.dimension_ranges(level)[1][0]
                y_step = self._view.dimension_ranges(level)[1][1]
                y_end = self._view.dimension_ranges(level)[1][2]
                range_x = ((x_end - x_start) // x_step) + 1
                range_y = ((y_end - y_start) // y_step) + 1
                self._level_dimensions.append((range_x, range_y))
        return self._level_dimensions

    @property
    def level_count(self):
        return self._view.num_derived_levels + 1

    @property
    def mpp(self):
        return self._view.scale[0], self._view.scale[1]

    @property
    def dtype(self):
        return np.dtype(np.uint8)

    @property
    def n_channels(self):
        return 3

    @property
    def level_downsamples(self):
        if not hasattr(self, '_level_downsamples'):
            self._level_downsamples = [self._view.dimension_ranges(level)[0][1] for level in range(self.level_count)]
        return self._level_downsamples


def get_reader_impl(slide_path):
    if Path(slide_path).suffix == '.isyntax':
        return IsyntaxReader
    else:
        return TiffReader
