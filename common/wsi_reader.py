import numpy as np
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
import re
from fractions import Fraction


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
            
        tile_w = tile_w + x if x < 0 else tile_w
        tile_h = tile_h + y if y < 0 else tile_h
        x = max(x, 0)
        y = max(y, 0)
        tile_w = width - x if (x + tile_w > width) else tile_w
        tile_h = height - y if (y + tile_h > height) else tile_h
        tile, alfa_mask = self._read_region((x,y), 0 if downsample_level_0 else level, (tile_w, tile_h))
        if downsample_level_0 and level > 0:
            tile_w = tile_w // downsample
            tile_h = tile_h // downsample
            x = x // downsample
            y = y // downsample
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_CUBIC).astype(bool)
        
        if normalize:
            tile = self._normalize(tile)
        
        padding = [(y-x_y[1],tile_size[1]-tile_h+min(x_y[1],0)), (x-x_y[0], tile_size[0]-tile_w+min(x_y[0],0))]
        tile = np.pad(tile, padding + [(0,0)]*(len(tile.shape)-2), 'constant', constant_values=0)
        alfa_mask = np.pad(alfa_mask, padding, 'constant', constant_values=0)
        
        return tile, alfa_mask
        
    def read_region_ds(self, x_y, downsample, tile_size, normalize=True, downsample_level_0=False):
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
            
        level = 0 if downsample_level_0 else self.get_best_level_for_downsample(downsample)
        x_y_level = [round(coord * downsample / self.level_downsamples[level]) for coord in x_y]
        tile_size_level = [round(dim * downsample / self.level_downsamples[level]) for dim in tile_size]
        tile, alfa_mask = self.read_region(x_y_level, level, tile_size_level, False, False)
        tile = cv2.resize(tile, tile_size, interpolation=cv2.INTER_CUBIC)
        alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), tile_size, interpolation=cv2.INTER_CUBIC).astype(bool)
        
        if normalize:
            tile = self._normalize(tile)
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
        slide_downsampled, alfa_mask = self.read_region((0,0), level, self.level_dimensions[level], normalize=normalize)
        slide_downsampled = cv2.resize(slide_downsampled, dims, interpolation=cv2.INTER_CUBIC)
        alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), dims, interpolation=cv2.INTER_CUBIC).astype(bool)
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
    def _round(x, base):
        return base * round(x/base)
        
class OpenSlideReader(WSIReader):
    def __init__(self, slide_path):
        import openslide
        self.slide_path = slide_path
        self._slide = openslide.open_slide(str(slide_path))
        
    def close(self):
        self.slide_path = None
        self._slide.close()
        if hasattr(self, '_tile_dimensions'):
            delattr(self, '_tile_dimensions')
           
    def _read_region(self, x_y, level, tile_size):
        ds = self.level_downsamples[level]
        x, y = x_y
        x_y = round(x * ds), round(y * ds)
        tile = np.array(self._slide.read_region(x_y, level, tile_size), dtype=np.uint8)
        alfa_mask = tile[:,:,3] > 0
        tile = tile[:,:,:3]
        return tile, alfa_mask
    
    def get_best_level_for_downsample(self, downsample):
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
    
    
class TiffReader(WSIReader):
    def __init__(self, slide_path, series=0):
        import tifffile
        self.slide_path = slide_path
        self.series = series
        self._reader = tifffile.TiffFile(str(slide_path))

    def close(self):
        self.slide_path = None
        self._reader.close()
        if hasattr(self, '_mpp'):
            delattr(self, '_mpp')
        if hasattr(self, '_tile_dimensions'):
            delattr(self, '_tile_dimensions')
        if hasattr(self, '_level_dimensions'):
            delattr(self, '_level_dimensions')
        if hasattr(self, '_level_downsamples'):
            delattr(self, '_level_downsamples')
        
    @property
    def tile_dimensions(self):
        if not hasattr(self, '_tile_dimensions'):
            self._tile_dimensions = []
            for level in range(self.level_count):
                page = self._reader.series[self.series].levels[level].pages[0]
                self._tile_dimensions.append((page.tilewidth, page.tilelength))
        return self._tile_dimensions
        
    def _read_region(self, x_y, level, tile_size):
        level = self._reader.series[self.series].levels[level]
        planes, alfa_mask = zip(*[self._read_page_region(page, x_y, tile_size) for page in level.pages])
        alfa_mask = [a for a in alfa_mask if a is not None]
        
        if len(planes) > 1 and planes[0].ndim == 2:
            planes = np.stack(planes, -1)
        else:
            planes = np.concatenate(planes, -1)

        if len(alfa_mask) == 0:
            alfa_mask = np.ones(planes.shape[:2], dtype=bool)
        elif len(alfa_mask) > 1 and alfa_mask[0].ndim == 2:
            alfa_mask = np.stack(alfa_mask, -1)
        else:
            alfa_mask = np.concatenate(alfa_mask, -1)
        
        if alfa_mask.ndim == 3 and (planes.ndim == 2 or alfa_mask.shape[2] != planes.shape[2]):
            alfa_mask = np.logical_and.reduce(alfa_mask, axis=2)
        
        return planes, alfa_mask
        
    @staticmethod
    def _get_alfa_mask(tile, samplesperpixel, extrasamples):
        alfa_mask = None
        off = samplesperpixel - len(extrasamples)
        alfa_mask_idx = [off + i for i, sample in enumerate(extrasamples) if sample > 0]
        if len(alfa_mask_idx) > 0:
            alfa_mask = tile[:,:,alfa_mask_idx] > 0
            tile = np.delete(tile, alfa_mask_idx, axis=2)
            if alfa_mask.ndim == 3 and alfa_mask.shape[2] == 1:
                alfa_mask = np.squeeze(alfa_mask, axis=2)
        return tile, alfa_mask
            
    def _read_page_region(self, page, x_y, tile_size):
        j0, i0 = x_y
        w, h = tile_size
        i1, j1 = i0 + h, j0 + w
        
        if not page.is_tiled:
            tile = page.asarray()[i0:i1, j0:j1]
            if tile.ndim == 3 and tile.shape[2] == 1:
                tile = np.squeeze(tile, axis=2)
            tile, alfa_mask = self._get_alfa_mask(tile, page.samplesperpixel, page.extrasamples)
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
                out[:,im_i: im_i + tile_height, im_j: im_j + tile_width] = tile

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        tile = out[:, im_i0: im_i0 + h, im_j0: im_j0 + w]
        tile = tile[0,:,:]
        
        if tile.ndim == 3 and tile.shape[2] == 1:
            tile = np.squeeze(tile, axis=2)

        tile, alfa_mask = self._get_alfa_mask(tile, page.samplesperpixel, page.extrasamples)
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
        if not hasattr(self, '_mpp'):
            self._mpp = (None, None)
            page = self._reader.series[0].levels[0].pages[0]
            if page.is_svs:
                import tifffile
                metadata = tifffile.tifffile.svs_description_metadata(page.description)
                self._mpp = (metadata['MPP'], metadata['MPP'])
            elif page.is_ome:
                root = ET.fromstring(self._reader.ome_metadata)
                namespace = re.search('^{.*}', root.tag)
                namespace = namespace.group() if namespace else ''
                pixels = list(root.findall(namespace + 'Image'))[self.series].find(namespace + 'Pixels')
                self._mpp = (float(pixels.get('PhysicalSizeX')), float(pixels.get('PhysicalSizeY')))
            elif page.is_philips:
                root = ET.fromstring(self._reader.philips_metadata)
                mpp = float(root.find("./Attribute/[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject/[@ObjectType='DPScannedImage']/Attribute/[@Name='PIM_DP_IMAGE_TYPE'][.='WSI']/Attribute[@Name='PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE']/Array/DataObject[@ObjectType='PixelDataRepresentation']/Attribute[@Name='DICOM_PIXEL_SPACING']").text)
                self._mpp = (mpp, mpp)
            elif page.is_ndpi or page.is_scn or page.is_qpi or True:
                page = self._reader.series[self.series].levels[0].pages[0]
                if page.tags['ResolutionUnit'].value == 3:
                    self._mpp = (1e4/float(Fraction(*page.tags['XResolution'].value)),\
                                 1e4/float(Fraction(*page.tags['YResolution'].value)))
        return self._mpp
    
    @property
    def dtype(self):
        return self._reader.series[self.series].levels[0].pages[0].dtype
    
    @property
    def n_channels(self):
        page = self._reader.series[self.series].levels[0].pages[0]
        channels = page.samplesperpixel
        return channels - len([sample for sample in page.extrasamples if sample > 0])

class IsyntaxReader(WSIReader):
    def __init__(self, slide_path):
        from pixelengine import PixelEngine
        from softwarerendercontext import SoftwareRenderContext
        from softwarerenderbackend import SoftwareRenderBackend
        self.slide_path = slide_path
        self._pe = PixelEngine(SoftwareRenderBackend(), SoftwareRenderContext())
        self._pe['in'].open(str(slide_path), 'ficom')
        self._view = self._pe['in']['WSI'].source_view
        trunc_bits = {0: [0, 0, 0]}
        self._view.truncation(False, False, trunc_bits)
        
    def close(self):
        self.slide_path = None
        self._pe['in'].close()
        if hasattr(self, '_tile_dimensions'):
            delattr(self, '_tile_dimensions')
        if hasattr(self, '_level_dimensions'):
            delattr(self, '_level_dimensions')
        if hasattr(self, '_level_downsamples'):
            delattr(self, '_level_downsamples')
        
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
        x_end, y_end = x_start + (tile_w-1)*ds, y_start + (tile_h-1)*ds
        view_range = [x_start, x_end, y_start, y_end, level]
        regions = self._view.request_regions([view_range],
                            self._view.data_envelopes(level),
                            True, [255,255,255], self._pe.BufferType(1))
        region, = self._pe.wait_any(regions)
        tile = np.empty(np.prod(tile_size)*4, dtype=np.uint8)
        region.get(tile)
        tile.shape = (tile_h, tile_w, 4)
        return tile[:,:,:3], tile[:,:,3] > 0

    @property
    def level_dimensions(self):
        if not hasattr(self, '_level_dimensions'):
            self._level_dimensions = []
            for level in range(self.level_count):
                x_step, x_end = self._view.dimension_ranges(level)[0][1:]
                y_step, y_end = self._view.dimension_ranges(level)[1][1:]
                range_x = (x_end + 1) // x_step
                range_y = (y_end + 1) // y_step
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
    elif Path(slide_path).suffix == '.ndpi':
        return OpenSlideReader
    else:
        return TiffReader
