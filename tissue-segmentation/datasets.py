import os
import glob
import cv2
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

from torch.utils.data import Dataset

from torchvision import transforms


class CLAHE(object):
    # histogram equalisation
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        HSV[:, :, 0] = self.clahe.apply(HSV[:, :, 0]) # TODO: this is a mistake, its supposed to be the V channel but that means retraining. Need to change the CLAHE function in deploy and retrain the model 
        img = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)

        return img


def augmentator():

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    ia.seed(np.random.randint(10000))
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode='symmetric',
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode='symmetric'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [  # convert images into their superpixel representation
                           iaa.WithChannels([0, 1, 2],
                                            iaa.OneOf([
                                                iaa.GaussianBlur((0, 3.0)),
                                                # blur images with a sigma between 0 and 3.0
                                                iaa.AverageBlur(k=(2, 7)),
                                                # blur image using local means with kernel sizes between 2 and 7
                                                iaa.MedianBlur(k=(3, 11)),
                                                # blur image using local medians with kernel sizes between 2 and 7
                                            ])),
                           iaa.WithChannels([0, 1, 2], iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                           # sharpen images
                           iaa.WithChannels([0, 1, 2], iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),
                           # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.WithChannels([0, 1, 2], iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ]))),
                           iaa.WithChannels([0, 1, 2], iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                                                 per_channel=0.5)),
                           # add gaussian noise to images
                           iaa.WithChannels([0, 1, 2], iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)])),
                           iaa.WithChannels([0, 1, 2], iaa.Invert(0.05, per_channel=True)),  # invert color channels
                           iaa.WithChannels([0, 1, 2], iaa.Add((-10, 10), per_channel=0.5)),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.WithChannels([0, 1, 2], iaa.AddToHueAndSaturation((-20, 20))),
                           # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.WithChannels([0, 1, 2], iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.BlendAlphaFrequencyNoise(
                                   exponent=(-4, 0),
                                   foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   background=iaa.contrast.LinearContrast((0.5, 2.0))
                               )])),
                           iaa.WithChannels([0, 1, 2], iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5)),
                           # improve or worsen the contrast
                           iaa.WithChannels([0, 1, 2], iaa.Grayscale(alpha=(0.0, 1.0))),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    return seq


class SegDataset(Dataset):

    def __init__(self, root, mode):

        files = glob.glob(os.path.join(root, mode, 'imgs', '*.jpg'))
        self.files = [name for name in files if os.path.exists(os.path.join(root, mode, 'labels', os.path.basename(name)))]
        self.labels = [os.path.join(root, mode, 'labels', os.path.basename(name)) for name in self.files]

        self.dir = root
        self.histeq = CLAHE()
        self.totensor = transforms.ToTensor()
        self.normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.seq = augmentator()
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # get image and label names
        img_name = self.files[idx]
        gt_name = self.labels[idx]

        # read the image and read ground truth
        image = cv2.cvtColor(cv2.imread(img_name, -1), cv2.COLOR_BGR2RGB)  # switch it to rgb
        gt = cv2.imread(gt_name, -1)
        if gt.ndim > 2:
            gt = gt[:, :, 0]
        gt[gt > 127] = 255
        gt[gt <= 127] = 0

        # normalise the colour and increase contrast with HistEQ
        image = self.histeq(image)

        # augmentation
        if self.mode == 'train':
            cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
            cat = self.seq.augment_image(cat)
            image = cat[:, :, 0:3]
            gt = cat[:, :, 3]
            gt[gt < 255] = 0

        # convert to torch
        image = self.totensor(image.copy())
        gt = self.totensor((gt / 255.0).astype(np.int))

        # normalised image between -1 and 1
        image = self.normalise(image)

        return image, gt, img_name


