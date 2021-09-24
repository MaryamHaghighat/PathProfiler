from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Sampler
import torch
import csv
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa


def augmentator():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    ia.seed(np.random.randint(10000))
    seq = iaa.Sequential(
        [
            # apply the following augmenters
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.25),  # vertically flip 25% of all images
            iaa.Rot90([0, 1, 2, 3]),
            sometimes(iaa.AddToHue((-30, 30))),
            sometimes(iaa.Affine(
                translate_percent={"x": (-0.03, 0.03), "y":  (-0.03, 0.03)},  # translate by -3 to +3 percent (per axis)
                shear=(-20, 20),  # shear by -20 to +20 degrees
                order=[0],  # use bicubic interpolation (fast)
                mode='reflect'  # use any of scikit-image's warping modes
            )),
        ],
        random_order=False
    )
    return seq


class Data(Dataset):
    def __init__(self, file_list, label_list, rand_margin):
        self.file_list = file_list
        self.label_list = label_list
        self.normalise = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.totensor = T.ToTensor()
        self.m = rand_margin
        self.seq = augmentator()

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index], -1)
        # As discussed in the paper, we saved images of 712*1024 pixels at 10X for annotation;
        # here we get the centre area with a small random margin for training
        if img.shape[0] > 512:
            m0 = np.random.randint(-self.m, self.m)
            m1 = np.random.randint(-self.m, self.m)
            img = img[100+m0:100+m0 + 512, 256+m1:256+m1 + 512, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_list[index]
        # Augment
        img = self.seq.augment_image(img)
        # resize
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # convert img
        img = self.normalise(self.totensor(img)).type(torch.FloatTensor)
        return img, label

    def __len__(self):
        return len(self.file_list)


class TestData(Dataset):
    def __init__(self, file_list, label_list):
        self.file_list = file_list
        self.label_list = label_list
        self.normalise = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.totensor = T.ToTensor()

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index], -1)
        if img.shape[0] > 512:
            img = img[100:100 + 512, 256:256 + 512, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_list[index]
        # resize
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # convert img
        img = self.normalise(self.totensor(img)).type(torch.FloatTensor)
        return img, label

    def __len__(self):
        return len(self.file_list)


class LabelSampler(Sampler):
    """Samples "batch_size" number of images using "weights" to get data from each class
       data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, batch_size, weights):
        weights = np.array(weights)
        labels = data_source.label_list
        self.sample_size = np.round(batch_size * weights).astype(np.int)
        self.ind_class0 = np.array(np.where([int(item[1]) for item in labels])).squeeze()  # No artefact
        self.ind_class1_1 = np.array(np.where([int(item[2]) == 1 for item in labels])).squeeze()  # staining issues
        self.ind_class1_2 = np.array(np.where([int(item[2]) == 2 for item in labels])).squeeze()  # staining issues
        self.ind_class2_1 = np.array(np.where([int(item[3]) == 1 for item in labels])).squeeze()  # out-of-focus
        self.ind_class2_2 = np.array(np.where([int(item[3]) == 2 for item in labels])).squeeze()  # out-of-focus
        self.ind_class3 = np.array(np.where([int(item[4]) for item in labels])).squeeze()  # Folding
        self.ind_class4 = np.array(np.where([int(item[5]) for item in labels])).squeeze()  # Other
        # unusable and NOT other
        self.ind_class5 = np.array(np.where([(int(item[0]) == 0 and (item != '000002')) for item in labels])).squeeze()

    def __iter__(self):
        total = self.__len__()
        for _ in range(total):
            samples_class0 = list(np.random.choice(self.ind_class0, size=self.sample_size[0], replace=True))
            samples_class1_1 = list(np.random.choice(self.ind_class1_1, size=self.sample_size[1], replace=True))
            samples_class1_2 = list(np.random.choice(self.ind_class1_2, size=self.sample_size[2], replace=True))
            samples_class2_1 = list(np.random.choice(self.ind_class2_1, size=self.sample_size[3], replace=True))
            samples_class2_2 = list(np.random.choice(self.ind_class2_2, size=self.sample_size[4], replace=True))
            samples_class3 = list(np.random.choice(self.ind_class3, size=self.sample_size[5], replace=True))
            samples_class4 = list(np.random.choice(self.ind_class4, size=self.sample_size[6], replace=True))
            samples_class5 = list(np.random.choice(self.ind_class5, size=self.sample_size[7], replace=True))
            out = samples_class0 + samples_class1_1 + samples_class1_2 + samples_class2_1 + samples_class2_2 + \
                samples_class3 + samples_class4 + samples_class5
            yield np.array(out)

    def __len__(self):
        return 500000


class split_train_test_val(object):
    def __init__(self):
        self.kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=88)
        self.train_inds = list()
        self.test_inds = list()
        self.val_inds = list()

    def __call__(self, data):
        self.data = data
        self.label_list = [''.join(row[1:].astype('str').tolist()) for row in self.data.values]
        i = 0
        for _, ind in self.kfold.split(self.label_list, self.label_list):
            if i < 8:
                self.train_inds.extend(ind)
            elif i == 8:
                self.test_inds.extend(ind)
            else:
                self.val_inds.extend(ind)
            i += 1
        self._writeinds('train_list.csv', self.data, self.train_inds)
        self._writeinds('test_list.csv', self.data, self.test_inds)
        self._writeinds('val_list.csv', self.data, self.val_inds)

    @staticmethod
    def _writeinds(file, data, inds):
        with open(file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['filename', 'usability', 'no-artefact', 'staining-issues', 'out-of-focus',
                                 'folding', 'other'])
            for ind in inds:
                csv_writer.writerow([*list(data.values[ind, :])])

