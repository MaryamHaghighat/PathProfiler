import sys
sys.path.append("common")
from datasets import Data, TestData, LabelSampler
from models import modelIQA
import torch
import csv
import numpy as np
import pandas
from torch.utils.data import Dataset
import argparse
import os
import torch.backends.cudnn as cudnn
import cv2


''' Train a multi-label ResNet18 model with six quality-related outputs: 
1- usability, 2- no artefact, 3- staining issues, 4- out-of-focus, 5- folding and 6- other artefacts.
Our dataset is labelled with {0, 1, 2} for the third and fourth outputs, with "1" indicating "slight" and 
"2" indicating "severe" staining issues/out-of-focus. All other outputs are labelled with either 0 or 2.
Labels are normalized to 1 for training, i.e. we will have {0, .5, 1} target values in the loss function. 
'''

parser = argparse.ArgumentParser(description='Training multi-label ResNet18 model')
parser.add_argument('--checkpoint_folder', type=str, default='checkpoint')
parser.add_argument('--train_dataset', type=str, default='train_list.csv')
parser.add_argument('--val_dataset', type=str, default='val_list.csv')
parser.add_argument('--loss_fn', type=str, default='huber', help='Choose between MSE, huber or BCE loss functions')
parser.add_argument('--gpu_id', type=str, default='1')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--n_classes', type=int, default=6, help='number of output classes')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=.0001)
parser.add_argument('--rand_margin', type=float, default=10)
parser.add_argument('--num_iter_per_epoch', type=int, default=30)
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


# initialize CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    # Seed
    torch.manual_seed(args.seed)
    # create checkpoint folder
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    # load datasets
    data_train = pandas.read_csv(args.train_dataset, header=0)
    X_train = data_train.filename.tolist()
    y_train = [''.join(row[1:].astype('str').tolist()) for row in data_train.values]

    data_val = pandas.read_csv(args.val_dataset, header=0)
    X_val = data_val.filename.tolist()
    y_val = [''.join(row[1:].astype('str').tolist()) for row in data_val.values]
    train_dataset = Data(X_train, y_train, args.rand_margin)
    val_dataset = TestData(X_val, y_val)
    train_sampler = LabelSampler(train_dataset, batch_size=args.batch_size, weights=[.2, .1, .1, .06, .08, .2, .2, 0.06])
    kwargs = {'num_workers': args.num_workers, 'shuffle': False, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                               worker_init_fn=worker_init_fn, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, **kwargs)
    # Define models
    model = modelIQA(args)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']

    # set model evaluation metrics
    t_loss_meter = AverageMeter()
    v_loss_meter = AverageMeter()
    t_accuracy_meter = AverageMeter()
    v_accuracy_meter = AverageMeter()
    t_f1_micro_meter = AverageMeter()
    v_f1_micro_meter = AverageMeter()
    t_f1_macro_meter = AverageMeter()
    v_f1_macro_meter = AverageMeter()
    t_loss_meter.reset()
    v_loss_meter.reset()
    t_accuracy_meter.reset()
    v_accuracy_meter.reset()
    t_f1_micro_meter.reset()
    v_f1_micro_meter.reset()
    t_f1_macro_meter.reset()
    v_f1_macro_meter.reset()
    
    # create training_report.csv to save the evaluation metrics
    report_file = os.path.join(args.checkpoint_folder, 'training_report.csv')
    if not args.resume:
        with open(report_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ['epoch', 't_loss_meter.avg', 'v_loss_meter.avg', 't_accuracy_meter.avg', 'v_accuracy_meter.av',
                 't_f1_micro_meter.avg', 'v_f1_micro_meter.avg', 't_f1_macro_meter.avg', 'v_f1_macro_meter.avg'])


    # training
    for epoch in range(start_epoch, args.n_epoch):
        np.random.seed(epoch)

        # train model with training set
        model.train(args, train_loader, epoch, t_loss_meter, t_accuracy_meter, t_f1_micro_meter, t_f1_macro_meter)
        # evaluate model with validation set
        model.val(val_loader, v_loss_meter, v_accuracy_meter, v_f1_micro_meter, v_f1_macro_meter)
        # save evaluation metrics
        with open(report_file, 'a+', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [epoch, t_loss_meter.avg, v_loss_meter.avg, t_accuracy_meter.avg, v_accuracy_meter.avg,
                 t_f1_micro_meter.avg, v_f1_micro_meter.avg, t_f1_macro_meter.avg, v_f1_macro_meter.avg])
        # reset evaluation metrics for next epoch
        t_loss_meter.reset()
        t_accuracy_meter.reset()
        t_f1_micro_meter.reset()
        t_f1_macro_meter.reset()
        v_loss_meter.reset()
        v_accuracy_meter.reset()
        v_f1_micro_meter.reset()
        v_f1_macro_meter.reset()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterList(object):
    """Computes and stores the average and current value"""

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.reset()

    def reset(self):
        self.val = [0] * self.nclasses
        self.avg = [0] * self.nclasses
        self.sum = [0] * self.nclasses
        self.count = [0] * self.nclasses

    def update(self, val, n):
        for i in range(self.nclasses):
            if n[i] > 0:
                self.val[i] = val[i]
                self.sum[i] += val[i] * n[i]
                self.count[i] += n[i]
                self.avg[i] = self.sum[i] / self.count[i]


if __name__ == "__main__":
    main()
