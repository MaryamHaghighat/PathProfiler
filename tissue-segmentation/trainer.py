import os
import time
import glob
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from unet import UNet
from datasets import SegDataset
import lovasz_losses as L

from visdom import Visdom

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_folder', default='checkpoints', type=str,
                    help='directory to output weights')
parser.add_argument('--freq_checkpoint', default=1000, type=int,
                    help='how often to save model weights')
parser.add_argument('--dir', default='training_data', type=str,
                    help='directory of training and validation data')
parser.add_argument('--resume', default='checkpoint_147800.pth', type=str,
                    help='if provided, resume training from a given model path')

# optimiser
parser.add_argument('--epochs', default=20001, type=int,
                    help='total number of training epochs')
parser.add_argument('--batch_size', default=4, type=int,
                    help='training batch size')
parser.add_argument('--learning_rate', default=0.0002, type=float,
                    help='learning rate')

# gpu
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--workers', default=6, type=int,
                    help='number of workers for dataloader')

# verbose
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='to visualise training progress or not')
parser.add_argument('--freq_vis', default=10, type=int,
                    help='how often to display the training progress')

args = parser.parse_args()

if args.visdom:
    viz = Visdom()
    viz.close()
    win_input = viz.image(np.random.rand(3, 512, 512))
    win_output = viz.image(np.random.rand(3, 512, 512))
    win_predict = viz.image(np.random.rand(3, 512, 512))

    win_input_val = viz.image(np.random.rand(3, 512, 512))
    win_output_val = viz.image(np.random.rand(3, 512, 512))
    win_predict_val = viz.image(np.random.rand(3, 512, 512))

    win_loss = viz.line(np.random.rand(10, 2))


def main():
    # create a checkpoint folder if not exists
    os.makedirs(args.checkpoint_folder, exist_ok=True)

    # initialize CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    #######################################################################################################
    # create datasets and dataset loaders
    train_dataset = SegDataset(args.dir, mode='train')
    val_dataset = SegDataset(args.dir, mode='val')
    print('train data %d' % (len(train_dataset)))
    print('val data %d' % (len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False)

    # create model
    net = UNet()
    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    # create an optimizer and loss criterion
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # optionally resume from a checkpoint
    start_iter = 0
    epoch = 0
    train_loss = []
    val_loss = []

    t_loss_meter = AverageMeter()
    t_loss_meter.reset()

    v_loss_meter = AverageMeter()
    v_loss_meter.reset()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter']

            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)

    for iter_ in range(start_iter, args.epochs):
        if args.visdom and iter_ % args.freq_vis == 0:
            visualise = True
        else:
            visualise = False

        try:
            train_input, train_gt, train_name = next(train_loader_iter)
        except:
            # make sure that latest data is frequently loaded
            train_dataset = SegDataset(args.dir, mode='train')
            print('train data %d' % (len(train_dataset)))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers,
                                                       shuffle=True)

            train_loader_iter = iter(train_loader)
            train_input, train_gt, train_name = next(train_loader_iter)

            epoch += 1

        # training and validation loop
        train(train_input, train_gt, net, optimizer, t_loss_meter, iter_, visualise)

        try:
            val_input, val_gt, val_name = next(val_loader_iter)
        except:
            val_dataset = SegDataset(args.dir, mode='val')
            print('val data %d' % (len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=args.batch_size,
                                                     num_workers=args.workers,
                                                     shuffle=False)

            val_loader_iter = iter(val_loader)
            val_input, val_gt, val_name = next(val_loader_iter)
        validate(val_input, val_gt, net, v_loss_meter, iter_, visualise)

        # save checkpoint
        if iter_ % args.freq_checkpoint == 0:

            train_loss.append(t_loss_meter.avg)
            val_loss.append(v_loss_meter.avg)

            save_name = os.path.join(args.checkpoint_folder,
                                     'checkpoint_' + str(iter_) + '.pth')
            save_variable_name = os.path.join(args.checkpoint_folder,
                                              'variables.h5')

            save_checkpoint({
                'iter': iter_ + 1,
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss},
                {'train_loss': np.array(train_loss),
                 'val_loss': np.array(val_loss)},
                filename=save_name,
                variable_name=save_variable_name)

            t_loss_meter.reset()
            v_loss_meter.reset()

            # visualise the plot
            if visualise:
                viz.line(np.vstack([train_loss, val_loss]).T, opts=dict(title='loss'), win=win_loss)


def train(input, label,
          net, optimizer,
          loss_meter, iter_, visualise):

    t = time.time()

    # switch to train mode
    net.train()

    input_var = input.cuda()
    label_var = label.cuda()

    # set the grad to zero
    optimizer.zero_grad()

    # run the model
    predict = net(input_var)

    # calculate loss
    out = F.softmax(predict, dim=1)
    loss = L.lovasz_softmax(out, label_var)

    # backward and optimizer
    loss.backward()
    optimizer.step()

    # record loss
    loss_meter.update(loss.item(), input.size(0))

    duration = time.time() - t

    train_print_str = 'iter: %d ' \
                      'train_loss: %.3f ' \
                      'train_time: %.3f ' 

    print(train_print_str % (iter_, loss.item(), duration))

    if visualise:
        viz.image((input[0].cpu() + 1.0) / 2.0, opts=dict(title='input train'), win=win_input)
        viz.image(label[0].cpu().type(torch.FloatTensor), opts=dict(title='label train'), win=win_output)
        _, indices = torch.max(predict[0].cpu().data, 0)
        viz.image(indices.type(torch.FloatTensor), opts=dict(title='predict train'), win=win_predict)


def validate(input, label, net, loss_meter, iter_, visualise):

    t = time.time()

    # switch to validation mode
    net.eval()

    input_var = input.cuda()
    label_var = label.cuda()

    predict = net(input_var)

    # calculate loss
    out = F.softmax(predict, dim=1)
    loss = L.lovasz_softmax(out, label_var)
    loss_meter.update(loss.item(), input.size(0))

    duration = time.time() - t

    val_print_str = 'iter: %d ' \
                    'val_loss: %.3f ' \
                    'val_time: %.3f'
    print(val_print_str % (iter_, loss.item(), duration))

    if visualise:
        viz.image((input[0].cpu() + 1.0) / 2.0, opts=dict(title='input val'), win=win_input_val)
        viz.image(label[0].cpu().type(torch.FloatTensor), opts=dict(title='label val'), win=win_output_val)
        _, indices = torch.max(predict[0].cpu().data, 0)
        viz.image(indices.type(torch.FloatTensor), opts=dict(title='predict val'), win=win_predict_val)


def save_checkpoint(state,
                    variables,
                    filename='checkpoint.pth.tar', variable_name='variables.pth'):
    torch.save(state, filename)
    torch.save(variables, variable_name)


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


if __name__ == '__main__':
    main()
