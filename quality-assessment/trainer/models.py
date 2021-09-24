import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import os


def split_scale(item, scale):
    return [float(ch)*scale for ch in item]


def binary_target(a):
    return [0. if ch == '0' else 1. for ch in a]


class ResNet18(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.down = nn.Linear(512, n_classes)

    def embedding(self, x):
        x = self.resnet(x)
        x = F.relu(x.view(x.size(0), x.size(1)))
        x = self.down(x)
        return x


class modelIQA:
    def __init__(self, args):

        # Hyper Parameters
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.n_epoch = args.n_epoch
        self.model = ResNet18(n_classes=args.n_classes)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=args.lr)
        # resume
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                loc = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
                checkpoint = torch.load(args.resume, map_location=loc)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.5, patience=5, verbose=True)
        if torch.cuda.is_available():
            self.model.cuda()
        self.loss_fn = args.loss_fn
        if self.loss_fn == 'BCE':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif self.loss_fn == 'MSE':
            self.loss_func = nn.MSELoss()
        elif self.loss_fn == 'huber':
            self.loss_func = nn.SmoothL1Loss()

    # Train the Model
    def train(self, args, train_loader, epoch, loss_meter, accuracy_meter, f1_micro_meter, f1_macro_meter):
        print('Training ...')
        self.model.train()  # Change model to 'train' mode.
        for i, (data, target) in enumerate(train_loader):
            if i > self.num_iter_per_epoch:
                break
            # convert string labels to binary multilabels
            target_bin = torch.tensor([binary_target(item) for item in target])
            # convert string labels to normalized multilabels (i.e., 0, 0.5, 1)
            target = torch.tensor([split_scale(item, .5) for item in target])
            # set thresholds to calculate the evaluation metrics accuracy, f1 score, etc
            thresholds = .5 * torch.ones(target.size())
            thresholds[:, 2:4] = .25

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target_bin = target_bin.cuda()
                thresholds = thresholds.cuda()

            # Forward + Backward + Optimize
            logits = self.model.embedding(data)
            loss = self.loss_func(logits, target)
            loss_meter.update(loss.item(), target.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outputs = nn.Sigmoid()(logits) if self.loss_fn == 'BCE' else logits
            # binarize outputs to calculate evaluation metrics
            outputs[outputs >= thresholds] = 1
            outputs[outputs < thresholds] = 0
            accuracy = (outputs == target_bin).sum().item() / (target.size(0) * target.size(1))
            f1_micro = f1_score(target_bin.cpu().data.numpy(), outputs.cpu().data.numpy(), average='micro')
            f1_macro = f1_score(target_bin.cpu().data.numpy(), outputs.cpu().data.numpy(), average='macro')
            accuracy_meter.update(accuracy, (target.size(0) * target.size(1)))
            f1_micro_meter.update(f1_micro, (target.size(0) * target.size(1)))
            f1_macro_meter.update(f1_macro, (target.size(0) * target.size(1)))

            if i % 10 == 0:
                print('iter: %d, train loss: %.3f' % (i, loss_meter.avg))
                print('iter: %d, train accuracy: %.3f' % (i, accuracy_meter.avg))

        self.scheduler.step(loss_meter.avg)
        # save checkpoint
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        save_name = os.path.join(args.checkpoint_folder,  'checkpoint_' + str(epoch) + '.pth')
        torch.save(state, save_name)

    def val(self, val_loader, loss_meter, accuracy_meter, f1_micro_meter, f1_macro_meter):
        with torch.no_grad():
            self.model.eval()

        for i, (data, target) in enumerate(val_loader):
            if i > self.num_iter_per_epoch:
                break
            # convert string labels to binary multilabels
            target_bin = torch.tensor([binary_target(item) for item in target])
            # convert string labels to normalized multilabels (i.e., 0, 0.5, 1)
            target = torch.tensor([split_scale(item, .5) for item in target])
            # set thresholds to calculate the evaluation metrics accuracy, f1 score, etc
            thresholds = .5 * torch.ones(target.size())
            thresholds[:, 2:4] = .25

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target_bin = target_bin.cuda()
                thresholds = thresholds.cuda()

            # Forward pass
            logits = self.model.embedding(data)
            loss = self.loss_func(logits, target)
            loss_meter.update(loss.item(), target.size(0))
            outputs = nn.Sigmoid()(logits) if self.loss_fn == 'BCE' else logits
            # binarize outputs to calculate evaluation metrics
            outputs[outputs >= thresholds] = 1
            outputs[outputs < thresholds] = 0
            accuracy = (outputs == target_bin).sum().item() / (target.size(0) * target.size(1))
            f1_micro = f1_score(target_bin.cpu().data.numpy(), outputs.cpu().data.numpy(), average='micro')
            f1_macro = f1_score(target_bin.cpu().data.numpy(), outputs.cpu().data.numpy(), average='macro')
            accuracy_meter.update(accuracy, (target.size(0) * target.size(1)))
            f1_micro_meter.update(f1_micro, (target.size(0) * target.size(1)))
            f1_macro_meter.update(f1_macro, (target.size(0) * target.size(1)))
            print('iter: %d, val loss: %.3f' % (i, loss_meter.avg))
            print('iter: %d, val accuracy: %.3f' % (i, accuracy_meter.avg))



