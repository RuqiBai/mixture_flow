import torch


import torchvision
import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import pdb
import random
import json
from models.conv_iResNet import iResNet64


parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-norm', '--norm', dest='norm', action='store_true',
                    help='compute norms of conv operators')
parser.add_argument('-interpolate', '--interpolate', dest='interpolate', action='store_true', help='train iresnet')
parser.add_argument('-analysisTraceEst', '--analysisTraceEst', dest='analysisTraceEst', action='store_true',
                    help='analysis of trace estimation')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--tuning', default=0.02, type=float, help='tuning parameter')
args = parser.parse_args()


def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init
    """
    batches = []
    seen = 0
    for x, y in dataloader:
        batches.append(x)
        seen += x.size(0)
        if seen >= batch_size:
            break
    batch = torch.cat(batches)
    return batch


def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.


def main(args):
    def train(epoch, model):
        model.train()
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print('|  Number of Trainable Parameters: ' + str(params))
        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, scheduler.get_last_lr()[0]))

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)  # GPU settings
            inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
            optimizer.zero_grad()
            z, trace = model(inputs)  # Forward Propagation
            loss = 0.
            for i in range(10):
                mask = (targets.eq(i).double() * (args.tuning + 1) - args.tuning)

                if device == 'cuda':
                    loss -= ((model.module.logpz(z, torch.ones_like(targets) * i) + trace) * mask).mean()
                else:
                    loss -= ((model.logpz(z, torch.ones_like(targets) * i) + trace) * mask).mean()
            logpx = (model.module.logpz(z, targets) + trace).mean()
            loss.backward()  # Backward Propagation
            optimizer.step()  # Optimizer update
            print(batch_idx, round(loss.item(), 3), round(bits_per_dim(logpx, inputs).item(), 3), end='\r',flush=True)

    def classify(model, device, testloader):
        model.eval()
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
            if device == 'cuda':
                predicted = model.module.classify(inputs)
            else:
                predicted = model.classify(inputs)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return 1. * correct / total
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = iResNet64()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    # load data
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_objective = checkpoint['objective']
            print('objective: ' + str(best_objective))
            model = checkpoint['model']
            if device == 'cuda':
                model.module.set_num_terms(args.numSeriesTerms)
            else:
                model.set_num_terms(args.numSeriesTerms)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    print('|  Train Epochs: ' + str(args.epochs))
    print('|  Initial Learning Rate: ' + str(args.lr))

    elapsed_time = 0
    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.2)
    for epoch in range(1, 1 + args.epochs):
        # train
        train(epoch, model)
        scheduler.step()
        # test
        acc = classify(model, device, testloader)
        print('* Test results : objective = %.2f%%' % (100. * acc))
        if acc > best_acc:
            print('Saving..')
            state = {
                'model': model.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_normalization.pth')
            best_acc = acc


if __name__ == '__main__':
    main(args)
