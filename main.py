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
import visdom
import os
import sys
import time
import argparse
import pdb
import random
import json
from models.utils_cifar import train, test, std, mean, get_hms, interpolate
from models.conv_iResNet import conv_iResNet as iResNet


parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-norm', '--norm', dest='norm', action='store_true',
                    help='compute norms of conv operators')
parser.add_argument('-interpolate', '--interpolate', dest='interpolate', action='store_true', help='train iresnet')
parser.add_argument('-analysisTraceEst', '--analysisTraceEst', dest='analysisTraceEst', action='store_true',
                    help='analysis of trace estimation')
def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)


def anaylse_trace_estimation(model, testset, use_cuda, extension):
    # setup range for analysis
    numSamples = np.arange(10) * 10 + 1
    numIter = np.arange(10)
    # setup number of datapoints
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    # TODO change

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        # compute trace
        out_bij, p_z_g_y, trace, gt_trace = model(inputs[:, :, :8, :8],
                                                  exact_trace=True)
        trace = [t.cpu().numpy() for t in trace]
        np.save('gtTrace' + extension, gt_trace)
        np.save('estTrace' + extension, trace)
        return


def test_spec_norm(model, in_shapes, extension):
    i = 0
    j = 0
    params = [v for v in model.module.state_dict().keys() \
              if "bottleneck" and "weight" in v \
              and not "weight_u" in v \
              and not "weight_orig" in v \
              and not "bn1" in v and not "linear" in v]
    print(len(params))
    print(len(in_shapes))
    svs = []
    for param in params:
        if i == 0:
            input_shape = in_shapes[j]
        else:
            input_shape = in_shapes[j]
            input_shape[1] = int(input_shape[1] // 4)

        convKernel = model.module.state_dict()[param].cpu().numpy()
        input_shape = input_shape[2:]
        fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
        t_fft_coeff = np.transpose(fft_coeff)
        U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
        Dflat = np.sort(D.flatten())[::-1]
        print("Layer " + str(j) + " Singular Value " + str(Dflat[0]))
        svs.append(Dflat[0])
        if i == 2:
            i = 0
            j += 1
        else:
            i += 1
    np.save('singular_values' + extension, svs)
    return
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
def main():
    args = parser.parse_args()
    # load data
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)


    # load model
    model = iResNet(nBlocks=[7,7,7], nStrides=[1,1,1],
                                nChannels=[32,64,128], nClasses=10,
                                in_shape=[3,32,32],
                                coeff=0.9,
                                numTraceSamples=1,
                                numSeriesTerms=1,
                                n_power_iter = 5,
                                actnorm=(not True),
                                learn_prior=True,
                                nonlin="elu")
    init_batch = get_init_batch(trainloader, 1024)
    with torch.no_grad():
        model(init_batch, ignore_logdet=True)
        print("initialized")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
        cudnn.benchmark = True
        in_shapes = model.module.get_in_shapes()
    else:
        in_shapes = model.get_in_shapes()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_objective = checkpoint['objective']
            print('objective: ' + str(best_objective))
            model = checkpoint['model']
            if use_cuda:
                model.module.set_num_terms(args.numSeriesTerms)
            else:
                model.set_num_terms(args.numSeriesTerms)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    print("1")
    try_make_dir(args.save_dir)
    if args.analysisTraceEst:
        anaylse_trace_estimation(model, testset, use_cuda, args.extension)
        return

    if args.norm:
        test_spec_norm(model, in_shapes, args.extension)
        return

    if args.interpolate:
        interpolate(model, testloader, testset, start_epoch, use_cuda, best_objective, args.dataset)
        return
    print("here")
    if args.evaluate:
        test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
        if use_cuda:
            model.module.set_num_terms(args.numSeriesTerms)
        else:
            model.set_num_terms(args.numSeriesTerms)
        model = torch.nn.DataParallel(model.module)
        test(best_objective, args, model, start_epoch, testloader, use_cuda, test_log)
        return
    print("hehe")
    print('|  Train Epochs: ' + str(args.epochs))
    print('|  Initial Learning Rate: ' + str(args.lr))

    elapsed_time = 0
    test_objective = -np.inf


    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
        f.write(json.dumps(args.__dict__))

    train_log = open(os.path.join(args.save_dir, "train_log.txt"), 'w')

    for epoch in range(1, 1 + 200):
        start_time = time.time()
        # train
        train(args, model, optimizer, epoch, trainloader, trainset, use_cuda, train_log)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    print('Testing model')
    test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
    # test
    test_objective = test(test_objective, args, model, epoch, testloader, use_cuda, test_log)
    print('* Test results : objective = %.2f%%' % (test_objective))
    with open(os.path.join(args.save_dir, 'final.txt'), 'w') as f:
        f.write(str(test_objective))

if __name__ == '__main__':
    main()
