import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.autograd import Variable
import os
import sys
import math
import numpy as np
import json
import multiprocessing


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def out_im(im):
    imc = torch.clamp(im, -.5, .5)
    return imc + .5



def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)





def _determine_shapes(model):
    in_shapes = model.module.get_in_shapes()
    i = 0
    j = 0
    shape_list = list()
    for key, _ in model.named_parameters():
        if "bottleneck" not in key or "weight" not in key:  # non conv-parameters
            shape_list.append(None)
            continue
        shape_list.append(tuple(in_shapes[j]))
        if i == 2:
            i = 0
            j += 1
        else:
            i += 1
    return shape_list


def _clipping_comp(param, key, coeff, input_shape, use_cuda):
    if "bottleneck" not in key or "weight" not in key:  # non conv-parameters
        return
    # compute SVD via FFT
    convKernel = param.data.cpu().numpy()
    input_shape = input_shape[1:]
    fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
    t_fft_coeff = np.transpose(fft_coeff)
    U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
    if np.max(D) > coeff:
        # first projection onto given norm ball
        Dclip = np.minimum(D, coeff)
        coeffsClipped = np.matmul(U * Dclip[..., None, :], V)
        convKernelClippedfull = np.fft.ifft2(coeffsClipped, axes=[0, 1]).real
        # 1) second projection back to kxk filter kernels
        # and transpose to undo previous transpose operation (used for batch SVD)
        kernelSize1, kernelSize2 = convKernel.shape[2:]
        convKernelClipped = np.transpose(convKernelClippedfull[:kernelSize1, :kernelSize2])
        # reset kernel (using in-place update)
        if use_cuda:
            param.data += torch.tensor(convKernelClipped).float().cuda() - param.data
        else:
            param.data += torch.tensor(convKernelClipped).float() - param.data
    return


def clip_conv_layer(model, coeff, use_cuda):
    shape_list = _determine_shapes(model)
    num_cores = multiprocessing.cpu_count()
    for (key, param), shape in zip(model.named_parameters(), shape_list):
        _clipping_comp(param, key, coeff, shape, use_cuda)
    return


def interpolate(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, out_bij, _ = model(inputs)
        loss = criterion(out, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model

    acc = 100. * correct.type(torch.FloatTensor) / float(total)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.4f%%" % (epoch, loss.data[0], acc), flush=True)

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.4f%%' % (acc), flush=True)
        state = {
            'model': model if use_cuda else model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/' + dataset + os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + fname + '.t7')
        best_acc = acc
    return best_acc


softmax = nn.Softmax(dim=1)
