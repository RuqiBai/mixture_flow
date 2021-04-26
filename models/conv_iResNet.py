"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from .model_utils import injective_pad, ActNorm2D, Split
from .model_utils import squeeze as Squeeze
from .model_utils import MaxMinGroup
from spectral_norm_conv_inplace import spectral_norm_conv
from spectral_norm_fc import spectral_norm_fc
from matrix_utils import exact_matrix_logarithm_trace, power_series_matrix_logarithm_trace
from torch.distributions import constraints


class LogisticTransform(torch.distributions.Transform):
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """
    codomain = constraints.real
    domain = constraints.unit_interval
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, LogisticTransform)

    def _call(self, x):
        return x.log() - (-x).log1p()

    def _inverse(self, y):
        return torch.sigmoid(y)

    def log_abs_det_jacobian(self, x, y):
        return F.softplus(y) + F.softplus(-y)


def logistic_distribution(loc, log_scale):
    scale = torch.exp(log_scale) + 1e-5
    base_distribution = distributions.Uniform(torch.zeros_like(loc), torch.ones_like(loc))
    transforms = [LogisticTransform(), distributions.AffineTransform(loc=loc, scale=scale)]
    logistic = distributions.TransformedDistribution(base_distribution, transforms)
    return logistic


def downsample_shape(shape):
    return (shape[0] * 4, shape[1] // 2, shape[2] // 2)


class conv_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu"):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape

        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride ** 2
        kernel_size1 = 3  # kernel size for first conv
        layers.append(
            self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1),
                                        (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size2 = 1  # kernel size for second conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0),
                                                  (int_ch, h, w), kernel_size2))
        layers.append(nonlin())
        kernel_size3 = 3  # kernel size for third conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=1),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride == 2:
            x = self.squeeze.forward(x)

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x
        return y, trace + an_logdet

    def inverse(self, y, maxIter=100):
        # inversion of ResNet-block (fixed-point iteration)
        x = y
        for iter_index in range(maxIter):
            summand = self.bottleneck_block(x)
            x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        # inversion of squeeze (dimension shuffle)
        if self.stride == 2:
            x = self.squeeze.inverse(x)
        return x

    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff,
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)


class scale_block(nn.Module):
    def __init__(self, steps, in_shape, int_dim, squeeze=True, n_terms=0, n_samples=0,
                 coeff=.9, input_nonlin=True, actnorm=True, split=True,
                 n_power_iter=5, nonlin="relu"):
        super(scale_block, self).__init__()
        self.in_shape = in_shape
        if squeeze:
            self.squeeze = Squeeze(2)
            conv_shape = downsample_shape(in_shape)
        else:
            self.squeeze = None
            conv_shape = in_shape

        if split:
            self.split = Split()
            n = int(conv_shape[0] // 2)
            out_shape1 = (n, conv_shape[1], conv_shape[2])
            out_shape2 = (conv_shape[0] - n, conv_shape[1], conv_shape[2])
            self.out_shapes = [out_shape1, out_shape2]
        else:
            self.split = None
            self.out_shapes = [conv_shape]

        self.stack = self._make_stack(steps, n_terms, n_samples, conv_shape, int_dim,
                                      input_nonlin, coeff, actnorm, n_power_iter, nonlin)

    @staticmethod
    def _make_stack(steps, n_terms, n_samples, in_shape, int_dim,
                    input_nonlin, coeff, actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        for i in range(steps):
            block_list.append(conv_iresnet_block(in_shape, int_dim, n_samples, n_terms,
                                                 stride=1, input_nonlin=True if input_nonlin else i > 0,
                                                 coeff=coeff, actnorm=actnorm,
                                                 n_power_iter=n_power_iter, nonlin=nonlin))

        return block_list

    def forward(self, x, ignore_logdet=False):
        if self.squeeze is not None:
            x = self.squeeze(x)

        traces = []
        z = x
        for block in self.stack:
            z, trace = block(z, ignore_logdet=ignore_logdet)
            traces.append(trace)

        trace = torch.zeros_like(traces[0])
        for k in range(len(traces)):
            trace += traces[k]

        if self.split is None:
            return [z], trace
        else:
            z1, z2 = self.split(z)
            return [z1, z2], trace

    def inverse(self, z, z2=None, maxIter=100):
        if self.split is None:
            x = z
        else:
            assert z2 is not None
            x = self.split.inverse(z, z2)

        for block in reversed(self.stack):
            x = block.inverse(x, maxIter=maxIter)

        if self.squeeze is None:
            return x
        else:
            return self.squeeze.inverse(x)


class conv_iResNet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels,
                 coeff=.9, nClasses=10,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 block=conv_iresnet_block,
                 actnorm=True, learn_prior=True,
                 nonlin="relu"):
        super(conv_iResNet, self).__init__()
        assert len(nBlocks) == len(nStrides) == len(nChannels)
        self.nBlocks = nBlocks
        self.nClasses = nClasses
        # parameters for trace estimation
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter

        print('')
        print(' == Building iResNet %d == ' % (sum(nBlocks) * 3 + 1))


        self.stack, self.in_shapes, self.final_shape = self._make_stack(nChannels, nBlocks, nStrides,
                                                                        in_shape, coeff, block,
                                                                        actnorm, n_power_iter, nonlin)
        # make prior distribution
        self._make_prior(learn_prior)

    def _make_prior(self, learn_prior):
        dim = np.prod(self.in_shapes[0])
        self.prior_mu = nn.Parameter(torch.zeros((self.nClasses, dim)).float(), requires_grad=learn_prior)
        self.prior_logstd = nn.Parameter(torch.zeros((self.nClasses, dim)).float(), requires_grad=learn_prior)
    
    def prior(self, label):
        return distributions.Normal(self.prior_mu[label], torch.exp(self.prior_logstd[label]))

    def logpz(self, z, label):
        return self.prior(label).log_prob(z.view(z.size(0), -1)).sum(dim=1)

    def _make_stack(self, nChannels, nBlocks, nStrides, in_shape, coeff, block,
                    actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        in_shapes = []
        for i, (int_dim, stride, blocks) in enumerate(zip(nChannels, nStrides, nBlocks)):
            for j in range(blocks):
                in_shapes.append(in_shape)
                block_list.append(block(in_shape, int_dim,
                                        numTraceSamples=self.numTraceSamples,
                                        numSeriesTerms=self.numSeriesTerms,
                                        stride=(stride if j == 0 else 1),  # use stride if first layer in block else 1
                                        input_nonlin=(i + j > 0),  # add nonlinearity to input for all but fist layer
                                        coeff=coeff,
                                        actnorm=actnorm,
                                        n_power_iter=n_power_iter,
                                        nonlin=nonlin))
                if stride == 2 and j == 0:
                    in_shape = downsample_shape(in_shape)

        return block_list, in_shapes, in_shape

    def get_in_shapes(self):
        return self.in_shapes

    def inspect_singular_values(self):
        i = 0
        j = 0
        params = [v for v in self.state_dict().keys()
                  if "bottleneck" in v and "weight_orig" in v
                  and not "weight_u" in v
                  and not "bn1" in v
                  and not "linear" in v]
        print(len(params))
        print(len(self.in_shapes))
        svs = []
        for param in params:
            input_shape = tuple(self.in_shapes[j])
            # get unscaled parameters from state dict
            convKernel_unscaled = self.state_dict()[param].cpu().numpy()
            # get scaling by spectral norm
            sigma = self.state_dict()[param[:-5] + '_sigma'].cpu().numpy()
            convKernel = convKernel_unscaled / sigma
            # compute singular values
            input_shape = input_shape[1:]
            fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
            t_fft_coeff = np.transpose(fft_coeff)
            D = np.linalg.svd(t_fft_coeff, compute_uv=False, full_matrices=False)
            Dflat = np.sort(D.flatten())[::-1]
            print("Layer " + str(j) + " Singular Value " + str(Dflat[0]))
            svs.append(Dflat[0])
            if i == 2:
                i = 0
                j += 1
            else:
                i += 1
        return svs

    def forward(self, x, ignore_logdet=False):
        """ iresnet forward """

        z = x
        traces = []
        for block in self.stack:
            z, trace = block(z, ignore_logdet=ignore_logdet)
            traces.append(trace)
        tmp_trace = torch.zeros_like(traces[0])
        for k in range(len(traces)):
            tmp_trace += traces[k]
        return z, tmp_trace

    def inverse(self, z, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            x = z
            for i in range(len(self.stack)):
                x = self.stack[-1 - i].inverse(x, maxIter=max_iter)
        return x

    def classify(self, x):
        z, trace = self(x)
        max_logpx = None
        for i in range(self.nClasses):
            logpz = self.logpz(z, i)
            logpx = logpz + trace
            if max_logpx is None:
                max_logpx = logpx
                max_likelihood = torch.zeros_like(logpx)
            else:
                mask = logpx.gt(max_logpx)
                # print(logpx)
                # print(max_logpx)
                # print(mask)
                max_logpx = logpx * mask + max_logpx * (mask.logical_not())
                # print(max_logpx)
                max_likelihood = i * mask + max_likelihood * (mask.logical_not())


        return max_likelihood

    def sample(self, batch_size, label, max_iter=10):
        """sample from prior and invert"""
        with torch.no_grad():
            # only send batch_size to prior, prior has final_shape as attribute
            samples = self.prior(label=label).rsample((batch_size,))
            samples = samples.view([batch_size,] + self.final_shape)
            return self.inverse(samples, max_iter=max_iter)

    def set_num_terms(self, n_terms):
        for block in self.stack:
            for layer in block.stack:
                layer.numSeriesTerms = n_terms


def iResNet64():
    return conv_iResNet(nBlocks=[7,7,7], nStrides=[1,1,1],
                                nChannels=[32,64,128], nClasses=10,
                                in_shape=[3,32,32],
                                coeff=0.9,
                                numTraceSamples=1,
                                numSeriesTerms=1,
                                n_power_iter = 5,
                                actnorm=(not True),
                                learn_prior=True,
                                nonlin="elu")
if __name__ == "__main__":
    scale = 1.
    loc = 0.
    base_distribution = distributions.Uniform(0., 1.)
    transforms_1 = [distributions.SigmoidTransform().inv, distributions.AffineTransform(loc=loc, scale=scale)]
    logistic_1 = distributions.TransformedDistribution(base_distribution, transforms_1)

    transforms_2 = [LogisticTransform(), distributions.AffineTransform(loc=loc, scale=scale)]
    logistic_2 = distributions.TransformedDistribution(base_distribution, transforms_2)

    x = torch.zeros(2)
    print(logistic_1.log_prob(x), logistic_2.log_prob(x))
    1 / 0

    diff = lambda x, y: (x - y).abs().sum()
    batch_size = 13
    channels = 3
    h, w = 32, 32
    in_shape = (batch_size, channels, h, w)
    x = torch.randn((batch_size, channels, h, w), requires_grad=True)

    block = conv_iresnet_block(in_shape[1:], 32, stride=1, actnorm=True)
    out, tr = block(x)  # , ignore_logdet=True)
    print("block")
    for i in range(10):
        x_re = block.inverse(out, i)
        print(i, diff(x, x_re))

    steps = 4
    int_dim = 32
    sb = scale_block(steps, in_shape[1:], int_dim, True, 5, 1, True, .9, False, True, True)

    [z1, z2], tr = sb(x)  # , ignore_logdet=True)
    print("scale block")
    for i in range(1):
        x_re = sb.inverse(z1, z2, i)
        print(i, diff(x, x_re))

    resnet = conv_iResNet(in_shape[1:], [4, 4, 4], [1, 2, 2], [32, 32, 32],
                          init_ds=2, actnorm=True)
    print(resnet.final_shape)
    z, lpz, tr = resnet(x)  # , ignore_logdet=True)
    for i in range(1):
        x_re = resnet.inverse(z, i)
        print("{} iters error {}".format(i, (x - x_re).abs().sum()))

    out, logpz, tr = resnet(x)  # , ignore_logdet=True)
    print(logpz)
    print(tr)
    print([o.size() for o in out])
    print(resnet.z_shapes())
    sample = resnet.sample(33)
    print(sample.size())
    print('multiscale')
    for i in range(20):
        x_re = resnet.inverse(out, i)
        print(i, diff(x, x_re))
