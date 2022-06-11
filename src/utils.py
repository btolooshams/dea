"""
Copyright (c) 2020 Bahareh Tolooshams

utils

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np


class DEALoss1D(torch.nn.Module):
    def __init__(self, model_dist, loss_dist):
        super(DEALoss1D, self).__init__()

        self.model_dist = model_dist
        self.loss_dist = loss_dist

        self.a = 0.8
        self.win_size = 4
        self.data_range = 255
        self.channel = 1

    def forward(self, y, Hx, Q=1):
        if self.model_dist == "gaussian":
            loss = F.mse_loss(y, Hx, reduction="none")
        elif self.model_dist == "binomial":
            if self.loss_dist == "binomial":
                loss = -torch.mean(y * (Hx), dim=-1) + torch.mean(
                    torch.log1p(torch.exp(Hx)), dim=-1
                )
            elif self.loss_dist == "gaussian":
                loss = F.mse_loss(y, torch.nn.Sigmoid()(Hx), reduction="none")
            elif self.loss_dist == "ms_ssim":
                loss = self.a * (
                    1
                    - MS_SSIM(
                        win_size=self.win_size,
                        data_range=self.data_range,
                        channel=self.channel,
                    )(y, torch.nn.Sigmoid()(Hx))
                ) + (1 - self.a) * torch.nn.L1Loss()(y, torch.nn.Sigmoid()(Hx))
            else:
                print("ERROR: the loss is not implemented!")
        elif self.model_dist == "poisson":
            if self.loss_dist == "poisson":
                loss = -torch.mean(y * (Hx), dim=-1) + torch.mean(
                    torch.exp(Hx), dim=-1
                )
            elif self.loss_dist == "gaussian":
                loss = F.mse_loss(y, Q * torch.exp(Hx), reduction="none")
            elif self.loss_dist == "ms_ssim":
                loss = self.a * (
                    1
                    - MS_SSIM(
                        win_size=self.win_size,
                        data_range=self.data_range,
                        channel=self.channel,
                    )(y, Q * torch.exp(Hx))
                ) + (1 - self.a) * torch.nn.L1Loss()(y, Q * torch.exp(Hx))
            else:
                print("ERROR: the loss is not implemented!")
        return torch.mean(loss)

class DEALoss2D(torch.nn.Module):
    def __init__(self, model_dist, loss_dist):
        super(DEALoss2D, self).__init__()

        self.model_dist = model_dist
        self.loss_dist = loss_dist

        self.a = 0.8
        self.win_size = 4
        self.data_range = 255
        self.channel = 1

    def forward(self, y, Hx, Q=1):
        if self.model_dist == "gaussian":
            loss = F.mse_loss(y, Hx, reduction="none")
        elif self.model_dist == "binomial":
            if self.loss_dist == "binomial":
                loss = -torch.mean(y * (Hx), dim=(-1, -2)) + torch.mean(
                    torch.log1p(torch.exp(Hx)), dim=(-1, -2)
                )
            elif self.loss_dist == "gaussian":
                loss = F.mse_loss(y, torch.nn.Sigmoid()(Hx), reduction="none")
            elif self.loss_dist == "ms_ssim":
                loss = self.a * (
                    1
                    - MS_SSIM(
                        win_size=self.win_size,
                        data_range=self.data_range,
                        channel=self.channel,
                    )(y, torch.nn.Sigmoid()(Hx))
                ) + (1 - self.a) * torch.nn.L1Loss()(y, torch.nn.Sigmoid()(Hx))
            else:
                print("ERROR: the loss is not implemented!")
        elif self.model_dist == "poisson":
            if self.loss_dist == "poisson":
                loss = -torch.mean(y * (Hx), dim=(-1, -2)) + torch.mean(
                    torch.exp(Hx), dim=(-1, -2)
                )
            elif self.loss_dist == "gaussian":
                loss = F.mse_loss(y, Q * torch.exp(Hx), reduction="none")
            elif self.loss_dist == "ms_ssim":
                loss = self.a * (
                    1
                    - MS_SSIM(
                        win_size=self.win_size,
                        data_range=self.data_range,
                        channel=self.channel,
                    )(y, Q * torch.exp(Hx))
                ) + (1 - self.a) * torch.nn.L1Loss()(y, Q * torch.exp(Hx))
            else:
                print("ERROR: the loss is not implemented!")
        return torch.mean(loss)

def normalize1d(x):
    return F.normalize(x, dim=-1)


def normalize2d(x):
    return F.normalize(x, dim=(-1, -2))


def err1d_H(H, H_hat):

    H = H.detach().cpu().numpy()
    H_hat = H_hat.detach().cpu().numpy()

    num_conv = H.shape[0]

    err = []
    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :] * H_hat[conv, 0, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def err2d_H(H, H_hat):

    H = H.clone().detach().cpu().numpy()
    H_hat = H_hat.clone().detach().cpu().numpy()

    num_conv = H.shape[0]

    err = []

    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :, :] * H_hat[conv, 0, :, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def PSNR(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    max_x = np.max(x)
    return 20 * np.log10(max_x) - 10 * np.log10(mse)


def calc_pad_sizes(x, dictionary_dim=8, stride=1):
    left_pad = stride
    right_pad = (
        0
        if (x.shape[3] + left_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[3] + left_pad - dictionary_dim) % stride)
    )
    top_pad = stride
    bot_pad = (
        0
        if (x.shape[2] + top_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[2] + top_pad - dictionary_dim) % stride)
    )
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad


def conv_power_method(
    D, image_size, num_iters=100, stride=1, model_distribution="gaussian"
):
    needles_shape = [
        int(((image_size[0] - D.shape[-2]) / stride) + 1),
        int(((image_size[1] - D.shape[-1]) / stride) + 1),
    ]
    x = torch.randn(1, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = F.conv_transpose2d(x, D, stride=stride)
        if model_distribution == "binomial":
            y = torch.sigmoid(y)
        elif model_distribution == "poisson":
            y = torch.exp(y)
        x = F.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))
