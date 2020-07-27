"""
Copyright (c) 2020 Bahareh Tolooshams

crsae model

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import utils


class DEA1D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(DEA1D, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.model_distribution = hyp["model_distribution"]

        self.relu = torch.nn.ReLU()
        if self.model_distribution == "binomial":
            self.sigmoid = torch.nn.Sigmoid()

        if H is None:
            H = torch.randn((self.num_conv, 1, self.dictionary_dim), device=self.device)
            H = F.normalize(H, p=2, dim=-1)
        self.register_parameter("H", torch.nn.Parameter(H))

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(self.get_param("H").data, p=2, dim=-1)

    def forward(self, x, mu=0):
        num_batches = x.shape[0]

        D_in = x.shape[-1]
        D_enc = F.conv1d(x, self.get_param("H"), stride=self.stride).shape[-1]

        self.lam = self.sigma * torch.sqrt(
            2 * torch.log(torch.zeros(1, device=self.device) + (self.num_conv * D_enc))
        )

        x_old = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        yk = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        x_new = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            H_yk_mu = (
                F.conv_transpose1d(yk, self.get_param("H"), stride=self.stride) + mu
            )
            if self.model_distribution == "gaussian":
                x_tilda = x - H_yk_mu
            elif self.model_distribution == "binomial":
                x_tilda = x - self.sigmoid(H_yk_mu)
            elif self.model_distribution == "poisson":
                x_tilda = x - torch.exp(H_yk_mu)

            x_new = (
                yk + F.conv1d(x_tilda, self.get_param("H"), stride=self.stride) / self.L
            )
            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(
                    x_new
                )
            else:
                x_new = self.relu(x_new - self.lam / self.L)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = F.conv_transpose1d(x_new, self.get_param("H"), stride=self.stride) + mu

        return z, x_new, self.lam


class DEA2Dfree(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(DEA2Dfree, self).__init__()

        self.T = hyp["num_iters"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.model_distribution = hyp["model_distribution"]
        self.single_bias = hyp["single_bias"]
        self.L = hyp["L"]
        self.peak = hyp["peak"]
        self.nonlin = None

        if self.single_bias:
            self.register_parameter(
                "b",
                torch.nn.Parameter(
                    torch.zeros((1, self.num_conv, 1, 1), device=self.device)
                    + hyp["lam"]
                ),
            )
        else:
            for t in range(self.T):
                self.register_parameter(
                    "b{}".format(t),
                    torch.nn.Parameter(
                        torch.zeros((1, self.num_conv, 1, 1), device=self.device)
                        + hyp["lam"]
                    ),
                )

        if hyp["nonlin"] == "ELU":
            self.nonlin = torch.nn.ELU(alpha=self.peak)
        elif hyp["nonlin"] == "LeakyReLU":
            self.nonlin = torch.nn.LeakyReLU()
        elif hyp["nonlin"] == "ReLU":
            self.nonlin = torch.nn.ReLU()
        elif hyp["nonlin"] == "SELU":
            self.nonlin = torch.nn.SELU()

        if self.model_distribution == "binomial":
            self.sigmoid = torch.nn.Sigmoid()

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )

        H = F.normalize(H, p="fro", dim=(-1, -2))

        if self.L is None:
            self.L = utils.conv_power_method(
                H,
                [512, 512],
                num_iters=200,
                stride=self.stride,
                model_distribution=self.model_distribution,
            )
            self.L *= 5
        else:
            self.L = torch.tensor(self.L, device=self.device).float()

        H /= torch.sqrt(self.L)

        We = torch.clone(H)
        Wd = torch.clone(H)

        self.register_parameter("H", torch.nn.Parameter(H))
        self.register_parameter("We", torch.nn.Parameter(We))
        self.register_parameter("Wd", torch.nn.Parameter(Wd))

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )
        self.get_param("We").data = F.normalize(
            self.get_param("We").data, p="fro", dim=(-1, -2)
        )
        self.get_param("Wd").data = F.normalize(
            self.get_param("Wd").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x, mu=0):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[3]

        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        for t in range(self.T):
            H_yk_mu = (
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride) + mu
            )
            if self.model_distribution == "gaussian":
                x_tilda = x_batched_padded - H_yk_mu
            elif self.model_distribution == "binomial":
                x_tilda = x_batched_padded - self.sigmoid(H_yk_mu)
            elif self.model_distribution == "poisson":
                x_tilda = x_batched_padded - torch.exp(H_yk_mu)
                if self.nonlin is not None:
                    x_tilda = self.nonlin(x_tilda)

            x_new = x_new + F.conv2d(x_tilda, self.get_param("We"), stride=self.stride)

            if self.single_bias:
                if self.twosided:
                    x_new = F.relu(torch.abs(x_new) - self.get_param("b")) * torch.sign(
                        x_new
                    )
                else:
                    x_new = F.relu(x_new - self.get_param("b"))
            else:
                if self.twosided:
                    x_new = F.relu(
                        torch.abs(x_new) - self.get_param("b{}".format(t))
                    ) * torch.sign(x_new)
                else:
                    x_new = F.relu(x_new - self.get_param("b{}".format(t)))

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("Wd"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, 0


class DEA2Dtied(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(DEA2Dtied, self).__init__()

        self.T = hyp["num_iters"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.model_distribution = hyp["model_distribution"]
        self.single_bias = hyp["single_bias"]
        self.L = hyp["L"]
        self.peak = hyp["peak"]
        self.nonlin = None

        if self.single_bias:
            self.register_parameter(
                "b",
                torch.nn.Parameter(
                    torch.zeros((1, self.num_conv, 1, 1), device=self.device)
                    + hyp["lam"]
                ),
            )
        else:
            for t in range(self.T):
                self.register_parameter(
                    "b{}".format(t),
                    torch.nn.Parameter(
                        torch.zeros((1, self.num_conv, 1, 1), device=self.device)
                        + hyp["lam"]
                    ),
                )

        if hyp["nonlin"] == "ELU":
            self.nonlin = torch.nn.ELU(alpha=self.peak)
        elif hyp["nonlin"] == "LeakyReLU":
            self.nonlin = torch.nn.LeakyReLU()
        elif hyp["nonlin"] == "ReLU":
            self.nonlin = torch.nn.ReLU()
        elif hyp["nonlin"] == "SELU":
            self.nonlin = torch.nn.SELU()

        if self.model_distribution == "binomial":
            self.sigmoid = torch.nn.Sigmoid()

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )

        H = F.normalize(H, p="fro", dim=(-1, -2))

        if self.L is None:
            self.L = utils.conv_power_method(
                H,
                [512, 512],
                num_iters=200,
                stride=self.stride,
                model_distribution=self.model_distribution,
            )
            self.L *= 5
        else:
            self.L = torch.tensor(self.L, device=self.device).float()

        H /= torch.sqrt(self.L)

        self.register_parameter("H", torch.nn.Parameter(H))

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x, mu=0):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[3]

        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        for t in range(self.T):
            H_yk_mu = (
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride) + mu
            )
            if self.model_distribution == "gaussian":
                x_tilda = x_batched_padded - H_yk_mu
            elif self.model_distribution == "binomial":
                x_tilda = x_batched_padded - self.sigmoid(H_yk_mu)
            elif self.model_distribution == "poisson":
                x_tilda = x_batched_padded - torch.exp(H_yk_mu)
                if self.nonlin is not None:
                    x_tilda = self.nonlin(x_tilda)

            x_new = x_new + F.conv2d(x_tilda, self.get_param("H"), stride=self.stride)

            if self.single_bias:
                if self.twosided:
                    x_new = F.relu(torch.abs(x_new) - self.get_param("b")) * torch.sign(
                        x_new
                    )
                else:
                    x_new = F.relu(x_new - self.get_param("b"))
            else:
                if self.twosided:
                    x_new = F.relu(
                        torch.abs(x_new) - self.get_param("b{}".format(t))
                    ) * torch.sign(x_new)
                else:
                    x_new = F.relu(x_new - self.get_param("b{}".format(t)))

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, 0
