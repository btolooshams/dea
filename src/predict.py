"""
Copyright (c) 2020 Bahareh Tolooshams

predict

:author: Bahareh Tolooshams
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.gridspec as gridspec

from datetime import datetime
from sacred import Experiment

import sys

sys.path.append("src/")

import model, generator, trainer, utils

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("predict")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@ex.automain
def predict():

    random_date = "2020_02_03_22_18_02"
    num_epochs = 399

    test_path = "../data/test_img/"
    img_list = ["man", "couple", "boat", "bridge", "cameraman", "house", "peppers"]

    # test_path = "../data/Set12/"
    # test_path = "../data/BSD68/"

    hyp = {
        "experiment_name": "poisson_ista15_stride7_peak1_11x11_169_onesided_tied_random_elu_smllr",
        "dataset": "VOC",
        "year": "2012",
        "segmentation": False,
        "image_set": "train",
        "network": "DEA2Dtied",
        "data_distribution": "poisson",
        "model_distribution": "poisson",
        "loss_distribution": "gaussian",
        "dictionary_dim": 11,
        "stride": 7,
        "num_conv": 169,
        "peak": 1,
        "L": None,
        "num_iters": 15,
        "twosided": False,
        "batch_size": 1,
        "num_epochs": 400,
        "zero_mean_filters": False,
        "normalize": False,
        "lr": 1e-4,
        "lr_decay": 0.8,
        "lr_step": 25,
        "cyclic": False,
        "amsgrad": False,
        "info_period": 7000,
        "lam": 0.01,
        "nonlin": "ELU",
        "single_bias": True,
        "shuffle": True,
        "crop_dim": (128, 128),
        "init_with_DCT": False,
        "init_with_saved_file": False,
        "test_path": "../data/test_img/",
        "denoising": True,
        "mu": 0.0,
        "supervised": True,
    }

    PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)

    enable_plot = 0
    T = 20
    mu = 0

    if test_path == "../data/test_img/":
        num_test = len(img_list)
        c = ["k", "b", "r", "g", "purple", "orange", "y", "m", "c"]

    loss = torch.zeros(num_epochs, device=device)
    psnr = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        loss[epoch] = torch.tensor(
            torch.load(os.path.join(PATH, "loss_epoch{}.pt".format(epoch)), device)
        )
        psnr[epoch] = np.load(os.path.join(PATH, "psnr_epoch{}.npy".format(epoch)))[0]

    best_epoch = torch.argmin(loss)
    print("best_epoch", best_epoch)
    best_epoch = num_epochs

    loss = loss.detach().cpu().numpy()
    plt.figure(), plt.plot(loss), plt.xlabel("epochs"), plt.ylabel("loss")
    plt.savefig(os.path.join(PATH, "loss.png"))

    plt.figure(), plt.plot(psnr), plt.xlabel("epochs"), plt.ylabel("psnr")
    plt.savefig(os.path.join(PATH, "psnr.png"))

    print(hyp["experiment_name"])
    print(random_date)

    print("load data.")
    test_loader = generator.get_path_loader(1, test_path, shuffle=False)

    print("create model.")
    net_init = torch.load(os.path.join(PATH, "model_init.pt"))
    net = torch.load(os.path.join(PATH, "model_epoch{}.pt".format(best_epoch)))

    if hyp["data_distribution"] == "binomial":
        Jg = hyp["num_trials"]

    with torch.no_grad():

        ctr = 0
        PSNR_noisy_list_all = list()
        PSNR_init_list_all = list()
        PSNR_list_all = list()

        for img_org, tar in test_loader:

            PSNR_noisy_list = list()
            PSNR_init_list = list()
            PSNR_list = list()
            for t in range(T):
                if hyp["data_distribution"] == "gaussian":
                    img_noisy = (
                        img_org + hyp["noiseSTD"] / 255 * torch.randn(img_org.shape)
                    ).to(device)
                    img = img_org.to(device)
                elif hyp["data_distribution"] == "binomial":
                    img = img_org.to(device)
                    sampler = torch.distributions.bernoulli.Bernoulli(probs=img)
                    img_noisy = sampler.sample()
                    for j in range(Jg - 1):
                        img_noisy += sampler.sample()
                    img_noisy /= Jg
                elif hyp["data_distribution"] == "poisson":
                    img = img_org.to(device)
                    Q = torch.max(img) / hyp["peak"]
                    rate = img / Q
                    if torch.isnan(torch.min(rate)):
                        continue
                    sampler = torch.distributions.poisson.Poisson(rate)
                    if hyp["model_distribution"] == "poisson":
                        img_noisy = sampler.sample()
                    else:
                        img_noisy = sampler.sample() * Q

                Hx_hat, _, _ = net_init(img_noisy, mu)

                if hyp["model_distribution"] == "gaussian":
                    img_init_hat = Hx_hat + mu
                elif hyp["model_distribution"] == "binomial":
                    img_init_hat = torch.nn.Sigmoid()(Hx_hat + mu)
                elif hyp["model_distribution"] == "poisson":
                    img_init_hat = torch.exp(Hx_hat + mu) * Q

                img_init_hat = img_init_hat[0, 0, :, :].detach().cpu().numpy()

                Hx_hat, _, _ = net(img_noisy, mu)

                if hyp["model_distribution"] == "gaussian":
                    img_hat = Hx_hat + mu
                elif hyp["model_distribution"] == "binomial":
                    img_hat = torch.nn.Sigmoid()(Hx_hat + mu)
                elif hyp["model_distribution"] == "poisson":
                    img_hat = torch.exp(Hx_hat + mu) * Q
                else:
                    print("data distribution not exist!")

                if t == 0:
                    if test_path == "../data/test_img/":
                        torchvision.utils.save_image(
                            img,
                            os.path.join(PATH, "{}_clean.png".format(img_list[ctr])),
                        )
                        torchvision.utils.save_image(
                            img_noisy * Q,
                            os.path.join(PATH, "{}_noisy.png".format(img_list[ctr])),
                        )
                        torchvision.utils.save_image(
                            img_hat,
                            os.path.join(PATH, "{}_est.png".format(img_list[ctr])),
                        )

                img_hat = img_hat[0, 0, :, :].detach().cpu().numpy()

                img_noisy = img_noisy[0, 0, :, :].detach().cpu().numpy()
                img = img[0, 0, :, :].detach().cpu().numpy()

                PSNR_noisy = utils.PSNR(img, img_noisy)
                PSNR_init = utils.PSNR(img, img_init_hat)
                PSNR = utils.PSNR(img, img_hat)

                PSNR_noisy_list.append(PSNR_noisy)
                PSNR_init_list.append(PSNR_init)
                PSNR_list.append(PSNR)

            PSNR_noisy = np.mean(PSNR_noisy_list)
            PSNR_init = np.mean(PSNR_init_list)
            PSNR = np.mean(PSNR_list)

            PSNR_noisy_list_all.append(PSNR_noisy)
            PSNR_init_list_all.append(PSNR_init)
            PSNR_list_all.append(PSNR)

            if test_path == "../data/test_img/":
                print(img_list[ctr], PSNR_noisy, PSNR_init, PSNR)

            ctr += 1

        PSNR_noisy_all = np.mean(PSNR_noisy_list_all)
        PSNR_init_all = np.mean(PSNR_init_list_all)
        PSNR_all = np.mean(PSNR_list_all)

        print(test_path, PSNR_noisy_all, PSNR_init_all, PSNR_all)
