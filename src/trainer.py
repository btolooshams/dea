"""
Copyright (c) 2020 Bahareh Tolooshams

data generator

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

import utils


def train_ae(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    PATH="",
    test_loader=None,
    epoch_start=0,
    epoch_end=1,
    H_true=None,
    org_img=None,
):

    info_period = hyp["info_period"]
    device = hyp["device"]
    zero_mean_filters = hyp["zero_mean_filters"]
    normalize = hyp["normalize"]
    network = hyp["network"]
    mu = hyp["mu"]
    supervised = hyp["supervised"]
    Jg = hyp["num_trials"]
    data_distribution = hyp["data_distribution"]
    model_distribution = hyp["model_distribution"]
    peak = hyp["peak"]
    train_L = hyp["train_L"]

    if data_distribution == "gaussian":
        noiseSTD = hyp["noiseSTD"]

    if normalize:
        net.normalize()

    if hyp["denoising"]:
        if test_loader is not None:
            with torch.no_grad():
                psnr = []
                t = 0
                for idx_test, (img_test, _) in enumerate(test_loader):
                    t += 1

                    psnr_i = 0
                    N = 20
                    for _ in range(N):

                        if data_distribution == "gaussian":
                            img_test_noisy = (
                                img_test + noiseSTD / 255 * torch.randn(img_test.shape)
                            ).to(device)
                        elif data_distribution == "binomial":
                            img_test = img_test.to(device)
                            sampler = torch.distributions.bernoulli.Bernoulli(
                                probs=img_test
                            )
                            img_test_noisy = sampler.sample()
                            for j in range(Jg - 1):
                                img_test_noisy += sampler.sample()
                            img_test_noisy /= Jg
                        elif data_distribution == "poisson":
                            if img_test[img_test > 0] is not None:
                                img_test[img_test == 0] = torch.min(
                                    img_test[img_test > 0]
                                )
                            img_test = img_test.to(device)
                            Q = torch.max(img_test) / peak
                            rate = img_test / Q
                            if torch.isnan(torch.min(rate)):
                                continue
                            sampler = torch.distributions.poisson.Poisson(rate)

                            if model_distribution == "poisson":
                                img_test_noisy = sampler.sample()
                            else:
                                img_test_noisy = sampler.sample() * Q

                        Hx_hat, _, _ = net(img_test_noisy, mu)

                        if model_distribution == "gaussian":
                            img_test_hat = Hx_hat + mu
                        elif model_distribution == "binomial":
                            img_test_hat = torch.nn.Sigmoid()(Hx_hat + mu)
                        elif model_distribution == "poisson":
                            img_test_hat = Q * torch.exp(Hx_hat + mu)

                        psnr_i += utils.PSNR(
                            img_test[0, 0, :, :].detach().cpu().numpy(),
                            img_test_hat[0, 0, :, :].detach().cpu().numpy(),
                        )

                    psnr.append(psnr_i / N)

                    if model_distribution == "poisson":
                        noisy_psnr = utils.PSNR(
                            img_test[0, 0, :, :].detach().cpu().numpy(),
                            (Q * img_test_noisy[0, 0, :, :]).detach().cpu().numpy(),
                        )
                    else:
                        noisy_psnr = utils.PSNR(
                            img_test[0, 0, :, :].detach().cpu().numpy(),
                            img_test_noisy[0, 0, :, :].detach().cpu().numpy(),
                        )

                np.save(os.path.join(PATH, "psnr_init.npy"), np.array(psnr))
                print(
                    "PSNR: {}, {}".format(
                        np.round(np.array(noisy_psnr), decimals=4),
                        np.round(np.array(psnr), decimals=4),
                    )
                )

    nan_ctr = 0

    for epoch in tqdm(range(epoch_start, epoch_end)):
        scheduler.step()
        loss_all = 0
        for idx, (img, _) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()

            if data_distribution == "gaussian":
                img_noisy = (img + noiseSTD / 255 * torch.randn(img.shape)).to(device)
                img = img.to(device)
            elif data_distribution == "binomial":
                img = img.to(device)
                sampler = torch.distributions.bernoulli.Bernoulli(probs=img)
                img_noisy = sampler.sample()
                for j in range(Jg - 1):
                    img_noisy += sampler.sample()
                img_noisy /= Jg
            elif data_distribution == "poisson":
                if torch.sum(img) != 0:
                    img[img == 0] = torch.min(img[img > 0])
                else:
                    continue
                img = img.to(device)
                Q = torch.max(img) / peak
                rate = img / Q
                if torch.isnan(torch.min(rate)):
                    continue
                sampler = torch.distributions.poisson.Poisson(rate)
                if model_distribution == "poisson":
                    img_noisy = sampler.sample()
                else:
                    img_noisy = sampler.sample() * Q

            if torch.isnan(torch.sum(img_noisy)):
                print("img_noisy got NaN!")
                continue

            Hx, x_hat, _ = net(img_noisy, mu)

            if torch.isnan(torch.sum(Hx)):
                print("Hx got NaN!")

                nan_ctr += 1

                if nan_ctr > 20:
                    break
                else:
                    continue

            if supervised:
                if model_distribution == "poisson":
                    loss = criterion(img, Hx, Q)
                else:
                    loss = criterion(img, Hx)
            else:
                loss = criterion(img_noisy, Hx)

            if loss > 10:
                f.write("skip. loss is large! \r\n")
                print("skip. loss is large!")

                continue

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if zero_mean_filters:
                net.zero_mean()
            if normalize:
                net.normalize()

            if idx % info_period == 0:

                if H_true is not None:
                    err_H = utils.err2d_H(H_true, net.get_param("H"))
                    print(
                        "loss:{:.4f}, err_H:{:4f}\n".format(loss.item(), np.mean(err_H))
                    )
                else:
                    print("loss:{:.4f}\n".format(loss.item()))

            torch.cuda.empty_cache()

        # ===================log========================

        if hyp["denoising"]:
            if test_loader is not None:
                with torch.no_grad():
                    psnr = []
                    t = 0
                    for idx_test, (img_test, _) in enumerate(test_loader):
                        t += 1

                        psnr_i = 0
                        N = 20
                        for _ in range(N):
                            if data_distribution == "gaussian":
                                img_test_noisy = (
                                    img_test
                                    + noiseSTD / 255 * torch.randn(img_test.shape)
                                ).to(device)
                            elif data_distribution == "binomial":
                                img_test = img_test.to(device)
                                sampler = torch.distributions.bernoulli.Bernoulli(
                                    probs=img_test
                                )
                                img_test_noisy = sampler.sample()
                                for j in range(Jg - 1):
                                    img_test_noisy += sampler.sample()
                                img_test_noisy /= Jg
                            elif data_distribution == "poisson":
                                if torch.sum(img_test) != 0:
                                    img_test[img_test == 0] = torch.min(
                                        img_test[img_test > 0]
                                    )
                                img_test = img_test.to(device)
                                Q = torch.max(img_test) / peak
                                rate = img_test / Q
                                if torch.isnan(torch.min(rate)):
                                    continue
                                sampler = torch.distributions.poisson.Poisson(rate)
                                if model_distribution == "poisson":
                                    img_test_noisy = sampler.sample()
                                else:
                                    img_test_noisy = sampler.sample() * Q

                            Hx_hat, _, _ = net(img_test_noisy, mu)

                            if model_distribution == "gaussian":
                                img_test_hat = Hx_hat + mu
                            elif model_distribution == "binomial":
                                img_test_hat = torch.nn.Sigmoid()(Hx_hat + mu)
                            elif model_distribution == "poisson":
                                img_test_hat = Q * torch.exp(Hx_hat + mu)

                            psnr_i += utils.PSNR(
                                img_test[0, 0, :, :].detach().cpu().numpy(),
                                img_test_hat[0, 0, :, :].detach().cpu().numpy(),
                            )

                        psnr.append(psnr_i / N)

                    np.save(
                        os.path.join(PATH, "psnr_epoch{}.npy".format(epoch)),
                        np.array(psnr),
                    )
                    print("PSNR: {}".format(np.round(np.array(psnr), decimals=4)))

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))
        torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

        print(
            "epoch [{}/{}], loss:{:.4f} ".format(
                epoch + 1, hyp["num_epochs"], loss.item()
            )
        )

        if torch.isnan(torch.min(net.get_param("H"))):
            print("network got NaN!")
            break

    return net


def train_ae_simulated(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    PATH="",
    test_loader=None,
    epoch_start=0,
    epoch_end=1,
    H_true=None,
):
    info_period = hyp["info_period"]
    noiseSTD = hyp["noiseSTD"]
    device = hyp["device"]
    zero_mean_filters = hyp["zero_mean_filters"]
    normalize = hyp["normalize"]

    if normalize:
        net.normalize()

    for epoch in tqdm(range(epoch_start, epoch_end)):
        scheduler.step()
        loss_all = 0
        for idx, (y, _, mu, _) in tqdm(enumerate(data_loader)):

            y = y.to(device)
            mu = mu.to(device)

            # ===================forward=====================
            Hx, _, _ = net(y, mu)

            loss = criterion(y, Hx)

            loss_all += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if zero_mean_filters:
                net.zero_mean()
            if normalize:
                net.normalize()

            if idx % info_period == 0:
                print("loss:{:.4f}\n".format(loss.item()))

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))
        torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

        print(
            "epoch [{}/{}], loss:{:.4f}".format(
                epoch + 1, hyp["num_epochs"], loss.item()
            )
        )

    return net
