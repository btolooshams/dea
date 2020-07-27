"""
Copyright (c) 2020 Bahareh Tolooshams

train

:author: Bahareh Tolooshams
"""


import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sparselandtools.dictionaries import DCTDictionary
import os
from tqdm import tqdm
from datetime import datetime
from sacred import Experiment
from torchsummary import summary


import sys

sys.path.append("src/")

import model, generator, trainer, utils

from conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")


ex = Experiment("train", ingredients=[config_ingredient])


@ex.automain
def run(cfg):

    hyp = cfg["hyp"]

    print(hyp)

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    os.makedirs(PATH)

    filename = os.path.join(PATH, "hyp.pickle")
    with open(filename, "wb") as file:
        pickle.dump(hyp, file)

    print("load data.")
    if hyp["dataset"] == "image":
        train_loader = generator.get_path_loader(
            hyp["batch_size"], hyp["test_path"], shuffle=False
        )
        test_loader = train_loader

    elif hyp["dataset"] == "VOC":
        train_loader, _ = generator.get_VOC_loaders(
            hyp["batch_size"],
            crop_dim=hyp["crop_dim"],
            shuffle=hyp["shuffle"],
            image_set=hyp["image_set"],
            segmentation=hyp["segmentation"],
            year=hyp["year"],
        )
        test_loader = generator.get_path_loader(1, hyp["test_path"], shuffle=False)

    if hyp["init_with_DCT"]:
        dct_dictionary = DCTDictionary(
            hyp["dictionary_dim"], np.int(np.sqrt(hyp["num_conv"]))
        )
        H_init = dct_dictionary.matrix.reshape(
            hyp["dictionary_dim"], hyp["dictionary_dim"], hyp["num_conv"]
        ).T

        if hyp["logabsDCT"]:
            H_init = np.log(np.abs(H_init) + 1e-6)

        H_init = np.expand_dims(H_init, axis=1)
        H_init = torch.from_numpy(H_init).float().to(hyp["device"])
    else:
        H_init = None

    print("create model.")
    if hyp["network"] == "DEA1D":
        net = model.DEA1D(hyp, H_init)
    elif hyp["network"] == "DEA2Dfree":
        net = model.DEA2Dfree(hyp, H_init)
    elif hyp["network"] == "DEA2Dtied":
        net = model.DEA2Dtied(hyp, H_init)
    else:
        print("model does not exist!")

    torch.save(net, os.path.join(PATH, "model_init.pt"))

    if hyp["network"] == "DEA1D":
        criterion = utils.DEALoss1D(hyp["model_distribution"], hyp["loss_distribution"])
    else:
        criterion = utils.DEALoss2D(hyp["model_distribution"], hyp["loss_distribution"])

    if hyp["init_with_DCT"]:
        net.normalize()

    optimizer = optim.Adam(
        net.parameters(), lr=hyp["lr"], eps=1e-3, weight_decay=hyp["weight_decay"]
    )
    if hyp["cyclic"]:
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=hyp["base_lr"],
            max_lr=hyp["max_lr"],
            step_size_up=hyp["step_size"],
            cycle_momentum=False,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
        )

    print("train auto-encoder.")
    net = trainer.train_ae(
        net,
        train_loader,
        hyp,
        criterion,
        optimizer,
        scheduler,
        PATH,
        test_loader,
        0,
        hyp["num_epochs"],
        H_init,
    )
