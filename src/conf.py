"""
Copyright (c) 2020 Bahareh Tolooshams

config

:author: Bahareh Tolooshams
"""

import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@config_ingredient.config
def cfg():
    hyp = {
        "experiment_name": "simulated_debug",
        "dataset": "simulated",
        "year": "2012",
        "segmentation": True,
        "image_set": "train",
        "network": "DEA1D",
        "data_distribution": "binomial",
        "model_distribution": "binomial",
        "loss_distribution": "gaussian",
        "max_num_groups": 1000,
        "num_groups": 1000,
        "num_trials": 30,
        "num_dict": 1,
        "peak": 2,
        "mu": 0,
        "amp": 16,
        "dictionary_dim": 50,
        "stride": 1,
        "num_conv": 3,
        "train_L": False,
        "L": 10,
        "num_iters": 250,
        "twosided": False,
        "batch_size": 32,
        "num_epochs": 20,
        "epoch_stage": 1,
        "cyclic": False,
        "zero_mean_filters": False,
        "normalize": True,
        "lr": 0.001,
        "amsgrad": False,
        "lr_decay": 0.5,
        "lr_step": 25,
        "noiseSTD": 0.23,
        "info_period": 10,
        "crop_dim": (64, 64),
        "sigma": 0.23,
        "lam": 0.1,
        "nonlin": "ELU",
        "single_bias": True,
        "weight_decay": 0,
        "supervised": True,
        "shuffle": True,
        "denoising": True,
        "logabsDCT": False,
        "init_with_DCT": True,
        "init_with_saved_file": False,
        "test_path": "../data/lena/",
        "device": device,
    }


####################
###### PEAK 4 ######
####################
@config_ingredient.named_config
def poisson_ista15_stride7_peak4_11x11_169_onesided_free_random_elu():
    hyp = {
        "experiment_name": "poisson_ista15_stride7_peak4_11x11_169_onesided_free_random_elu",
        "dataset": "VOC",
        "year": "2012",
        "segmentation": False,
        "image_set": "train",
        "network": "DEA2Dfree",
        "data_distribution": "poisson",
        "model_distribution": "poisson",
        "loss_distribution": "gaussian",
        "dictionary_dim": 11,
        "stride": 7,
        "num_conv": 169,
        "peak": 4,
        "L": None,
        "num_iters": 15,
        "twosided": False,
        "batch_size": 1,
        "num_epochs": 400,
        "zero_mean_filters": False,
        "normalize": False,
        "lr": 1e-3,
        "lr_decay": 0.8,
        "lr_step": 25,
        "cyclic": False,
        "amsgrad": False,
        "info_period": 7000,
        "lam": 0.1,
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


@config_ingredient.named_config
def poisson_ista15_stride7_peak4_11x11_169_onesided_tied_random_elu():
    hyp = {
        "experiment_name": "poisson_ista15_stride7_peak4_11x11_169_onesided_tied_random_elu",
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
        "peak": 4,
        "L": None,
        "num_iters": 15,
        "twosided": False,
        "batch_size": 1,
        "num_epochs": 400,
        "zero_mean_filters": False,
        "normalize": False,
        "lr": 1e-3,
        "lr_decay": 0.8,
        "lr_step": 25,
        "cyclic": False,
        "amsgrad": False,
        "info_period": 7000,
        "lam": 0.1,
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


####################
###### PEAK 2 ######
####################
@config_ingredient.named_config
def poisson_ista15_stride7_peak2_11x11_169_onesided_free_random_elu_smllr():
    hyp = {
        "experiment_name": "poisson_ista15_stride7_peak2_11x11_169_onesided_free_random_elu_smllr",
        "dataset": "VOC",
        "year": "2012",
        "segmentation": False,
        "image_set": "train",
        "network": "DEA2Dfree",
        "data_distribution": "poisson",
        "model_distribution": "poisson",
        "loss_distribution": "gaussian",
        "dictionary_dim": 11,
        "stride": 7,
        "num_conv": 169,
        "peak": 2,
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


@config_ingredient.named_config
def poisson_ista15_stride7_peak2_11x11_169_onesided_tied_random_elu_smllr():
    hyp = {
        "experiment_name": "poisson_ista15_stride7_peak2_11x11_169_onesided_tied_random_elu_smllr",
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
        "peak": 2,
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


####################
###### PEAK 1 ######
####################


@config_ingredient.named_config
def poisson_ista15_stride7_peak1_11x11_169_onesided_free_random_elu_smllr():
    hyp = {
        "experiment_name": "poisson_ista15_stride7_peak1_11x11_169_onesided_free_random_elu_smllr",
        "dataset": "VOC",
        "year": "2012",
        "segmentation": False,
        "image_set": "train",
        "network": "DEA2Dfree",
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


@config_ingredient.named_config
def poisson_ista15_stride7_peak1_11x11_169_onesided_tied_random_elu_smllr():
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
