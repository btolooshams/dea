[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# DEA (deep exponential auto-encoder)

### Paper

Bahareh Tolooshams*, Andrew H. Song*, Simona Temereanca, and Demba Ba. [Convolutional dictionary learning based auto-encoders for natural exponential-family distributions](https://proceedings.icml.cc/static/paper_files/icml/2020/5733-Paper.pdf)

##
### Trained Models

Trained models are stored in `results/trained_models`.

##
### PATH

For any scripts to run, make sure you are in `src` directory.

##
### Configuration


Create a configuration function in `conf.py` containing a dictionary of hyperparameters for your experiment.

```
@config_ingredient.named_config
def exp1():
hyp = {
    "experiment_name": "poisson_peak4",
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
```

##
### Training

`python train.py with cfg.exp1`

##
### Results

When training is done, the results are saved in `results/{experiment_name}/{random_date}`.

`random_date` is a datetime string generated at the begining of the training.

##
### Prediction

Run `predict.py`. Make sure to specify the parameters from line 42 - 89.
