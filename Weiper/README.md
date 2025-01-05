![alt text](https://github.com/mgranz/weiper_icml/blob/master/WeiPer_v1.png?raw=true)

Repository of the 2024 NeurIPS submission.

This is a working integration into the OpenOOD framework.
For further information on OpenOOD, see: https://github.com/Jingkang50/OpenOOD/tree/main/openood

# Instructions
## Installation 
We also provide an enviroment.yml file for setting up an environment with the necessary dependenciy versions.

Examplary for conda users:
```
conda env create --file environment.yml 
conda activate weiper_env
```

Then, install OpenOOD:
```
cd OpenOOD
pip install .
```
This will install the package `openood` with NAC, and Weiper+KLDiv  as additional postprocessors.

Download all the data and network checkpoints calling the `/scripts/download/download.sh` in the `OpenOOD` directory.

## Test the methods
Open the `Evaluate_WeiPer.ipynb` notebook and follow the instructions.
The hyperparameters can be set in `./OpenOOD/configs/postprocessors/weiper_kldiv.yml`.

Here is the list of hyperparameters that we found:
| Hyperparameter | CIFAR10 (ResNet18) | CIFAR100 (ResNet18) | ImageNet-1K (ResNet50) | ImageNet-1K (ViT-B/16) |
|----------------|---------------------|---------------------|------------------------|-------------------------|
| `n_repeats`             | 100                 | 100                 | 100                    | 100                     |
| `perturbation_distance`             | 1.8                 | 2.4                 | 2.4                    | 2.0                     |
| `n_bins`         | 100                 | 100                 | 100                    | 80                      |
| `lambda_1`            | 2.5                 | 0.1                 | 2.5                    | 2.5                     |
| `lambda_2`            | 0.1                 | 1                   | 0.25                   | 0.1                     |
| `smoothing`            | 4                   | 4                   | 40                     | 40                      |
| `smoothing_perturbed`            | 15                  | 40                  | 15                     | 15                      |

## Quick Hyperparameter Search
We provide a quick way of searching for hyperparameters by presampling the latents for each model. We recommend to not use the build-in hyperparameter search of OpenOOD as our method has more parameters than others and thus will take substantially longer to execute.

Follow these steps to reproduce the result:

1. Store the penultimate features for each model:
`python3 store_latents.py --datasets 'cifar' --device 0`,
for CIFAR10 and CIFAR100 and `--datasets 'all'` for all datasets.

2. Search for hyperparameters:
`python3 hyperparameter_search_on_latents.py --dataset 'cifar10' --device 0`,
for a search on CIFAR10. Set `--dataset 'imagenet' --resnet False` for searching for hyperparameters on ImageNet using ViT-B/16.


## How to find the code for the methods
The code is located in the postprocessor directory of OpenOOD:


`./OpenOOD/openood/postprocessors/weiper_kldiv/`

`./OpenOOD/openood/postprocessors/weiper_kldiv_postprocessor.py/`

# Density Evolution 

<img src="https://github.com/mgranz/weiper_icml/blob/master/Weiper/ResNet18_penultimate_layer_resize.gif" width="340" height="340" />
Evolution of the densities of specific penultimate output dimensions over the training period. Blue is the training data, red the OOD data and purple/turquois is the test data.

# Citation

If you use WeiPer in your research, please cite our paper:
```
@article{granz2024weiper,
  title={WeiPer: OOD Detection using Weight Perturbations of Class Projections},
  author={Granz, Maximilian and Heurich, Manuel and Landgraf, Tim},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
