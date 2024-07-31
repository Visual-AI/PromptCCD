# Data & Setup

This README provides guidelines on how to prepare and structure the datasets. In our paper, we provides 7 CCD benchmarks datasets, i.e., CIFAR100, ImageNet-100, TinyImageNet, and Caltech-101 for generic datasets and Aircraft, Stanford Cars, and CUB for fine-grained datasets.

## Dataset

#### CIFAR100
1. install **cifar2png** package using pip. source: [link](https://github.com/knjcode/cifar2png).
    ```shell
    $ pip install cifar2png
    ```
2. Specifically, you can use below code to download the dataset:
    ```shell
    $ cifar2png cifar100 data/cifar-100-images --name-with-batch-index
    ```
3. The structure should be:
    ```
    cifar-100-images/
    ├── test/
    └── train/
    ```

#### ImageNet-100
1. Download the ImageNet dataset from [ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/index.php).
2. The structure should be:
    ```
    ILSVRC12/
    ├── train/
    └── val/
    ```

#### TinyImageNet
1. Download the TinyImageNet dataset from this [link](http://cs231n.stanford.edu/tiny-imagenet-200.zip).
2. Unzip the file and the structure should be:
    ```
    tiny-imagenet-200/
    ├── test/
    ├── train/
    ├── val/
    ├── wnids.txt
    └── words.txt
    ```

#### Caltech-101
1. Download the Caltech-101 dataset from this [link](https://www.kaggle.com/datasets/imbikramsaha/caltech-101).
2. Unzip the file and the structure should be:
    ```
    caltech-101/
    ├── accordion/
    ├── airplanes/
    ├── ...
    ├── wrench/
    └── yin_yang/
    ```

#### Fine-grained datasets, *e.g.*, Aircraft, Stanford Cars, and CUB
Please follow the instruction [here](https://github.com/sgvaze/SSB/blob/main/DATA.md) to download these datasets.

## Setup

Create `data` directory and attached symbolic links between the dataset paths and `data` dir. This can be done by:

*See `util/data_util.py` for more details / if you want to change the dataset path(s).

```shell
# create data directory
$ mkdir data

# attach symbolic links for each dataset path:
# CIFAR100
$ ln -s /dataset/path/cifar-100-images /repository/path/promptccd/data/cifar-100-images

# ImageNet-100
$ ln -s /dataset/path/ILSVRC12 /repository/path/promptccd/data/imagenet

# TinyImageNet
$ ln -s /dataset/path/tiny-imagenet-200 /repository/path/promptccd/data/tiny-imagenet-200

# Caltech-101
$ ln -s /dataset/path/caltech-101 /repository/path/promptccd/data/caltech-101

# Aircraft
$ ln -s /dataset/path/aircraft/fgcv-aircraft-2013b /repository/path/promptccd/data/fgcv-aircraft-2013b

# Stanford Cars
$ ln -s /dataset/path/stanford_car /repository/path/promptccd/data/stanford_car

# CUB
$ ln -s /dataset/path/CUB/CUB_200_2011 /repository/path/promptccd/data/CUB/CUB_200_2011
```




## Citations
The citations for the original datasets (*following the order above*) are:

```
%---------------------------------------------------------
% CIFAR100
%---------------------------------------------------------
@article{krizhevsky2009learning,
    title     = {Learning multiple layers of features from tiny images},
    author    = {Krizhevsky, A. and Hinton, G.},
    journal   = {Master's thesis, Department of Computer Science, University of Toronto},
    year      = {2009},
}
```

```
%---------------------------------------------------------
% ImageNet-100
%---------------------------------------------------------
@article{russakovsky2015imagenet,
    title     = {Imagenet large scale visual recognition challenge},
    author    = {Russakovsky, Olga and Deng, Jia and Su, Hao and Krause, Jonathan and Satheesh, Sanjeev and Ma, Sean and Huang, Zhiheng and Karpathy, Andrej and Khosla, Aditya and Bernstein, Michael and others},
    journal   = {International Journal of Computer Vision (IJCV)},
    year      = {2015},
}
```

```
%---------------------------------------------------------
% TinyImageNet
%---------------------------------------------------------
@article{le2015tiny,
    title     = {Tiny imagenet visual recognition challenge},
    author    = {Le, Ya and Yang, Xuan},
    journal   = {CS 231N},
    year      = {2015},
}
```

```
%---------------------------------------------------------
% Caltech-101
%---------------------------------------------------------
@article{fei2006one,
    title     = {One-shot learning of object categories},
    author    = {Fei-Fei, Li and Fergus, Robert and Perona, Pietro},
    journal   = {Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    year      = {2006},
    publisher = {IEEE},
}
```

```
%---------------------------------------------------------
% Aircraft
%---------------------------------------------------------
@article{maji13fine-grained,
    title    = {Fine-Grained Visual Classification of Aircraft},
    author   = {S. Maji and J. Kannala and E. Rahtu
                        and M. Blaschko and A. Vedaldi},
    journal  = {arXiv preprint arXiv:1306.5151},
    year     = {2013},
}
```

```
%---------------------------------------------------------
% Stanford Cars
%---------------------------------------------------------
@inproceedings{krause20133d,
    title     ={3d object representations for fine-grained categorization},
    author    ={Krause, Jonathan and Stark, Michael and Deng, Jia and Fei-Fei, Li},
    booktitle ={4th International IEEE Workshop on  3D Representation and Recognition (3dRR-13)},
    year      ={2013},
}
```

```
%---------------------------------------------------------
% CUB
%---------------------------------------------------------
@techreport{WahCUB_200_2011,
    Title       = {{The Caltech-UCSD Birds-200-2011 Dataset}},
    Author      = {Catherine Wah and Steve Branson and Peter Welinder and Pietro Perona and Serge Belongie},
    Year        = {2011},
    Institution = {California Institute of Technology},
}
```

