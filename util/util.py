# -----------------------------------------------------------------------------
# Functions for general utility
# -----------------------------------------------------------------------------
import os
import random
import logging
import shutil

import numpy as np
import torch

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def seed_everything(seed):
    '''
    Set manual seed to maintain reproducibility
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    torch.set_flush_denormal(True) # cek dlu
    info(f"Set seed at {seed}")


def check_model_dir(save_path, config_path, train):
    '''
    Create experiment directories if not exists
    '''
    model_dir = save_path
    pth_dir = os.path.join(model_dir, "model") # Save training model of AE as .ckpt file
    gmm_dir = os.path.join(model_dir, "gmm") # Save gmm .npy file
    pred_labels_dir = os.path.join(model_dir, "pred_labels")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)
    if not os.path.exists(gmm_dir):
        os.makedirs(gmm_dir)
    if not os.path.exists(pred_labels_dir):
        os.makedirs(pred_labels_dir)

    if train:
        shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))


def info(text):
    '''
    Output text info in terminal console
    '''
    print("-"*80)
    logging.info(f" {text}")
    print("-"*80)