#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_cuda_optimization(benchmark, deterministic):
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

def set_image_reference():
    # Disable PIL limitation
    Image.MAX_IMAGE_PIXELS = None
    # cv2.resize tries to multithread and somewhere something goes into a deadlock
    cv2.setNumThreads(0)

def setup(benchmark, deterministic, seed=None):
    set_image_reference()
    set_cuda_optimization(benchmark, deterministic)
    set_random_seed(seed)