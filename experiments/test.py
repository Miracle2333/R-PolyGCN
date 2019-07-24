"""Poly GCN
   Training Script
"""

import os
import time
import sys
import argparse
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import re
import logging
import cv2
import csv
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

import zipfile
import urllib.request
import shutil

from config import Config
from data import building_provider
from torch.utils.data import DataLoader
from models.poly_gcn import MaskRCNN
from models import visualize


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir', type=str, help='the dir to load the data')
    parser.add_argument('--log_dir', type=str, help='the dir to save logs and models')
    parser.add_argument('--output_dir', type=str, help='the output path of the testing results')
    parser.add_argument('--model', type=str, help='the checkpoint file to resume from')
    args = parser.parse_args()

    return args


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


def get_data_loaders(DataProvider, data_dir, config):
    print('Loading Data')
    dataset_tets = DataProvider(data_dir=data_dir, split='test', mode='test', config=config)

    test_loader = DataLoader(dataset_tets, batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS,
                              collate_fn=building_provider.collate_fn)

    return test_loader


class BuildingConfig(Config.General_Config):
    # Give the configuration a recognizable name
    NAME = "building"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes

    # steps per epoch
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50




class Tester(object):
    def __init__(self, args, config):
        self.data_dir = args.data_dir
        self.global_step = 0
        self.epoch = 0
        self.config = config
        self.test_loader = get_data_loaders(DataProvider=building_provider.DataProvider,
                                                              data_dir=self.data_dir, config=self.config)
        self.model = MaskRCNN(config=self.config, model_dir=args.log_dir)

        if config.GPU_COUNT:
            self.model = self.model.cuda()

        self.output_path = args.output_dir

    def load_model(self):
        # Select weights file to load
        if args.model:
            if args.model.lower() == "coco":
                COCO_MODEL_PATH = os.path.join(os.getcwd(), "mask_rcnn_coco.pth")
                model_path = COCO_MODEL_PATH
            elif args.model.lower() == "last":
                # Find last trained weights
                model_path = self.model.find_last()[1]
            elif args.model.lower() == "imagenet":
                # Start from ImageNet trained weights
                model_path = self.config.IMAGENET_MODEL_PATH
            else:
                model_path = args.model
        else:
            model_path = ""

            # Load weights
        print("Loading weights ", model_path)
        self.model.load_weights(model_path)

    def test_model(self):
        results = self.model.detect(self.test_loader)
        output_path = self.output_path
        for i in range(0, len(results)):
            r = results[i]
            image = r['image']
            id = r['image_id']
            boxes = r['rois']
            polys = r['polys']
            scores = r['scores']
            class_ids = r['class_ids']
            visualize.display_detection(image, boxes, polys, class_ids, "building", id,
                                        output_path, scores)


if __name__ == '__main__':
    print('==> Parsing Args')
    args = get_args()
    print('==>load configuration')
    configs = BuildingConfig()
    print('Init Trainer')
    tester = Tester(args, configs)
    print('==> Start Loop over trainer')
    tester.load_model()
    tester.test_model()
