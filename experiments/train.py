"""Poly GCN
   Training Script
"""

import os
import sys
import argparse

from config import Config
from data import building_provider
from torch.utils.data import DataLoader
from models.poly_gcn import MaskRCNN

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir', type=str, help='the dir to load the data')
    parser.add_argument('--log_dir', type=str, help='the dir to save logs and models')
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
    dataset_train = DataProvider(data_dir=data_dir, split='train', mode='train', config=config)
    dataset_val = DataProvider(data_dir=data_dir, split='val', mode='train', config=config)

    train_loader = DataLoader(dataset_train, batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS, shuffle=True,
                              collate_fn=building_provider.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=config.NUM_WORKERS,
                            collate_fn=building_provider.collate_fn)

    return train_loader, val_loader


class BuildingConfig(Config.General_Config):
    # Give the configuration a recognizable name
    NAME = "building"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 80

    PNUM = 16
    CP_NUM = 16

    NUM_WORKERS = 4
    BATCH_SIZE = 1

    # DEBUG = True


class Trainer(object):
    def __init__(self, args, config):
        self.data_dir = args.data_dir
        self.epoch = 0
        self.config = config
        self.train_loader, self.val_loader = get_data_loaders(DataProvider=building_provider.DataProvider,
                                                              data_dir=self.data_dir, config=self.config)
        self.model = MaskRCNN(config=self.config, model_dir=args.log_dir)
        if self.config.GPU_COUNT:
            self.model = self.model.cuda()

    def train_model(self):
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
        self.model.prepare()


        # Training - Stage 1
        print("Training network localization(bounding boxes)")
        self.model.train_model(self.train_loader, self.val_loader,
                               learning_rate=self.config.LEARNING_RATE,
                               epochs=30,
                               layers='boxes')
        print("Training initial points")
        self.model.train_model(self.train_loader, self.val_loader,
                               learning_rate=self.config.LEARNING_RATE/10,
                               epochs=40,
                               layers='boxes')
        # self.model.train_model(self.train_loader, self.val_loader,
        #                        learning_rate=self.config.LEARNING_RATE,
        #                        epochs=3,
        #                        layers='fp')
        print("Training final points")
        # self.model.train_model(self.train_loader, self.val_loader,
        #                   learning_rate=self.config.LEARNING_RATE,
        #                   epochs=50,
        #                   layers='points_only')

        print("training all heads")
        self.model.train_model(self.train_loader, self.val_loader,
                               learning_rate=self.config.LEARNING_RATE/10,
                               epochs=70,
                               layers='points')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        self.model.train_model(self.train_loader, self.val_loader,
                          learning_rate=self.config.LEARNING_RATE/10,
                          epochs=80,
                          layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        self.model.train_model(self.train_loader, self.val_loader,
                          learning_rate=self.config.LEARNING_RATE / 20,
                          epochs=90,
                          layers='all')


if __name__ == '__main__':
    print('==> Parsing Args')
    args = get_args()
    print('==>load configuration')
    configs = BuildingConfig()
    print('Init Trainer')
    trainer = Trainer(args, configs)
    print('==> Start Loop over trainer')
    trainer.train_model()
