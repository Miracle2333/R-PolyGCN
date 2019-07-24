"""Mask RCNN"""

import torch.nn as nn
import numpy as np
from ..backbones import resnet, resnetxt
from ..neck import fpn
from ..head import rpn
from ..head import mask
from ..head import box_head
from ..roi_extractor import get_roi
from ..hat import GCN


class MaskRCNN(nn.Module):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 pretrained=None):
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        # self.set_log_dir()
        self.build(config=config)
        # self.initialize_weights()
        # self.loss_history = []
        # self.val_loss_history = []
        # self.pool_size = config.POOL_SIZE

    def build(self, config):
        """Design the Detector Architecture
        """
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        if config.BACKBONE == 'resnet':
            self.backbone = resnet.ResNet(depth=50)
        elif config.BACKBONE == 'resnetxt':
            self.backbone = resnetxt.ResNeXt(depth=50)

        self.neck = fpn()



