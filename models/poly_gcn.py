"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import datetime
import math
import os
import random
import re


import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import img_as_ubyte
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision import transforms

from utils import Utils as utils
from models.basic import ResNet, FPN, RPN, Classifier, Mask, Edge_Annotation, detection_layer_gcn, refine_detections_gcn
from models.basic import proposal_layer, detection_target_layer, detection_layer, pyramid_roi_align, interpolated_sum
from models.first_annotation import FirstAnnotation
from models.GNN.GCN import GCN
from models import loss as LossFunction
from models import visualize
from orig_models.ActiveSpline import ActiveSplineTorch
from orig_models.ActiveSpline import ActiveBoundaryTorch

# import visualize
from core.nms.nms_wrapper import nms
from core.roialign.roi_align.crop_and_resize import CropAndResizeFunction
from skimage.segmentation import mark_boundaries
import skimage.draw
import skimage.io

import matplotlib.pyplot as plt


############################################################
#  Logging Utility Functions
############################################################

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


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    # Print New Line on Complete
    if iteration == total:
        print()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.pool_size = config.POOL_SIZE
        self.coarse_to_fine_steps = 3  #for GCN steps
        self.n_adj = n_adj = 4
        self.cnn_feature_grids = cnn_feature_grids = [112, 56, 28, 28]
        self.psp_feature = [self.cnn_feature_grids[-1]]
        self.grid_size = 28
        self.state_dim = state_dim = 128
        self.final_dim = 64*4
        self.cp_num = config.CP_NUM
        self.p_num = config.PNUM
        self.spline = ActiveSplineTorch(self.cp_num, self.p_num, device=device,
                                        alpha=0.5)
        self.boundary = ActiveBoundaryTorch(self.cp_num, self.p_num, device=device,
                                        alpha=0.5)

        self.build(config=self.config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []
        self.global_step = 0
        self.print_freq = 40
        self.model_check_path = None

        # self.image_shape = config.IMAGE_SHAPE

    def prepare(self):
        self.set_log_dir(model_path=self.model_check_path)
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'logs', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.log_dir, 'logs', 'train_val'))

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()
        self.image_shape = config.IMAGE_SHAPE

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # self.skipfeatures = SkipFeatures(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1)

        # Generate Anchors
        self.anchors = torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                config.BACKBONE_SHAPES,
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float()
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # Initial Mask
        # self.coarse_mask = CoarseMask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # Initial Vertex and Edges
        self.first_annotation = FirstAnnotation(28, 256, 2)
        # #Edge annotation
        self.edge_annotation = Edge_Annotation(256, self.grid_size)

        # The number of GCN needed
        if self.coarse_to_fine_steps > 0:
            for step in range(self.coarse_to_fine_steps):
                if step == 0:
                    self.gnn = nn.ModuleList(
                        [GCN(state_dim=self.state_dim, feature_dim=self.final_dim + 2).to(device)])
                else:
                    self.gnn.append(GCN(state_dim=self.state_dim, feature_dim=self.final_dim + 2).to(device))
        else:

            self.gnn = GCN(state_dim=self.state_dim, feature_dim=self.final_dim + 2)

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #         # m.weight.data.normal_(0.0, 0.00002)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0.0, 0.01)
        #         nn.init.constant_(m.bias, 0)


    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False
            else:
                a = 1

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        # self.model_dir = dir_name
        self.model_check_path = checkpoint

        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):

            if (filepath != self.config.IMAGENET_MODEL_PATH) & (filepath != self.config.COCO_MODEL_PATH):
                self.load_state_dict(torch.load(filepath)['state_dict'], strict=False)
                save_state = torch.load(filepath)
                self.global_step = save_state['global_step']
                self.epoch = save_state['epoch']
                self.optimizer.load_state_dict(save_state['optimizer'])
                self.lr_decay.load_state_dict(save_state['lr_decay'])
            else:
                state_dict = torch.load(filepath)
                self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def save_checkpoint(self, epoch, save_name):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        # save_name = os.path.join(self.log_dir, 'epoch%d_step%d.pth' \
        #                          % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def predict(self, input, mode):
        """Major pipelines of the network for training and inference
        """
        molded_images = input[0]
        image_metas = input[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]
        # gcn_feature_maps = skip

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                  proposal_count=proposal_count,
                                  nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                  anchors=self.anchors,
                                  config=self.config)

        if mode == 'inference':

            fwd_polys = input[2]
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = torch.from_numpy(np.array([h, w, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :4] / scale

            # Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            # Create masks for detections
            mrcnn_mask_feature = self.mask(mrcnn_feature_maps, detection_boxes)

            # First INITIAL vertex prediction
            # edge_logits, vertex_logits, logprob, _ = self.first_annotation.forward(mrcnn_mask_feature)
            #
            # edge_logits = edge_logits.view(
            #     (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1)
            # vertex_logits = vertex_logits.view(
            #     (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1)
            #
            # feature_with_edges = torch.cat([mrcnn_mask_feature, edge_logits, vertex_logits], 1)
            #
            # conv_layers = [self.edge_annotation.edge_annotation_cnn(feature_with_edges)]

            mrcnn_mask_feature = mrcnn_mask_feature.permute(0, 2, 3, 1).view(-1, self.grid_size ** 2, 256)
            conv_layers = [mrcnn_mask_feature]

            # edge_logits = (edge_logits.squeeze(1)).view(-1, self.grid_size * self.grid_size)
            # vertex_logits = (vertex_logits.squeeze(1)).view(-1, self.grid_size * self.grid_size)

            # Step by step GCN
            pred_polys = []
            N = list(mrcnn_mask_feature.size())
            init_polys = np.zeros((N[0], self.config.CP_NUM, 2), dtype=np.float32)
            temp_poly = fwd_polys.copy()
            for i in range(0, N[0]):
                init_polys[i, :, :] = temp_poly[0, :, :]
            init_polys = torch.from_numpy(init_polys)

            for i in range(self.coarse_to_fine_steps):
                if i == 0:
                    component = utils.prepare_gcn_component(init_polys.numpy(),
                                                            self.psp_feature,
                                                            init_polys.size()[1],
                                                            n_adj=self.n_adj)
                    init_polys = init_polys.cuda()
                    adjacent = component['adj_matrix'].cuda()
                    init_poly_idx = component['feature_indexs'].cuda()
                    cnn_feature = []
                    try:
                        cnn_feature = self.edge_annotation.sampling(init_poly_idx, conv_layers)
                    except RuntimeError:
                        print(conv_layers[0].shape, mrcnn_mask_feature[0].shape)
                    input_feature = torch.cat((cnn_feature, init_polys), 2)

                else:
                    init_polys = gcn_pred_poly
                    cnn_feature = interpolated_sum(conv_layers, init_polys, self.psp_feature)
                    input_feature = torch.cat((cnn_feature, init_polys), 2)

                gcn_pred = self.gnn[i].forward(input_feature, adjacent)
                gcn_pred_poly = init_polys.to(device) + gcn_pred

                pred_polys.append(gcn_pred_poly)
                adjacent = adjacent
            pred_polys = pred_polys[-1]

            # Add back batch dimension
            detections = detections.unsqueeze(0)
            # mrcnn_mask = mrcnn_mask.unsqueeze(0)

            return [detections, pred_polys]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            vertex_masks = input[5]
            edge_masks = input[6]
            gt_polys = input[7]
            fwd_polys = input[8]


            # Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = torch.from_numpy(np.array([h, w, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale
            # gt_sboxes = gt_sboxes/scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            gt_sboxes = 0
            rois, target_class_ids, target_deltas, target_polys, target_mask, gt_roi_masks, vertex_target_masks, edge_target_masks = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_sboxes,  gt_masks, vertex_masks, edge_masks, gt_polys, self.config)

            # vertex_target_masks = vertex_target_masks.data.veiw(-1, self.grid_size*self.grid_size)
            # edge_target_masks = edge_target_masks.data.veiw(-1, self.grid_size*self.grid_size)

            if not rois.size()[0]:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                edge_logits = Variable(torch.FloatTensor())
                vertex_logits = Variable(torch.FloatTensor())
                feature_with_edges = Variable(torch.FloatTensor())
                pred_polys = Variable(torch.FloatTensor())
                adjacent = Variable(torch.FloatTensor())
                target_polys = Variable(torch.FloatTensor())
                rois = Variable(torch.FloatTensor())
                scale = Variable(torch.FloatTensor())
                detection_boxes = Variable(torch.FloatTensor())
                gt_roi_masks = Variable(torch.FloatTensor())


                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    edge_logits = edge_logits.cuda()
                    vertex_logits = vertex_logits.cuda()
                    feature_with_edges = feature_with_edges.cuda()
                    pred_polys = pred_polys.cuda()
                    adjacent = adjacent.cuda()
                    target_polys = target_polys.cuda()
                    rois = rois.cuda()
                    scale = scale.cuda()
                    detection_boxes = detection_boxes.cuda()
                    gt_roi_masks = gt_roi_masks.cuda()

            else:
                # Network Heads
                #extract roi_aligned features
                # rois_class_feature = pyramid_roi_align([rois] + mrcnn_feature_maps, self.pool_size, self.image_shape)
                #

                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                # Create masks for detections
                # mrcnn_mask = self.mask(mrcnn_feature_maps, rois)
                _, mrcnn_mask = self.mask(mrcnn_feature_maps, rois)
                # mrcnn_mask = mrcnn_mask_feature


                # Detections
                # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
                detections = detection_layer_gcn(self.config, rois, mrcnn_class, mrcnn_bbox, target_class_ids, image_metas)

                # Convert boxes to normalized coordinates
                # TODO: let DetectionLayer return normalized coordinates to avoid
                #       unnecessary conversions
                h, w = self.config.IMAGE_SHAPE[:2]
                scale = torch.from_numpy(np.array([h, w, h, w])).float()
                if self.config.GPU_COUNT:
                    scale = scale.cuda()
                detection_boxes = detections[:, :4] / scale

                # Add back batch dimension
                detection_boxes = detection_boxes.unsqueeze(0)

                rois_feature = pyramid_roi_align([detection_boxes] + mrcnn_feature_maps, 28, self.image_shape)

                # First INITIAL vertex prediction
                edge_logits, vertex_logits, logprob, _ = self.first_annotation.forward(rois_feature)

                edge_logits = edge_logits.view(
                    (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1)
                vertex_logits = vertex_logits.view(
                    (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1)

                feature_with_edges = torch.cat([rois_feature, edge_logits, vertex_logits], 1)

                conv_layers = [self.edge_annotation.edge_annotation_cnn(feature_with_edges)]
                #
                edge_logits = (edge_logits.squeeze(1)).view(-1, self.grid_size*self.grid_size)
                vertex_logits = (vertex_logits.squeeze(1)).view(-1, self.grid_size*self.grid_size)
                # mrcnn_mask_feature = mrcnn_mask_feature.permute(0, 2, 3, 1).view(-1, self.grid_size ** 2, 256)
                #
                # conv_layers = [mrcnn_mask_feature]

                # Step by step GCN
                pred_polys = []
                N = list(rois_feature.size())
                init_polys = np.zeros((N[0], self.cp_num, 2), dtype=np.float32)
                temp_poly = fwd_polys.numpy().copy()
                for i in range(0, N[0]):
                    init_polys[i, :, :] = temp_poly[0, 0, :, :]
                init_polys = torch.from_numpy(init_polys)

                for i in range(self.coarse_to_fine_steps):
                    if i == 0:
                        component = utils.prepare_gcn_component(init_polys.numpy(),
                                                                self.psp_feature,
                                                                init_polys.size()[1],
                                                                n_adj=self.n_adj)
                        init_polys = init_polys.cuda()
                        adjacent = component['adj_matrix'].cuda()
                        init_poly_idx = component['feature_indexs'].cuda()
                        cnn_feature = []
                        try:
                            cnn_feature = self.edge_annotation.sampling(init_poly_idx, conv_layers)
                        except RuntimeError:
                            print(conv_layers[0].shape, rois_feature[0].shape)
                        input_feature = torch.cat((cnn_feature, init_polys), 2)

                    else:
                        init_polys = gcn_pred_poly
                        cnn_feature = interpolated_sum(conv_layers, init_polys, self.psp_feature)
                        input_feature = torch.cat((cnn_feature, init_polys), 2)

                    gcn_pred = self.gnn[i].forward(input_feature, adjacent)
                    gcn_pred_poly = init_polys.to(device) + gcn_pred

                    pred_polys.append(gcn_pred_poly)
                    adjacent = adjacent
                pred_polys = pred_polys[-1]

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, pred_polys,
                    target_polys, mrcnn_mask, target_mask, vertex_target_masks, edge_target_masks, vertex_logits, edge_logits, detection_boxes.squeeze(0)*scale]

    def train_model(self, train_generator, val_generator, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting which layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

        # Pre-defined layer regular expressions
        layers_to_train = layers
        flag = 0
        if self.epoch >= epochs:
            return None
        layer_regex = {
            # all layers but the backbone
            "points_only": r"(mask.*)|(first_annotation.*)|(edge_annotation.*)|(gnn.*)",
            "points": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(first_annotation.*)|(edge_annotation.*)|(gnn.*)",
            # "points": r"(edge_annotation.*)|(gnn.*)",
            # "fp": r"(mask.*)|(first_annotation.*)",
            "boxes": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # "heads": r"(mask.*)|(first_annotation.*)|(edge_annotation.*)|(gnn.*)",
            # From a specific Resnet stage and up
            "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(first_annotation.*)|(edge_annotation.*)|(gnn.*)",
            "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(first_annotation.*)|(edge_annotation.*)|(gnn.*)",
            "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(first_annotation.*)|(edge_annotation.*)|(gnn.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        # train_set = Dataset(train_dataset, self.config, augment=True)
        # train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
        # val_set = Dataset(val_dataset, self.config, augment=True)
        # val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch + 1, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        if layers_to_train != 'boxes':
            flag = 1
            no_wd = []
            wd = []
            print('Weight Decay applied to: ')

            for name, p in self.named_parameters():
                if not p.requires_grad:
                    # No optimization for frozen params
                    continue

                if 'bn' in name or 'bias' in name:
                    no_wd.append(p)
                else:
                    wd.append(p)
                    # print(name)

            # Allow individual options
            self.optimizer = optim.Adam(
                [
                    {'params': no_wd, 'weight_decay': 0.0},
                    {'params': wd}
                ],
                lr=3e-5,
                # lr=3e-7,
                weight_decay=1e-5,
                amsgrad=False)

            self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=7,
                                                      gamma=0.1)
        else:
            # Optimizer object
            # Add L2 Regularization
            # Skip gamma and beta weights of batch normalization layers.
            trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
            weight_wo_bn_name = [name for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
            trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
            weight_only_bn_name = [name for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
            self.optimizer = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

            self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=7,
                                                      gamma=0.1)

        for epoch in range(self.epoch+1, epochs+1):
            if flag == 1:
                if (self.global_step % 3000 == 0):
                    self.lr_decay.step()
                    print('LR is now: ', self.optimizer.param_groups[0]['lr'])
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_poly_matching_sum, \
            loss_vertex_mask_sum, loss_edge_mask_sum = \
                self.train_epoch(train_generator, optimizer=self.optimizer, steps=self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss_sum, val_loss_rpn_class_sum, val_loss_rpn_bbox_sum, val_loss_mrcnn_class_sum, val_loss_mrcnn_bbox_sum, val_loss_mrcnn_mask_sum, val_loss_poly_matching_sum, \
            val_loss_vertex_mask_sum, val_loss_edge_mask_sum = \
                self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_poly_matching_sum,
                                        loss_vertex_mask_sum, loss_edge_mask_sum])
            self.val_loss_history.append([val_loss_sum, val_loss_rpn_class_sum, val_loss_rpn_bbox_sum, val_loss_mrcnn_class_sum,
                                          val_loss_mrcnn_bbox_sum, val_loss_mrcnn_mask_sum, val_loss_poly_matching_sum,
                                          val_loss_vertex_mask_sum, val_loss_edge_mask_sum])
            visualize.plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            self.save_checkpoint(epoch, self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_poly_matching_sum = 0
        loss_vertex_mask_sum = 0
        loss_edge_mask_sum = 0
        iousum = 0
        loss_mrcnn_mask_sum = 0
        step = 0
        accum = defaultdict(float)

        for inputs in datagenerator:

            optimizer.zero_grad()

            batch_count += 1

            images = inputs['fwd_img']
            image_metas = inputs['image_meta']
            rpn_match = inputs['rpn_match']
            rpn_bbox = inputs['rpn_bbox']
            gt_class_ids = inputs['class_ids'].int()
            gt_boxes = inputs['bboxes']
            # gt_sboxes = inputs['sboxes']
            gt_masks = inputs['masks']
            vertex_masks = inputs['vertex_masks']
            edge_masks = inputs['edge_masks']
            fwd_polys = inputs['fwd_polys']
            gt_polys = inputs['gt_polys']
            nor_gt_polys = inputs['nor_polys']

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            point_loss_weight = 1.0

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                # gt_sboxes = gt_sboxes.cuda()
                gt_masks = gt_masks.cuda()
                # gt_polys = gt_polys.cuda()
                vertex_masks = vertex_masks.cuda()
                edge_masks = edge_masks.cuda()
                nor_gt_polys = nor_gt_polys.cuda()
                # fwd_polys = fwd_polys.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, \
            pred_polys, target_polys, masks, target_mask, vertex_target_masks, edge_target_masks, vertex_logits, edge_logits, rois = \
                self.predict(
                    [images, image_metas, gt_class_ids, gt_boxes, gt_masks, vertex_masks, edge_masks, nor_gt_polys,
                     fwd_polys], mode='training')

            # if pred_polys.size()[0]:
            #     try:
            #         pred_polys = utils.sample_point(pred_polys, cp_num=self.cp_num, p_num=self.p_num)
            #     except:
            #         print(pred_polys.size())
            #         continue

            # Debug the training results
            # if self.config.DEBUG:
            #     bbox = mrcnn_bbox.data.cpu().numpy()
            #     # mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            #     # vertex_logits = vertex_logits.data.cpu().numpy()
            #     # edge_logits = vertex_logits.data.cpu().numpy()
            #     pred_polys_debug = pred_polys.data.cpu().numpy()
            #     images = images.squeeze(0)
            #     images = images.permute(1, 2, 0)
            #     images = images.data.cpu().numpy
            #
            #     figsize = (16, 16)
            #     _, ax = plt.subplots(1, figsize=figsize)
            #     auto_show = True
            #     height, width = (30, 30)
            #     ax.set_ylim(height + 10, -10)
            #     ax.set_xlim(-10, width + 10)
            #     ax.axis('off')
            #     title = 'gt'
            #     ax.set_title(title)
            #
            #     masked_image = images.astype(np.uint8).copy() + self.config.MEAN_PIXEL
            #     N = pred_polys_debug.shape[0]
            #     colors = utils.random_colors(N)
            #     for i in range(0, N):
            #         # y1, x1, y2, x2 = bboxes[i]
            #         # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
            #         #                       alpha=0.7, linestyle="dashed",
            #         #                     facecolor='none')
            #         # ax.add_patch(p)
            #         color = colors[i]
            #         color1 = colors[np.int((i / 2))]
            #         poly = pred_polys_debug[i, :, :] * 28
            #         poly = poly.astype(np.int)
            #
            #
            #
            #         pred_mask = np.zeros((height, width), dtype=np.uint8)
            #
            #         # masked_image = utils.apply_mask(masked_image, vertex_masks[:, :, i], color1)
            #         #
            #         # masked_image = utils.apply_mask(masked_image, edge_masks[:, :, i], color)
            #
            #         pred_mask = utils.draw_poly(pred_mask, poly, pred_mask, color=0)
            #
            #         # masked_image - utils.draw_box(masked_image, bboxes[i], color=0)
            #
            #         ax.imshow(pred_mask.astype(np.uint8))
            #         if auto_show:
            #             plt.show()
            #
            #     # final_rois, final_class_ids, final_scores, final_polys = \
            #     #     self.unmold_detections(detections, pred_polys,
            #     #                            orig_images.shape, windows[0])

            # Compute losses
            # if pred_polys.size()[0]:
            #     pred_sampled = self.boundary.sample_point(pred_polys, newpnum=self.p_num)
            # else:
            pred_sampled = pred_polys
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, gt_right_order, poly_mathcing_loss, fp_vertex_loss, fp_edge_loss = \
                LossFunction.compute_losses_no_edge(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox,
                                                    target_class_ids,
                                                    mrcnn_class_logits, target_deltas, mrcnn_bbox, masks, target_mask,
                                                    pred_sampled, target_polys, vertex_target_masks, edge_target_masks, vertex_logits, edge_logits, self.config)

            # loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + fp_vertex_loss + fp_edge_loss + poly_mathcing_loss
            #without fp vertex loss
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + poly_mathcing_loss + fp_vertex_loss + fp_edge_loss


            # Backpropagation
            try:
                loss.backward()
            except RuntimeError:
                 print(mrcnn_bbox.shape, target_polys.shape, pred_polys.shape)
                 print(self.named_parameters())
                 continue
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 40)

            # if (batch_count % self.config.BATCH_SIZE) == 0:
            optimizer.step()
            batch_count = 0

            # Progress
            with torch.no_grad():

                #calculate IOU
                if pred_polys.size()[0]:
                    # Only positive ROIs contribute to the loss. And only
                    # the class specific mask of each ROI.
                    positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
                    positive_class_ids = target_class_ids[positive_ix.data].long()
                    indices = torch.stack((positive_ix, positive_class_ids), dim=1)

                    # Gather the masks (predicted and true) that contribute to loss
                    orig_poly = target_polys[indices[:, 0].data, :, :]
                    preds = pred_sampled[indices[:, 0].data, :, :]
                    rois = rois[indices[:, 0].data, :]
                    pre_masks = masks[indices[:,0].data, indices[:,1].data, :, :]
                    gt_masks = target_mask[indices[:,0].data, :, :]


                    orig_poly = orig_poly.data.cpu().numpy()
                    preds = preds.detach().data.cpu().numpy()
                    rois = rois.detach().data.cpu().numpy()
                    rois = rois.astype(np.int32)
                    pre_masks = pre_masks.detach().data.cpu().numpy()
                    gt_masks = gt_masks.detach().data.cpu().numpy()

                    iou = 0
                    for i in range(preds.shape[0]):
                        curr_pred_poly = np.floor(preds[i] * 224).astype(np.int32)
                        curr_gt_poly = np.floor(orig_poly[i] * 224).astype(np.int32)

                        cur_iou, masks = utils.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                                                               np.array(curr_gt_poly, dtype=np.int32),
                                                               224,
                                                               224)
                        roi = rois[i]
                        iou += cur_iou
                        pre_mask = pre_masks[i]
                        gt_mask = gt_masks[i]
                        pre_mask = np.where(pre_mask >= 0.5, 255, 0).astype(np.uint8)
                        gt_mask = np.where(gt_mask >= 0.5, 255, 0).astype(np.uint8)

                    iou = iou / preds.shape[0]
                    gt_shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                    smooth_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                    gt_shape_mask = utils.draw_poly_line(gt_shape_mask, curr_gt_poly)
                    # shape_mask = utils.draw_poly_line(shape_mask, curr_predgcn_poly)
                    smooth_mask = utils.draw_poly_line(smooth_mask, curr_pred_poly)

                    img = images.squeeze(0).detach().data.cpu().numpy()

                else:
                    iou = 0
                    preds = np.zeros(2, dtype=np.float32)
                    masks = np.zeros((2, 224, 224), dtype=np.int32)
                    gt_shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                    smooth_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                    roi = np.zeros(4, dtype=np.int32)
                    roi = [400, 400, 450, 450]
                    img = images.squeeze(0).detach().data.cpu().numpy()
                    pre_mask = np.zeros((28, 28), dtype=np.float32)
                    gt_mask = np.zeros((28, 28), dtype=np.float32)


                accum['loss'] += float(loss.item())
                accum['point_loss'] += float(poly_mathcing_loss.item())
                accum['edge_loss'] += float(fp_vertex_loss.cpu().item() + fp_edge_loss.cpu().item())
                accum['iou'] += iou
                accum['length'] += 1
                accum['mask_loss'] += float(mrcnn_mask_loss.item())
                accum['bbox_loss'] += float(mrcnn_bbox_loss.item())
                accum['rpn_loss'] += float(rpn_bbox_loss.item())
                # if accum['iou'] > 20:
                #     print("Oh no")

                printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                 suffix="Train- Global Steps: {:.1f} Complete -loss: {:.3f} - rpn_class_loss: {:.3f} - rpn_bbox_loss: {:.3f} - mrcnn_class_loss: {:.3f} "
                                        "- mrcnn_bbox_loss: {:.3f} - mask_loss: {:.3f} - points_loss: {:.3f}  - IOU: {:.3f} - vertex_loss: {:.2f} - edge_loss: {:.3f} - N: {:.1f}".format(
                                     self.global_step, loss.data.cpu().item(), rpn_class_loss.data.cpu().item(),
                                     rpn_bbox_loss.data.cpu().item(),
                                     mrcnn_class_loss.data.cpu().item(), mrcnn_bbox_loss.data.cpu().item(),
                                     mrcnn_mask_loss.data.cpu().item(), poly_mathcing_loss.cpu().item(), accum['iou']/accum['length'],
                                     fp_vertex_loss.cpu().item(), fp_edge_loss.cpu().item(), preds.shape[0]), length=10)

                if step % self.print_freq == 0:
                    # Mean of accumulated values
                    for k in accum.keys():
                        if k == 'length':
                            continue
                        accum[k] /= accum['length']

                    # Add summaries
                    # masks = np.expand_dims(masks, -1).astype(np.uint8)  # Add a channel dimension
                    # masks = np.tile(masks, [1, 1, 1, 3])  # Make [2, H, W, 3]
                    # img = (img[-1, ...])
                    img = np.transpose(img, [1, 2, 0])  # Make [H, W, 3]
                    img = (img + self.config.MEAN_PIXEL).astype(np.uint8)
                    img = img[roi[0]:roi[2], roi[1]:roi[3], :]
                    img = skimage.transform.resize(img, [224, 224], order=1, mode="reflect")

                    pre_mask = np.expand_dims(pre_mask, -1).astype(np.uint8)  # Add a channel dimension
                    pre_mask = np.tile(pre_mask, [1, 1, 3])  # Make [2, H, W, 3]

                    gt_mask = np.expand_dims(gt_mask, -1).astype(np.uint8)  # Add a channel dimension
                    gt_mask = np.tile(gt_mask, [1, 1, 3])  # Make [2, H, W, 3]
                    #
                    #
                    # self.writer.add_image('pred_mask', masks[0], self.global_step)
                    # self.writer.add_image('gt_mask', masks[1], self.global_step)
                    self.writer.add_image('pre_polyline', smooth_mask, self.global_step)
                    self.writer.add_image('gt_polyline', gt_shape_mask, self.global_step)
                    self.writer.add_image('image', img, self.global_step)
                    self.writer.add_image('pre_mask', pre_mask, self.global_step)
                    self.writer.add_image('gt_mask', gt_mask, self.global_step)

                    for k in accum.keys():
                        if k == 'length':
                            continue
                        self.writer.add_scalar(k, accum[k], self.global_step)

                    accum = defaultdict(float)

                # Statistics
                loss_sum += loss.data.cpu().item() / steps
                loss_rpn_class_sum += rpn_class_loss.data.cpu().item() / steps
                loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu().item() / steps
                loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu().item() / steps
                loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu().item() / steps
                loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu().item()/steps
                loss_poly_matching_sum += poly_mathcing_loss.cpu().item() / steps
                loss_vertex_mask_sum += 0
                loss_edge_mask_sum += 0

            del (masks, pred_polys, preds, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox)

            # Break after 'steps' steps
            self.global_step += 1
            if (self.global_step % 3000 == 0):
                self.lr_decay.step()
                print('LR is now: ', self.optimizer.param_groups[0]['lr'])
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_poly_matching_sum, \
               loss_vertex_mask_sum, loss_edge_mask_sum

    @torch.no_grad()
    def valid_epoch(self, datagenerator, steps):

        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_poly_matching_sum = 0
        loss_vertex_mask_sum = 0
        loss_edge_mask_sum = 0
        loss_mrcnn_mask_sum = 0
        iousum = 0
        accum = defaultdict(float)

        for inputs in datagenerator:
            # print("1")
            images = inputs['fwd_img']
            image_metas = inputs['image_meta']
            rpn_match = inputs['rpn_match']
            rpn_bbox = inputs['rpn_bbox']
            gt_class_ids = inputs['class_ids'].int()
            gt_boxes = inputs['bboxes']
            # gt_sboxes = inputs['sboxes']
            gt_masks = inputs['masks']
            vertex_masks = inputs['vertex_masks']
            edge_masks = inputs['edge_masks']
            fwd_polys = inputs['fwd_polys']
            gt_polys = inputs['gt_polys']
            nor_gt_polys = inputs['nor_polys']

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            point_loss_weight = 1.0

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                # gt_sboxes = gt_sboxes.cuda()
                gt_masks = gt_masks.cuda()
                # gt_polys = gt_polys.cuda()
                vertex_masks = vertex_masks.cuda()
                edge_masks = edge_masks.cuda()
                nor_gt_polys = nor_gt_polys.cuda()
                # fwd_polys = fwd_polys.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, \
            pred_polys, target_polys, masks, target_mask, vertex_target_masks, edge_target_masks, vertex_logits, edge_logits, rois = \
                self.predict(
                    [images, image_metas, gt_class_ids, gt_boxes, gt_masks, vertex_masks, edge_masks,
                     nor_gt_polys,
                     fwd_polys], mode='training')


            if not target_class_ids.size()[0]:
                continue

            # if pred_polys.size()[0]:
            #     try:
            #         pred_polys = utils.sample_point(pred_polys, cp_num=self.cp_num, p_num=self.p_num)
            #     except:
            #         print(pred_polys.size())
            #         continue

            # Compute losses
            # if pred_polys.size()[0]:
            #     pred_sampled = self.boundary.sample_point(pred_polys, newpnum=self.p_num)
            #
            # else:
            pred_sampled = pred_polys
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, gt_right_order, poly_mathcing_loss, fp_vertex_loss, fp_edge_loss = \
                LossFunction.compute_losses_no_edge(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox,
                                                    target_class_ids,
                                                    mrcnn_class_logits, target_deltas, mrcnn_bbox, masks, target_mask,
                                                    pred_sampled, target_polys, vertex_target_masks, edge_target_masks,
                                                    vertex_logits, edge_logits, self.config)

            # loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + fp_vertex_loss + fp_edge_loss + poly_mathcing_loss
            # without fp vertex loss
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + poly_mathcing_loss + fp_vertex_loss + fp_edge_loss

            # calculate IOU
            if pred_polys.size()[0]:
                # Only positive ROIs contribute to the loss. And only
                # the class specific mask of each ROI.
                positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
                positive_class_ids = target_class_ids[positive_ix.data].long()
                indices = torch.stack((positive_ix, positive_class_ids), dim=1)

                # Gather the masks (predicted and true) that contribute to loss
                orig_poly = target_polys[indices[:, 0].data, :, :]
                preds = pred_sampled[indices[:, 0].data, :, :]
                rois = rois[indices[:, 0].data, :]
                pre_masks = masks[indices[:, 0].data, indices[:, 1].data, :, :]
                gt_masks = target_mask[indices[:, 0].data, :, :]

                orig_poly = orig_poly.data.cpu().numpy()
                preds = preds.detach().data.cpu().numpy()
                rois = rois.detach().data.cpu().numpy()
                rois = rois.astype(np.int32)
                pre_masks = pre_masks.detach().data.cpu().numpy()
                gt_masks = gt_masks.detach().data.cpu().numpy()

                iou = 0
                for i in range(preds.shape[0]):
                    curr_pred_poly = np.floor(preds[i] * 224).astype(np.int32)
                    curr_gt_poly = np.floor(orig_poly[i] * 224).astype(np.int32)

                    cur_iou, masks = utils.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                                                         np.array(curr_gt_poly, dtype=np.int32),
                                                         224,
                                                         224)
                    iou += cur_iou
                    roi = rois[i]
                    pre_mask = pre_masks[i]
                    gt_mask = gt_masks[i]
                    pre_mask = np.where(pre_mask >= 0.5, 255, 0).astype(np.uint8)
                    gt_mask = np.where(gt_mask >= 0.5, 255, 0).astype(np.uint8)

                iou = iou / preds.shape[0]
                gt_shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                smooth_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                gt_shape_mask = utils.draw_poly_line(gt_shape_mask, curr_gt_poly)
                # shape_mask = utils.draw_poly_line(shape_mask, curr_predgcn_poly)
                smooth_mask = utils.draw_poly_line(smooth_mask, curr_pred_poly)

                img = images.squeeze(0).detach().data.cpu().numpy()
            else:
                iou = 0
                preds = np.zeros(2, dtype=np.float32)
                masks = np.zeros((2, 224, 224), dtype=np.int32)
                gt_shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                smooth_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                roi = np.zeros(4, dtype=np.int32)
                roi = [400, 400, 450, 450]
                img = images.squeeze(0).detach().data.cpu().numpy()
                pre_mask = np.zeros((28, 28), dtype=np.float32)
                gt_mask = np.zeros((28, 28), dtype=np.float32)

            iousum += iou
            accum['loss'] += float(loss.item())
            accum['point_loss'] += float(poly_mathcing_loss.item())
            accum['edge_loss'] += float(fp_vertex_loss.cpu().item() + fp_edge_loss.cpu().item())
            accum['iou'] += iou
            accum['length'] += 1
            accum['mask_loss'] += float(mrcnn_mask_loss.item())
            accum['bbox_loss'] += float(mrcnn_bbox_loss.item())
            accum['rpn_loss'] += float(rpn_bbox_loss.item())

            printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                             suffix="Global Steps: {:.1f} Complete -loss: {:.3f} - rpn_class_loss: {:.3f} - rpn_bbox_loss: {:.3f} - mrcnn_class_loss: {:.3f} "
                                    "- mrcnn_bbox_loss: {:.3f} - mask_loss: {:.3f} - points_loss: {:.3f}  - IOU: {:.3f} - vertex_loss: {:.2f} - edge_loss: {:.3f}".format(
                                 self.global_step, loss.data.cpu().item(), rpn_class_loss.data.cpu().item(),
                                 rpn_bbox_loss.data.cpu().item(),
                                 mrcnn_class_loss.data.cpu().item(), mrcnn_bbox_loss.data.cpu().item(),
                                 mrcnn_mask_loss.data.cpu().item(), poly_mathcing_loss.cpu().item(),
                                 accum['iou'] / accum['length'],
                                 fp_vertex_loss.cpu().item(), fp_edge_loss.cpu().item()), length=10)

            # Statistics
            loss_sum += loss.data.cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.data.cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu().item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu().item() / steps
            loss_poly_matching_sum += poly_mathcing_loss.cpu().item() / steps
            loss_vertex_mask_sum += 0
            loss_edge_mask_sum += 0
            # loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu().item()/steps

            # Break after 'steps' steps
            del (pred_polys, preds, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox)

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        for k in accum.keys():
            if k == 'length':
                continue
            accum[k] /= accum['length']
            self.val_writer.add_scalar(k, accum[k], self.global_step)

        # masks = np.expand_dims(masks, -1).astype(np.uint8)  # Add a channel dimension
        # masks = np.tile(masks, [1, 1, 1, 3])  # Make [2, H, W, 3]
        # img = (data['img'].cpu().numpy()[-1, ...] * 255).astype(np.uint8)
        # img = np.transpose(img, [1, 2, 0])  # Make [H, W, 3]

        # self.val_writer.add_image('pred_mask', masks[0], self.global_step)
        # self.val_writer.add_image('gt_mask', masks[1], self.global_step)
        # self.val_writer.add_image('image', img, self.global_step)
        img = np.transpose(img, [1, 2, 0])  # Make [H, W, 3]
        img = (img + self.config.MEAN_PIXEL).astype(np.uint8)
        img = img[roi[0]:roi[2], roi[1]:roi[3], :]
        img = skimage.transform.resize(img, [224, 224], order=1, mode="reflect")

        pre_mask = np.expand_dims(pre_mask, -1).astype(np.uint8)  # Add a channel dimension
        pre_mask = np.tile(pre_mask, [1, 1, 3])  # Make [2, H, W, 3]

        gt_mask = np.expand_dims(gt_mask, -1).astype(np.uint8)  # Add a channel dimension
        gt_mask = np.tile(gt_mask, [1, 1, 3])  # Make [2, H, W, 3]

        # self.writer.add_image('pred_mask', masks[0], self.global_step)
        # self.writer.add_image('gt_mask', masks[1], self.global_step)
        self.val_writer.add_image('pre_polyline', smooth_mask, self.global_step)
        self.val_writer.add_image('gt_polyline', gt_shape_mask, self.global_step)
        self.val_writer.add_image('image', img, self.global_step)
        self.val_writer.add_image('pre_mask', pre_mask, self.global_step)
        self.val_writer.add_image('gt_mask', gt_mask, self.global_step)

        final_iou = np.mean(accum['iou'])
        print('[VAL] IoU: %f' % final_iou)
        del (masks, smooth_mask, gt_shape_mask)

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_poly_matching_sum, \
                   loss_vertex_mask_sum, loss_edge_mask_sum

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, pre_polys, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        pre_polys: [N, x, y]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        # masks = mrcnn_mask[np.arange(N), :, :, class_ids]
        polys = pre_polys[:N, :]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes and polygon coordinates to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # polys = np.multiply(polys - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            polys = np.delete(polys, exclude_ix, axis=0)
            N = class_ids.shape[0]

        polys = unmold_polys(polys, boxes).astype(np.int32)
        # polys = np.multiply(polys - shifts, scales).astype(np.int32)

        # Resize masks to original image size and set boundary threshold.
        # full_masks = []
        # for i in range(N):
        #     # Convert neural network mask to full size mask
        #     full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
        #     full_masks.append(full_mask)
        # full_masks = np.stack(full_masks, axis=-1)\
        #     if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, polys

    @torch.no_grad()
    def detect(self, test_generator):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        results = []

        for i, inputs in enumerate(test_generator):

            molded_images = inputs['fwd_img']
            orig_images = inputs['orig_img']
            image_metas = inputs['image_meta']
            fwd_polys = inputs['fwd_polys']
            image_metas = inputs['image_meta']
            windows = inputs['window']
            image_info = inputs['image_info'][0]
            image_id = image_info['id']

            orig_images = orig_images.squeeze().numpy()
            image_metas = image_metas.numpy()
            fwd_polys = fwd_polys.numpy()
            # windows = windows.numpy()

            # molded_images, image_metas, windows = self.mold_inputs(images)

            # Convert images to torch tensor
            # molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

            # To GPU
            if self.config.GPU_COUNT:
                molded_images = molded_images.cuda()
                # fwd_polys = fwd_polys.cuda()

            # Run object detection
            detections, pred_polys = \
                self.predict([molded_images, image_metas, fwd_polys], mode='inference')



            # Convert to numpy
            detections = detections.data.cpu().numpy()
            # mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            # vertex_logits = vertex_logits.data.cpu().numpy()
            # edge_logits = vertex_logits.data.cpu().numpy()
            pred_polys = pred_polys.data.cpu().numpy()
            # adjacent = adjacent.data.cpu().numpy()
            # Process detections

            final_rois, final_class_ids, final_scores, final_polys =\
                self.unmold_detections(detections[0], pred_polys,
                                       orig_images.shape, windows[0])
            results.append({
                "image": orig_images,
                'image_id': image_id,
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "polys": final_polys,
            })
        return results


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

def unmold_polys(polys, boxes):
    N = boxes.shape[0]
    scaled_polys = polys.copy()
    for i in range(0, N):
        box = boxes[i]
        poly = polys[i]
        x1, y1, x2, y2 = box[1], box[0], box[3], box[2]
        h, w = [y2 - y1, x2 - x1]
        poly[:, 0] = poly[:, 0]*h + y1
        poly[:, 1] = poly[:, 1]*w + x1
        scaled_polys[i] = poly

    return scaled_polys




