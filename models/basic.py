import numpy as np
import math
import re
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import img_as_ubyte

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms

from utils import Utils as utils
from core.nms.nms_wrapper import nms
from core.roialign.roi_align.crop_and_resize import CropAndResizeFunction


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2


# decov layer definition
def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    # image_id = meta[:, 0]
    image_shape = meta[:, 0:3]
    window = meta[:, 3:7]   # (y1, x1, y2, x2) window of image in in pixels
    # active_class_ids = meta[:, 8:]
    return image_shape, window


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


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


############################################################
#  Resnet Graph
############################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None
        # self.C6 = SkipFeatures(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        # x = self.C6(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


############################################################
#  FPN Graph
############################################################

class TopDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x, y):
        y = F.interpolate(y, scale_factor=2)
        x = self.conv1(x)
        return self.conv2(self.padding2(x+y))


class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        # self.C6 = C6
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        # skip = x
        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.interpolate(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.interpolate(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.interpolate(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out]


class SkipFeatures(nn.Module):
    """
        Pyramid Scene Parsing module
        """

    def __init__(self, in_features=256, out_features=256, sizes=(1, 2, 3, 6), n_classes=1):
        super(SkipFeatures, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes) // 4 + 1), out_features)
        self.relu = nn.ReLU()

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features // 4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features // 4, affine=True)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=True)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # import ipdb
        # ipdb.set_trace()
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        # out = self.final(bottle)

        return bottle


############################################################
#  Region Proposal Network
############################################################

class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        # rpn_probs = self.softmax(rpn_class_logits)
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    std_dev = (torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float())
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :] # TODO: Support batch size > 1 ff.
    anchors = anchors[order.data, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = (torch.from_numpy(np.array([height, width, height, width])).float())
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    #find corresponding fpn scale for the boxes
    image_area = (torch.FloatTensor([float(image_shape[0]*image_shape[1])]))
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)


    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


############################################################
#  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_sboxes, gt_masks, vertex_masks, edge_masks, gt_polys, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    # Currently only supports batchsize 1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    # gt_sboxes = gt_sboxes.squeeze(0)
    gt_polys = gt_polys.squeeze(0)
    gt_masks = gt_masks.squeeze(0)
    vertex_masks = vertex_masks.squeeze(0)
    edge_masks = edge_masks.squeeze(0)

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    # if torch.nonzero(gt_class_ids < 0).size()[0]:
    #     crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
    #     non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
    #     crowd_boxes = gt_boxes[crowd_ix.data, :]
    #     crowd_masks = gt_masks[crowd_ix.data, :, :]
    #     gt_class_ids = gt_class_ids[non_crowd_ix.data]
    #     gt_boxes = gt_boxes[non_crowd_ix.data, :]
    #     gt_masks = gt_masks[non_crowd_ix.data, :]
    #
    #     # Compute overlaps with crowd boxes [anchors, crowds]
    #     crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
    #     crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
    #     no_crowd_bool = crowd_iou_max < 0.001
    # else:
    #     no_crowd_bool =  Variable(torch.ByteTensor(proposals.size()[0]*[True]), requires_grad=False)
    #     if config.GPU_COUNT:
    #         no_crowd_bool = no_crowd_bool.cuda()

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    # Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).size()[0]:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data, :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        roi_gt_polys = gt_polys[roi_gt_box_assignment.data, :, :]

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement(positive_rois.data, roi_gt_boxes.data)
        std_dev = torch.from_numpy(config.BBOX_STD_DEV).float()
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data, :, :]
        roi_vertex_masks = vertex_masks[roi_gt_box_assignment.data, :, :]
        roi_edge_masks = edge_masks[roi_gt_box_assignment.data, :, :]


        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()
        masks = CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data
        vertex_target_masks = CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_vertex_masks.unsqueeze(1), boxes, box_ids).data
        edge_target_masks = CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_edge_masks.unsqueeze(1), boxes, box_ids).data
        masks = masks.squeeze(1)
        vertex_target_masks = vertex_target_masks.squeeze(1)
        edge_target_masks = edge_target_masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
        vertex_target_masks = torch.round(vertex_target_masks)
        edge_target_masks = torch.round(edge_target_masks)
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool
    # negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).size()[0] and positive_count>0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = (torch.zeros(negative_count)).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = (torch.zeros(negative_count, 4))
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = (torch.zeros(negative_count, config.PNUM, 2))
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        # try:
        roi_gt_polys = torch.cat([roi_gt_polys, zeros], dim=0)
        # except RuntimeError:
            # print(roi_gt_polys.shape, gt_polys.shape, gt_class_ids.shape, gt_boxes.shape, gt_masks.shape, masks.shape, roi_gt_box_assignment.shape)
            # roi_gt_polys = torch.cat([roi_gt_polys, zeros], dim=0)
        zeros = (torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]))
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)
        roi_masks = torch.cat([roi_masks, zeros], dim=0)
        vertex_target_masks = torch.cat([vertex_target_masks, zeros], dim=0)
        edge_target_masks = torch.cat([edge_target_masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = (torch.zeros(negative_count))
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = (torch.zeros(negative_count, 4)).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = (torch.zeros(negative_count, config.PNUM, 2))
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_polys = zeros
        zeros = (torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]))
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
        roi_masks = zeros
        vertex_target_masks = zeros
        edge_target_masks = zeros
    else:
        rois = (torch.FloatTensor())
        roi_gt_class_ids = (torch.IntTensor())
        deltas = (torch.FloatTensor())
        roi_gt_polys = (torch.FloatTensor())
        masks = (torch.FloatTensor())
        roi_masks = (torch.FloatTensor())
        vertex_target_masks = (torch.FloatTensor())
        edge_target_masks = (torch.FloatTensor())
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            roi_gt_polys = roi_gt_polys.cuda()
            masks = masks.cuda()
            roi_masks = roi_masks.cuda()
            vertex_target_masks = vertex_target_masks.cuda()
            edge_target_masks = edge_target_masks.cuda()

    return rois, roi_gt_class_ids, deltas, roi_gt_polys, masks, roi_masks, vertex_target_masks, edge_target_masks


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    return boxes


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, window = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

    return detections


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = (torch.from_numpy(np.array([height, width, height, width])).float())
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale

    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep_bool = class_ids>0

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    # keep = torch.nonzero(keep_bool)
    # keep = keep[:, 0]

    if torch.nonzero(keep_bool).size()[0]:
        keep = torch.nonzero(keep_bool)[:, 0]
    else:
        keep = torch.from_numpy(np.array([1]))

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep.data]
    pre_nms_scores = class_scores[keep.data]
    pre_nms_rois = refined_rois[keep.data]

    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]

        # Sort
        ix_rois = pre_nms_rois[ixs.data]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data,:]

        class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)

        # Map indicies
        class_keep = keep[ixs[order[class_keep].data].data]

        if i==0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)

    # Keep top detections
    roi_count = int(config.TRA_MAX_INSTANCES*0.33)
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1)), dim=1)

    return result


def detection_layer_gcn(config, rois, mrcnn_class, mrcnn_bbox, target_class_ids, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, window = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections_gcn(rois, mrcnn_class, mrcnn_bbox, target_class_ids, window, config)

    return detections


def refine_detections_gcn(rois, probs, deltas, target_class_ids, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
    positive_class_ids = target_class_ids[positive_ix.data].long()
    indices = torch.stack((positive_ix, positive_class_ids), dim=1)

    # Gather the masks (predicted and true) that contribute to loss

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)
    #
    # # Class probability of the top class of each ROI
    # # Class-specific bounding box deltas
    # idx = torch.arange(class_ids.size()[0]).long()
    # if config.GPU_COUNT:
    #     idx = idx.cuda()
    # class_scores = probs[indices[:, 0].data, class_ids.data]
    deltas_specific = deltas[indices[:, 0].data, indices[:,1].data]
    rois = rois[indices[:, 0].data]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = (torch.from_numpy(np.array([height, width, height, width])).float())
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale

    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    # keep_bool = class_ids>0
    #
    # # Filter out low confidence boxes
    # if config.DETECTION_MIN_CONFIDENCE:
    #     keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    # # keep = torch.nonzero(keep_bool)
    # # keep = keep[:, 0]
    #
    # if torch.nonzero(keep_bool).size()[0]:
    #     keep = torch.nonzero(keep_bool)[:, 0]
    # else:
    #     keep = torch.from_numpy(np.array([1]))
    #
    # # Apply per-class NMS
    # pre_nms_class_ids = class_ids[keep.data]
    # pre_nms_scores = class_scores[keep.data]
    # pre_nms_rois = refined_rois[keep.data]
    #
    # for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
    #     # Pick detections of this class
    #     ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]
    #
    #     # Sort
    #     ix_rois = pre_nms_rois[ixs.data]
    #     ix_scores = pre_nms_scores[ixs]
    #     ix_scores, order = ix_scores.sort(descending=True)
    #     ix_rois = ix_rois[order.data,:]
    #
    #     class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
    #
    #     # Map indicies
    #     class_keep = keep[ixs[order[class_keep].data].data]
    #
    #     if i==0:
    #         nms_keep = class_keep
    #     else:
    #         nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    # keep = intersect1d(keep, nms_keep)
    #
    # # Keep top detections
    # roi_count = int(config.TRA_MAX_INSTANCES*0.33)
    # top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    # keep = keep[top_ids.data]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = refined_rois

    return result





############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois]+x, self.pool_size, self.image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.sigmoid(mrcnn_class_logits)
        # mrcnn_probs = mrcnn_class_logits

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


# class CoarseMask(nn.Module):
#     def __init__(self, depth, pool_size, image_shape, num_classes):
#         super(CoarseMask, self).__init__()
#         act_fn_2 = nn.ReLU()
#         self.depth = depth
#         self.pool_size = pool_size
#         self.image_shape = image_shape
#         self.num_classes = num_classes
#         self.padding = SamePad2d(kernel_size=3, stride=1)
#         self.conv1 = nn.Conv2d(self.depth, self.depth, kernel_size=1, stride=1)
#         self.bn1 = nn.BatchNorm2d(self.depth, eps=0.001)
#         self.conv2 = nn.Conv2d(self.depth, self.depth, kernel_size=1, stride=1)
#         self.bn2 = nn.BatchNorm2d(self.depth, eps=0.001)
#         self.conv3 = nn.Conv2d(self.depth, num_classes, kernel_size=1, stride=1)
#         # self.bn3 = nn.BatchNorm2d(num_classes, eps=0.001)
#         # self.conv4 = nn.Conv2d(self.depth, num_classes, kernel_size=1, stride=1)
#         self.pool = AveSupPixPool()
#         self.unpool = SupPixUnpool()
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x, mask):
#         pld = self.pool(x, mask)
#         unpld = self.unpool(pld, mask)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         unpld = self.conv2(unpld)
#         unpld = self.bn2(unpld)
#         unpld = self.relu(unpld)
#         x = x + unpld
#         x = self.conv3(x)
#         x = self.sigmoid(x)
#
#         return x


class Mask(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        act_fn_2 = nn.ReLU()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(256, eps=0.001)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(256, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv6 = nn.Conv2d(256, 2, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        features = self.relu(x)
        x = self.conv6(features)
        x = self.sigmoid(x)

        return features, x


class Edge_Annotation(nn.Module):
    def __init__(self, in_channels, grid_size):
        super(Edge_Annotation, self).__init__()

        self.cnn_feature_grids = grid_size
        self.edge_annotation_concat_channels = 64 * 4

        edge_annotation_cnn_tunner_1 = nn.Conv2d(in_channels + 2, self.edge_annotation_concat_channels, kernel_size=3,
                                                 padding=1, bias=False)
        edge_annotation_cnn_tunner_bn_1 = nn.BatchNorm2d(self.edge_annotation_concat_channels)
        edge_annotation_cnn_tunner_relu_1 = nn.ReLU(inplace=True)

        edge_annotation_cnn_tunner_2 = nn.Conv2d(self.edge_annotation_concat_channels,
                                                 self.edge_annotation_concat_channels, kernel_size=3,
                                                 padding=1, bias=False)
        edge_annotation_cnn_tunner_bn_2 = nn.BatchNorm2d(self.edge_annotation_concat_channels)
        edge_annotation_cnn_tunner_relu_2 = nn.ReLU(inplace=True)

        self.edge_annotation_concat = nn.Sequential(edge_annotation_cnn_tunner_1,
                                                    edge_annotation_cnn_tunner_bn_1,
                                                    edge_annotation_cnn_tunner_relu_1,
                                                    edge_annotation_cnn_tunner_2,
                                                    edge_annotation_cnn_tunner_bn_2,
                                                    edge_annotation_cnn_tunner_relu_2)

    def edge_annotation_cnn(self, feature):

        final_feature_map = self.edge_annotation_concat(feature)

        return final_feature_map.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids**2, self.edge_annotation_concat_channels)

    def sampling(self, ids, features):

        cnn_out_feature = []
        for i in range(ids.size()[1]):
            id = ids[:, i, :]

            cnn_out = utils.gather_feature(id, features[i])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)


def interpolated_sum(cnns, coords, grids):

    X = coords[:, :, 0]
    Y = coords[:, :, 1]

    cnn_outs = []
    for i in range(len(grids)):
        grid = grids[i]

        Xs = X * grid
        X0 = torch.floor(Xs)
        X1 = X0 + 1

        Ys = Y * grid
        Y0 = torch.floor(Ys)
        Y1 = Y0 + 1

        w_00 = (X1 - Xs) * (Y1 - Ys)
        w_01 = (X1 - Xs) * (Ys - Y0)
        w_10 = (Xs - X0) * (Y1 - Ys)
        w_11 = (Xs - X0) * (Ys - Y0)

        X0 = torch.clamp(X0, 0, grid - 1)
        X1 = torch.clamp(X1, 0, grid - 1)
        Y0 = torch.clamp(Y0, 0, grid - 1)
        Y1 = torch.clamp(Y1, 0, grid - 1)

        N1_id = X0 + Y0 * grid
        N2_id = X0 + Y1 * grid
        N3_id = X1 + Y0 * grid
        N4_id = X1 + Y1 * grid

        M_00 = utils.gather_feature(N1_id, cnns[i])
        M_01 = utils.gather_feature(N2_id, cnns[i])
        M_10 = utils.gather_feature(N3_id, cnns[i])
        M_11 = utils.gather_feature(N4_id, cnns[i])
        cnn_out = w_00.unsqueeze(2) * M_00 + \
                  w_01.unsqueeze(2) * M_01 + \
                  w_10.unsqueeze(2) * M_10 + \
                  w_11.unsqueeze(2) * M_11

        cnn_outs.append(cnn_out)
    concat_features = torch.cat(cnn_outs, dim=2)
    return concat_features
