
"""Preprocess the Data"""

import glob
import json
import multiprocessing.dummy as multiprocessing
import os
import os.path as osp
import random
import re
import skimage
import colorsys
import numpy as np
import skimage.draw
import skimage.transform as transform
import matplotlib
from skimage.measure import find_contours
import matplotlib.pyplot as plt
if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import cv2
import numpy as np
import skimage.transform as transform
import torch
from torch.utils.data import Dataset

# from utils import Utils as utils
from orig_models import utils as utils
EPS = 1e-7


def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}

    for key in keys:
        val = [item[key] for item in batch_list]

        t = type(batch_list[0][key])

        if t is np.ndarray:

            if (key != "orig_poly") & (key != "image_list") & (key != "image_names"):
                try:
                    val = torch.from_numpy(np.stack(val, axis=0))
                except:
                    # for items that are not the same shape
                    # for eg: orig_poly
                    val = [item[key] for item in batch_list]
            else:
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated


class DataProvider(Dataset):
    """
    class for dataprovider
    """

    def __init__(self, data_dir, split, mode, config):
        """
        split: 'train', 'train_val' or 'val'
        config: hyperparameters for the network
        """
        self.config = config
        self.split = split
        self.mode = mode
        self.data_dir = data_dir

        self.image_ids = []
        self.image_info = []
        self.image_names = []
        self.instance_info = []
        self.polygons = []
        self.class_info = [{"id": 0, "name": "BG"}, {"id": 1, "name": "building"}]

        self.pnum = config.PNUM
        self.cp_num = config.CP_NUM
        self.image_side = 224

        self.read_dataset()
        print('Read %d images in %s split' % (len(self.image_ids), split))
        print('Read %d instances in %s split' % (len(self.instance_info), split))

        self.prepare()


    def add_class(self, class_id, class_name):
        # Add the class
        self.class_info.append({
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def add_instance(self, instance_id, image_path, **kwargs):
        instance_info = {
            "label": "building",
            "instance_id": instance_id,
            "image_path": image_path,
        }
        instance_info.update(kwargs)
        self.instance_info.append(instance_info)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(image_id)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, x, y):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        info = self.image_info
        mask = np.zeros([self.height, self.width],
                        dtype=np.uint8)
        rr, cc =skimage.draw.polygon(y, x)
        mask[rr, cc] = 1

        return mask

    def read_dataset(self):
        assert self.split in ["train", "val", "train1"]
        self.image_ids = []
        self.image_names = []
        if self.split == "test":
            dataset_dir = os.path.join(self.data_dir, self.split)
            test_files = os.listdir(dataset_dir)
            for f in test_files:
                filename = f
                image_path = os.path.join(dataset_dir, filename)
                self.height = height = 650
                self.width = width = 650
                self.add_image(
                    image_id=filename,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height)

        else:
            image_id = ' '
            image_name_temp = ' '
            dataset_dir = os.path.join(self.data_dir, self.split)
            anno_list = glob.glob(osp.join(dataset_dir, '*.json'))
            with open(anno_list[0], 'r') as f:
                annotations = json.load(f)
            self.height = 650
            self.width = 650
            polygons = []

            i = 0
            for a in annotations:
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. There are stores in the
                # shape_attributes (see json format above)
                # polygons = [r['shape_attributes'] for r in a['regions'].values()]

                poly = {}.fromkeys(['x', 'y'])
                poly['building_id'] = i+1
                x = [float(s) for s in re.findall(r'-?\d+\.?\d*', a['X'])]
                y = [float(s) for s in re.findall(r'-?\d+\.?\d*', a['Y'])]
                for k, t in enumerate(x):
                    if (t >= 650):
                        x[k] = 649.5
                for k, t in enumerate(y):
                    if (t >= 650):
                        y[k] = 649.5
                if self.split == 'train':
                    poly['x'] = x
                    poly['y'] = y
                else:
                    poly['x'] = x
                    poly['y'] = y
                if (len(x) == 0 | len(y) == 0):
                    continue
                elif (len(x) > 50):
                    continue
                elif (np.size(x, 0) < 2 | np.size(y, 0) < 2):
                    continue
                elif ((np.abs(np.max(x) - np.min(x)) < 7) | (np.abs(np.max(y) - np.min(y)) < 7)):
                    continue
                else:
                    polygons.append(poly)
                    image_name = a['ImageId']
                    filename = 'RGB-PanSharpen_' + image_name + '.tif'
                    image_path = os.path.join(dataset_dir, filename)
                    # image = skimage.io.imread(image_path)
                    # height, width = image.shape[:2]
                    height = 650
                    width = 650

                    self.add_instance(
                        instance_id=i,  # use file name as a unique image id
                        image_path=image_path,
                        image_name=image_name,
                        width=width, height=height,
                        polys=poly)
                    i = i + 1

                    if image_path == image_id:
                        continue
                    else:
                        image_id = image_path
                        self.image_ids.append(image_id)

                    if image_name == image_name_temp:
                        continue
                    else:
                        image_name_temp = image_name
                        self.image_names.append(image_name_temp)

            self.polygons = polygons


    def prepare(self):
        """Prepares the Dataset class for use.
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        config = self.config
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_ids[:] = 1
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_instances = len(self.instance_info)
        self.instance_ids = np.arange(self.num_instances)

    def __len__(self):
        return len(self.instance_info)

    def __getitem__(self, idx):
        return self.prepare_instance(idx)

    def prepare_instance(self, idx):
        instance = self.instance_info[idx]
        instance_id = self.instance_ids[idx]
        image_id = instance['image_path']
        image = self.load_image(image_id=image_id)
        config = self.config
        image_ids = self.image_ids
        image_names = self.image_names

        if self.split == "test":
            fwd_img, window, scale, padding = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM,
                                                        max_dim=config.IMAGE_MAX_DIM, padding=config.IMAGE_PADDING)
            image_meta = utils.compose_image_meta(image.shape, window)
            fwd_img = fwd_img.astype(np.float32) - self.config.MEAN_PIXEL
            fwd_img = fwd_img.transpose(2, 0, 1)
            pointsnp = utils.get_initial_points(self.cp_num)

            train_dict = {
                'instance_info': instance,
                'image_id': image_id,
                'image_list': image_ids,
                'image_names': image_names,
                'orig_img': image,
                'window': window,
                'image_meta': image_meta,
                'fwd_img': fwd_img,
                'fwd_polys': pointsnp,
                'label': "building"
            }
            return_dict = train_dict
        else:
            get_gt_poly = 'train' in self.mode or 'oracle' in self.mode
            max_num = self.pnum
            pnum = self.pnum
            cp_num = self.cp_num

            # create circle polygon data
            pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
            for i in range(cp_num):
                thera = 1.0 * i / cp_num * 2 * np.pi
                x = np.cos(thera)
                y = -np.sin(thera)
                pointsnp[i, 0] = x
                pointsnp[i, 1] = y

            fwd_poly = (0.7 * pointsnp + 1) / 2

            arr_fwd_poly = np.ones((cp_num, 2), np.float32) * 0.
            arr_fwd_poly[:, :] = fwd_poly

            if self.split == 'train':
                lo, hi = [0.1, 0.2]
            else:
                lo, hi = [0.15, 0.15]

            # get random number from a to b range
            context_expansion = random.uniform(lo, hi)

            crop_info = self.extract_crop(instance, image, context_expansion)

            img = crop_info['img']

            train_dict = {}
            if get_gt_poly:
                poly = crop_info['poly']

                orig_poly = poly.copy()

                gt_orig_poly = poly.copy()
                gt_orig_poly = utils.poly01_to_poly0g(gt_orig_poly, 28)

                # Get masks
                vertex_mask = np.zeros((28, 28), np.float32)
                edge_mask = np.zeros((28, 28), np.float32)
                vertex_mask = utils.get_vertices_mask(gt_orig_poly, vertex_mask)
                edge_mask = utils.get_edge_mask(gt_orig_poly, edge_mask)

                # if self.opts['get_point_annotation']:
                #     gt_poly_224 = np.floor(orig_poly * self.opts['img_side']).astype(np.int32)
                #     if self.opts['ext_points']:
                #         ex_0, ex_1, ex_2, ex_3 = utils.extreme_points(gt_poly_224, pert=self.opts['ext_points_pert'])
                #         nodes = [ex_0, ex_1, ex_2, ex_3]
                #         point_annotation = utils.make_gt(nodes, h=self.opts['img_side'], w=self.opts['img_side'])
                #         target_annotation = np.array([[0, 0]])
                gt_poly = self.uniformsample(poly, pnum)
                sampled_poly = self.uniformsample(poly, 70)
                arr_gt_poly = np.ones((pnum, 2), np.float32) * 0.
                arr_gt_poly[:, :] = gt_poly

                # Numpy doesn't throw an error if the last index is greater than size
                # if self.opts['get_point_annotation']:
                #     train_dict = {
                #         'target_annotation': target_annotation,
                #         'sampled_poly': sampled_poly,
                #         'orig_poly': orig_poly,
                #         'gt_poly': arr_gt_poly,
                #         'annotation_prior': point_annotation
                #
                #     }

                train_dict = {
                    'sampled_poly': sampled_poly,
                    'orig_poly': orig_poly,
                    'gt_poly': arr_gt_poly,
                }

                boundry_dic = {
                    'vertex_mask': vertex_mask,
                    'edge_mask': edge_mask
                }
                train_dict.update(boundry_dic)
                if 'train' in self.mode:
                    train_dict['label'] = instance['label']

            # for Torch, use CHW, instead of HWC
            img = img.transpose(2, 0, 1)
            # blank_image
            return_dict = {
                "instance": instance,
                'img': img,
                'image_list': image_ids,
                'image_names': image_names,
                'fwd_poly': arr_fwd_poly,
                'img_path': instance['image_path'],
                'patch_w': crop_info['patch_w'],
                'starting_point': crop_info['starting_point'],
                'context_expansion': context_expansion
            }

            return_dict.update(train_dict)

        return return_dict


    # def prepare_polygons(self, instances, gt_image):
    #
    #     full_masks = np.zeros([self.height, self.width, len(instances)],
    #                     dtype=np.uint8)
    #     vertex_masks = np.zeros((self.height, self.width, len(instances)), np.float32)
    #     edge_masks = np.zeros((self.height, self.width, len(instances)), np.float32)
    #     poly_id = np.zeros(len(instances), np.int)
    #     pnum = self.pnum
    #     cp_num = self.cp_num
    #     gt_polys = np.zeros((pnum, 2, len(instances)), np.float32)
    #     pointsnp = np.zeros((cp_num, 2, len(instances)), np.float32)
    #     for i, poly in enumerate(instances):
    #         poly_id[i] = poly['building_id']
    #         x = np.asarray(poly['x'])
    #         y = np.asarray(poly['y'])
    #         poly_points = np.zeros((len(x), 2), dtype=float)
    #         poly_points[:, 0] = y
    #         poly_points[:, 1] = x
    #         orig_poly_points = (np.floor(poly_points.copy())).astype(np.int32)
    #         # get region mask
    #         mask = self.load_mask(x, y)
    #         full_masks[:, :, i] = mask
    #         # get vertex and boundary edge mask
    #         edge_mask = np.zeros((self.height, self.width), np.float32)
    #         vertex_masks[:, :, i] = utils.get_vertices_mask(orig_poly_points, vertex_masks[:, :, i])
    #         edge_masks[:, :, i] = utils.get_edge_mask(orig_poly_points, edge_mask.copy())
    #         # resample the points to the same N number
    #         gt_poly = utils.uniformsample(poly_points, newpnum=pnum)
    #         gt_polys[:, :, i] = gt_poly
    #         # get initial cvircle points for gragh (normalized)
    #         pointsnp[:, :, i] = utils.get_initial_points(cp_num=cp_num)
    #
    #
    #     # normalize all the inputs
    #     image, image_meta, masks, vertex_masks, edge_masks, bboxes, nor_polys = utils.resize(image=gt_image,
    #                                                                             orig_mask=full_masks, gt_polys=gt_polys,
    #                                             vertex_mask=vertex_masks, edge_mask=edge_masks,
    #                                             config=self.config, use_mini_mask=True)
    #
    #     # get RPN boxes from anchors
    #     """could be improved"""
    #     rpn_match, rpn_boxes = build_rpn_targets(self.anchors, bboxes, self.config)
    #
    #     class_ids = poly_id.copy()
    #     class_ids = class_ids.astype(int)
    #     class_ids[:] = 1
    #
    #     # If more instances than fits in the array, sub-sample from them.
    #     if bboxes.shape[0] > self.config.MAX_GT_INSTANCES:
    #         ids = np.random.choice(
    #             np.arange(bboxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
    #         class_ids = class_ids[ids]
    #         bboxes = bboxes[ids]
    #         masks = masks[:, :, ids]
    #         vertex_masks = vertex_masks[:, :, ids]
    #         edge_masks = edge_masks[:, :, ids]
    #         gt_polys = gt_polys[:, :, ids]
    #         nor_polys = nor_polys[:, :, ids]
    #         pointsnp = pointsnp[:, :, ids]
    #
    #     #inspect the ground-truth data
    #     if self.config.DEBUG:
    #
    #         figsize = (16, 16)
    #         _, ax1 = plt.subplots(1, figsize=figsize)
    #         auto_show = True
    #         height, width = image.shape[:2]
    #         height, width = (30, 30)
    #         ax1.set_ylim(height + 10, -10)
    #         ax1.set_xlim(-10, width + 10)
    #         ax1.axis('off')
    #         title = 'gt'
    #         ax1.set_title(title)
    #         masked_image = image.astype(np.uint8).copy()
    #         N = masks.shape[2]
    #         colors = utils.random_colors(N)
    #         for i in range(0, masks.shape[2]):
    #             # y1, x1, y2, x2 = bboxes[i]
    #             # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
    #             #                       alpha=0.7, linestyle="dashed",
    #             #                     facecolor='none')
    #             # ax.add_patch(p)
    #             color = colors[i]
    #             color1 = colors[np.int((i/2))]
    #             poly_ini = pointsnp[:, :, i]
    #             poly_gt = nor_polys[:, :, i]
    #             poly_ini = poly_ini * 28
    #             poly_gt = poly_gt*28
    #             gt_poly_test = gt_polys[:, :, i]
    #             # poly = poly.astype(np.int)
    #
    #             pred_mask = np.zeros((height, width, 3), dtype=np.uint8)
    #
    #
    #             # masked_image = utils.draw_poly(pred_mask, gt_poly_test, masked_image, color=0)
    #
    #             # masked_image = utils.draw_box(masked_image, bboxes[i], color=0)
    #
    #             pred_mask = utils.draw_poly(pred_mask, poly_ini, pred_mask, color=0)
    #
    #             pred_mask = utils.draw_poly(pred_mask, poly_gt, pred_mask, color=0)
    #
    #             ax1.imshow(pred_mask.astype(np.uint8))
    #
    #             if auto_show:
    #                 plt.show(1)
    #
    #
    #
    #
    #     # convert
    #     image = image.astype(np.float32) - self.config.MEAN_PIXEL
    #     image = image.transpose(2, 0, 1)
    #     masks = masks.astype(int).transpose(2, 0, 1)
    #     vertex_masks = vertex_masks.astype(int).transpose(2, 0, 1)
    #     edge_masks = edge_masks.astype(int).transpose(2, 0, 1)
    #     gt_polys = gt_polys.transpose(2, 0, 1)
    #     nor_polys = nor_polys.transpose(2, 0, 1)
    #     pointsnp = pointsnp.transpose(2, 0, 1)
    #     rpn_match = rpn_match[:, np.newaxis]
    #
    #     instance_info = {
    #         'fwd_img': image,
    #         'image_meta': image_meta,
    #         'nor_polys': nor_polys,
    #         'gt_polys': gt_polys,
    #         'fwd_polys': pointsnp,
    #         'poly_ids': poly_id,
    #         'class_ids': class_ids,
    #         'masks': masks,
    #         'vertex_masks': vertex_masks,
    #         'edge_masks': edge_masks,
    #         'bboxes': bboxes,
    #         'rpn_match': rpn_match,
    #         'rpn_bbox': rpn_boxes,
    #         'label': "building"
    #     }
    #
    #     return instance_info

    def uniformsample(self, pgtnp_px2, newpnum):

        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i:i + 1]
                pe_1x2 = pgtnext_px2[i:i + 1]

                pnewnum = edgenum[i]
                wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    def extract_crop(self, instance, img, context_expansion):
        img = img.astype(np.float32) / 255.0

        get_poly = True

        poly = instance['polys']

        xs_wise = np.asarray(poly['x'])
        ys_wise = np.asarray(poly['y'])

        if self.split == 'val':
            xs = xs_wise[::-1]
            ys = ys_wise[::-1]
        else:
            xs = xs_wise
            ys = ys_wise

        bbox = utils.crop_single_box_gcn(xs, ys)
        # bbox = instance['bbox']
        x0, y0, w, h = bbox

        x_center = x0 + (1 + w) / 2.
        y_center = y0 + (1 + h) / 2.

        widescreen = True if w > h else False

        if not widescreen:
            img = img.transpose((1, 0, 2))
            x_center, y_center, w, h = y_center, x_center, h, w
            if get_poly:
                xs, ys = ys, xs

        x_min = int(np.floor(x_center - w * (1 + context_expansion) / 2.))
        x_max = int(np.ceil(x_center + w * (1 + context_expansion) / 2.))

        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min
        # NOTE: Different from before

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(self.image_side) / patch_w

        patch_img = img[y_min:y_max, x_min:x_max, :]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        new_img = transform.rescale(new_img, scale_factor, order=1,
                                    preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)
        # assert new_img.shape == [self.opts['img_side'], self.opts['img_side'], 3]

        starting_point = [x_min, y_min - top_margin]

        if get_poly:
            xs = (xs - x_min) / float(patch_w)
            ys = (ys - (y_min - top_margin)) / float(patch_w)

            xs = np.clip(xs, 0 + EPS, 1 - EPS)
            ys = np.clip(ys, 0 + EPS, 1 - EPS)

        if not widescreen:
            # Now that everything is in a square
            # bring things back to original mode
            new_img = new_img.transpose((1, 0, 2))
            starting_point = [y_min - top_margin, x_min]
            if get_poly:
                xs, ys = ys, xs

        return_dict = {
            'img': new_img,
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen
        }

        if get_poly:
            poly = np.array([xs, ys]).T
            return_dict['poly'] = poly

        return return_dict

