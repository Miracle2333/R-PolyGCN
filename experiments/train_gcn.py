"""Poly GCN
   Training Script
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.chdir("/home/kang/PolyGCN/")


import argparse
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import shutil
import numpy as np
import sys
# sys.path.append("/home/kang/PolyGCN/")
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Config

from data import gcn_provider
from torch.utils.data import DataLoader
from orig_models.GNN import poly_gnn
from orig_models import losses
from orig_models import metrics
from orig_models import utils
from orig_models.ActiveSpline import ActiveSplineTorch, ActiveBoundaryTorch

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir', type=str, help='the dir to load the data')
    parser.add_argument('--log_dir', type=str, help='the dir to save logs and models')
    parser.add_argument('--model_dir', type=str, help='the initial weight to load')
    parser.add_argument('--resume', type=str, help='the checkpoint file to resume from', default=None)
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

    train_loader = DataLoader(dataset_train, batch_size=16,
                              shuffle=True, num_workers=8,
                              collate_fn=gcn_provider.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=16,
                            shuffle=True, num_workers=4,
                            collate_fn=gcn_provider.collate_fn)

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
    STEPS_PER_EPOCH = 3000
    VALIDATION_STEPS = 120

    BATCH_SIZE = 16

    PNUM = 16
    CP_NUM = 16

    NUM_WORKERS = 4
    BATCH_SIZE = 16

    # DEBUG = True


class Trainer(object):
    def __init__(self, args, config):
        self.config = config
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.log_dir = args.log_dir
        self.global_step = 0
        self.epoch = 0
        self.max_epochs = 30
        self.val_freq = 500
        self.print_freq = 50
        self.save_freq = 1000
        self.max_train_step = config.STEPS_PER_EPOCH
        self.max_val_step = config.VALIDATION_STEPS
        self.config = config
        self.train_loader, self.val_loader = get_data_loaders(DataProvider=gcn_provider.DataProvider,
                                                              data_dir=self.data_dir, config=self.config)
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'logs', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.log_dir, 'logs', 'train_val'))

        self.pool_size = config.POOL_SIZE
        self.coarse_to_fine_steps = 3  # for GCN steps
        self.n_adj = 4
        self.cnn_feature_grids = [112, 56, 28, 28]
        self.psp_feature = [self.cnn_feature_grids[-1]]
        self.grid_size = 28
        self.state_dim = 128
        self.final_dim = 64 * 4
        self.cp_num = config.CP_NUM
        self.p_num = config.PNUM
        self.fp_weight = 2.5
        self.grad_clip = 40
        # self.spline = ActiveSplineTorch(self.cp_num, self.p_num, device=device,
        #                                 alpha=0.5)
        self.spline = ActiveBoundaryTorch(self.cp_num, self.p_num, device=device,
                                        alpha=0.5)

        self.lr = 3e-5
        self.weight_decay = 1e-5
        self.lr_decay = 7

        self.model = poly_gnn.PolyGNN(state_dim=self.state_dim,
                                      n_adj=self.n_adj,
                                      cnn_feature_grids=self.cnn_feature_grids,
                                      coarse_to_fine_steps=self.coarse_to_fine_steps,
                                      get_point_annotation=False,
                                      )

        if config.GPU_COUNT:
            self.model = self.model.cuda()

        self.model.encoder.reload(self.model_dir)

        # OPTIMIZER
        no_wd = []
        wd = []
        print("Weight Decay applied to:")

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue

            if 'bn' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)
                # print(name),

        # Allow individual options
        self.optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=False)

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay,
                                                  gamma=0.1)

        if args.resume is not None:
            self.resume(args.resume)

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.log_dir, 'epoch%d_step%d.pth' \
                                 % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        self.model.reload(path)
        save_state = torch.load(path)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])

    def loop(self):
        for epoch in range(self.epoch, self.max_epochs):
            if not self.config.DEBUG:
                self.save_checkpoint(epoch)
                self.lr_decay.step()
                print('LR is now: %.7f' % (self.optimizer.param_groups[0]['lr']))
            self.train(epoch)

    def train(self, epoch):
        print('Starting training')

        self.model.train()

        accum = defaultdict(float)
        # To accumulate stats for printing

        for step, data in enumerate(self.train_loader):
            if step == self.max_train_step:
                break
            if len(data['img']) == 1:
                continue

            # if self.opts['get_point_annotation']:
            #     img = data['img'].to(device)
            #     annotation = data['annotation_prior'].to(device).unsqueeze(1)
            #
            #     img = torch.cat([img, annotation], 1)
            img = data['img'].to(device)

            self.optimizer.zero_grad()
            if (self.global_step) % self.val_freq == 0:
                self.validate(epoch)

            if (self.global_step) % self.save_freq == 0:
                self.save_checkpoint(epoch)


            output = self.model(img, data['fwd_poly'])

            loss_sum = 0
            pred_cps = output['pred_polys'][-1]

            # pred_polys = self.spline.sample_point(pred_cps, self.p_num)

            gt = data['gt_poly'].to(device)

            gt_right_order, poly_mathcing_loss_sum = losses.poly_mathcing_loss(self.p_num,
                                                                               pred_cps, gt, loss_type='L1')
            loss_sum += poly_mathcing_loss_sum

            edge_annotation_loss = 0

            curr_fp_edge_loss = self.fp_weight * losses.fp_edge_loss(data['edge_mask'].to(device),
                                                                             output['edge_logits'])
            edge_annotation_loss += curr_fp_edge_loss

            fp_vertex_loss = self.fp_weight * losses.fp_vertex_loss(data['vertex_mask'].to(device),
                                                                            output['vertex_logits'])
            edge_annotation_loss += fp_vertex_loss

            loss_sum += edge_annotation_loss
            loss_sum.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            preds= pred_cps.detach().data.cpu().numpy()
            preds_gcn = pred_cps.detach().data.cpu().numpy()
            with torch.no_grad():
                # Get IoU
                iou = 0
                # orig_poly = data['orig_poly']
                orig_poly = gt.data.cpu().numpy()

                for i in range(preds.shape[0]):
                    curr_pred_poly = np.floor(preds[i] * 224).astype(np.int32)
                    # curr_predgcn_poly = np.floor(preds_gcn[i] * 224).astype(np.int32)
                    curr_gt_poly = np.floor(orig_poly[i] * 224).astype(np.int32)

                    cur_iou, masks = metrics.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                                                           np.array(curr_gt_poly, dtype=np.int32),
                                                           224,
                                                           224)

                    iou += cur_iou

                # shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                gt_shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                smooth_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                gt_shape_mask = utils.draw_poly_line(gt_shape_mask, curr_gt_poly)
                # shape_mask = utils.draw_poly_line(shape_mask, curr_predgcn_poly)
                smooth_mask = utils.draw_poly_line(smooth_mask, curr_pred_poly)

                iou = iou / preds.shape[0]
                accum['loss'] += float(loss_sum.item())
                accum['point_loss'] += float(poly_mathcing_loss_sum.item())
                accum['iou'] += iou
                accum['length'] += 1
                accum['edge_annotation_loss'] += float(edge_annotation_loss.item())
                print(
                    "TRA [%s] Epoch: %d, Step: %d, Complete: %.3f,  Polygon Loss: %.3f,  IOU: %.3f,  Point_loss: %.3f, vertex_loss: %.3f, edge_loss: %.3f" \
                    % (str(datetime.now()), epoch, self.global_step, step/self.max_train_step, accum['loss'] / accum['length'], accum['iou'] / accum['length'], poly_mathcing_loss_sum.data.cpu().item(), fp_vertex_loss.data.cpu().item(), curr_fp_edge_loss.data.cpu().item()))
                if step % self.print_freq == 0:
                    # Mean of accumulated values
                    for k in accum.keys():
                        if k == 'length':
                            continue
                        accum[k] /= accum['length']

                    # Add summaries
                    masks = np.expand_dims(masks, -1).astype(np.uint8)  # Add a channel dimension
                    masks = np.tile(masks, [1, 1, 1, 3])  # Make [2, H, W, 3]
                    # shape_mask = np.expand_dims(shape_mask, -1).astype(np.uint8)
                    # shape_mask = np.tile(shape_mask, [1, 1, 1, 3])
                    img = (data['img'].cpu().numpy()[-1, ...] * 255).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])  # Make [H, W, 3]

                    # self.writer.add_image('pred_mask', masks[0], self.global_step)
                    # self.writer.add_image('gt_mask', masks[1], self.global_step)
                    self.writer.add_image('image', img, self.global_step)
                    # self.writer.add_image('pred_polyline', shape_mask, self.global_step)
                    self.writer.add_image('pre_polyline', smooth_mask, self.global_step)
                    self.writer.add_image('gt_polyline', gt_shape_mask, self.global_step)


                    for k in accum.keys():
                        if k == 'length':
                            continue
                        self.writer.add_scalar(k, accum[k], self.global_step)

                    # print(
                    # "[%s] Epoch: %d, Step: %d, Polygon Loss: %f,  IOU: %f" \
                    # % (str(datetime.now()), epoch, self.global_step, accum['loss'], accum['iou']))

                    accum = defaultdict(float)

            del (output, masks, pred_cps, preds, loss_sum)
            self.global_step += 1

    def validate(self, epoch):
        print('Validating')
        self.model.eval()
        # Leave train mode

        with torch.no_grad():
            ious = []
            accum = defaultdict(float)
            for step, data in enumerate(self.val_loader):

                if step == self.max_val_step:
                    break

                if len(data['orig_poly']) == 1:
                    continue
                # if self.opts['get_point_annotation']:
                #     img = data['img'].to(device)
                #     annotation = data['annotation_prior'].to(device).unsqueeze(1)
                #
                #     img = torch.cat([img, annotation], 1)
                image = data['img'].numpy().copy()

                img = data['img'].to(device)

                output = self.model(img, data['fwd_poly'])

                pred_cps = output['pred_polys'][-1]
                # pred_cps = torch.gather(pred_cps, 2, torch.LongTensor([1, 0]).to(device))
                # pred_polys = self.spline.sample_point(pred_cps, self.p_num)
                gt = data['gt_poly'].to(device)

                loss_sum = 0

                gt_right_order, poly_mathcing_loss_sum = losses.poly_mathcing_loss(self.p_num,
                                                                                   pred_cps,
                                                                                   gt,
                                                                                   loss_type='L1')
                loss_sum += poly_mathcing_loss_sum

                edge_annotation_loss = 0

                curr_fp_edge_loss = self.fp_weight * losses.fp_edge_loss(data['edge_mask'].to(device),
                                                                    output['edge_logits'])
                edge_annotation_loss += curr_fp_edge_loss

                fp_vertex_loss = self.fp_weight * losses.fp_vertex_loss(data['vertex_mask'].to(device),
                                                                   output['vertex_logits'])
                edge_annotation_loss += fp_vertex_loss

                loss_sum += edge_annotation_loss

                pred_polys_val = pred_cps.data.cpu().numpy().copy()
                # print(pred_polys.shape)
                # Get IoU
                iou = 0
                preds_gcn = pred_cps.detach().data.cpu().numpy()
                orig_poly = data['orig_poly']
                gt_polys = gt.data.cpu().numpy()
                orig_poly = gt_polys

                for i in range(pred_cps.shape[0]):
                    # curr_pred_poly = utils.poly01_to_poly0g(pred_polys_val[i], self.model.grid_size)
                    # curr_gt_poly = utils.poly01_to_poly0g(orig_poly[i], self.model.grid_size)
                    # i, masks = metrics.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                    #                                        np.array(curr_gt_poly, dtype=np.int32),
                    #                                        self.model.grid_size,
                    #                                        self.model.grid_size)

                    curr_pred_poly = np.floor(pred_polys_val[i] * 224).astype(np.int32)
                    # curr_predgcn_poly = np.floor(preds_gcn[i] * 224).astype(np.int32)
                    curr_gt_poly = np.floor(orig_poly[i] * 224).astype(np.int32)

                    iou_per, masks = metrics.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                                                           np.array(curr_gt_poly, dtype=np.int32),
                                                           224,
                                                           224)

                    iou += iou_per

                # shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                gt_shape_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                smooth_mask = np.zeros((224, 224, 3), dtype=np.uint8)
                gt_shape_mask = utils.draw_poly_line(gt_shape_mask, curr_gt_poly)
                # shape_mask = utils.draw_poly_line(shape_mask, curr_predgcn_poly)
                smooth_mask = utils.draw_poly_line(smooth_mask, curr_pred_poly)

                iou = iou / pred_cps.shape[0]
                ious.append(iou)

                accum['loss'] += float(loss_sum.item())
                accum['point_loss'] += float(poly_mathcing_loss_sum.item())
                accum['iou'] += iou
                accum['length'] += 1
                accum['edge_annotation_loss'] += float(edge_annotation_loss.item())

                print(
                    "VAL [%s] Epoch: %d, Step: %d, Complete: %.3f, Polygon Loss: %.3f,  IOU: %.3f,  Point_loss: %.3f, vertex_loss: %.3f, edge_loss: %.3f" \
                    % (str(datetime.now()), epoch, self.global_step, step/self.max_val_step, accum['loss'] / accum['length'],
                       accum['iou'] / accum['length'], poly_mathcing_loss_sum.data.cpu().item(),
                       fp_vertex_loss.data.cpu().item(), curr_fp_edge_loss.data.cpu().item()))
                del (output, loss_sum)


            for k in accum.keys():
                if k == 'length':
                    continue
                accum[k] /= accum['length']
                self.val_writer.add_scalar(k, accum[k], self.global_step)

            masks = np.expand_dims(masks, -1).astype(np.uint8)  # Add a channel dimension
            masks = np.tile(masks, [1, 1, 1, 3])  # Make [2, H, W, 3]
            image = (image[-1, ...] * 255).astype(np.uint8)
            image = np.transpose(image, [1, 2, 0])  # Make [H, W, 3]

            # self.val_writer.add_image('pred_mask', masks[0], self.global_step)
            # self.val_writer.add_image('gt_mask', masks[1], self.global_step)
            self.val_writer.add_image('image', image, self.global_step)
            # self.val_writer.add_image('pred_polyline', shape_mask, self.global_step)
            self.val_writer.add_image('pre_polyline', smooth_mask, self.global_step)
            self.val_writer.add_image('gt_polyline', gt_shape_mask, self.global_step)
            iou = np.mean(ious)
            print('[VAL] IoU: %.3f' % iou)
            accum = defaultdict(float)
            del (masks, smooth_mask, pred_cps, gt_shape_mask)

        self.model.train()


if __name__ == '__main__':
    print('==> Parsing Args')
    args = get_args()
    print('==>load configuration')
    configs = BuildingConfig()
    print('Init Trainer')
    trainer = Trainer(args, configs)
    print('==> Start Loop over trainer')
    trainer.loop()
