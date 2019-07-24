
import torch
import json
import os
import argparse
import numpy as np
import warnings
import skimage
from skimage import io
import skimage.io as sio
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import sys
from collections import defaultdict


from config import Config
from data import gcn_provider
from torch.utils.data import DataLoader
from orig_models.GNN import poly_gnn
from orig_models import losses
from orig_models import metrics
from orig_models import utils
from orig_models.ActiveSpline import ActiveSplineTorch, ActiveBoundaryTorch
import timeit
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ('==> Using Devices %s' % (device))

def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir', type=str, help='the dir to load the data')
    parser.add_argument('--log_dir', type=str, help='the dir to save logs and models')
    parser.add_argument('--model_dir', type=str, help='the initial weight to load')
    parser.add_argument('--outputpath', type=str, help='the out put path for detection results')
    parser.add_argument('--resume', type=str, help='the checkpoint file to resume from', default=None)
    args = parser.parse_args()

    return args


def get_data_loaders(DataProvider, data_dir, config):
    print('Loading Test Data')

    dataset_test = DataProvider(data_dir=data_dir, split='train', mode='train', config=config)

    test_loader = DataLoader(dataset_test, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=4,
                            collate_fn=gcn_provider.collate_fn)

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
    NUM_CLASSES = 1 + 1

    # steps per epoch
    STEPS_PER_EPOCH = 3000
    VALIDATION_STEPS = 50

    BATCH_SIZE = 16

    PNUM = 16
    CP_NUM = 16

    NUM_WORKERS = 4
    BATCH_SIZE = 16

    # DEBUG = True

class Tester(object):

    def __init__(self, args, config):
        self.config = config
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.log_dir = args.log_dir
        self.outputpath = args.outputpath
        self.max_test_step = config.VALIDATION_STEPS
        self.config = config
        self.test_loader = get_data_loaders(DataProvider=gcn_provider.DataProvider,
                                                              data_dir=self.data_dir, config=self.config)
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'logs', 'test'))

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
        self.min_area = 100   #need to re-define
        # self.spline = ActiveSplineTorch(self.cp_num, self.p_num, device=device,
        #                                 alpha=0.5)
        self.spline = ActiveBoundaryTorch(self.cp_num, self.p_num, device=device,
                                          alpha=0.5)

        self.model = poly_gnn.PolyGNN(state_dim=self.state_dim,
                                      n_adj=self.n_adj,
                                      cnn_feature_grids=self.cnn_feature_grids,
                                      coarse_to_fine_steps=self.coarse_to_fine_steps,
                                      get_point_annotation=False,
                                      ).to(device)

        print ('==> Reloading Models')
        self.model.reload(args.resume, strict=False)

    def process_outputs(self, data, output, save=True):
        """
        Process outputs to get final outputs for the whole image
        Optionally saves the outputs to a folder for evaluation
        """

        pred_spline = output['pred_polys']

        # preds = self.spline.sample_point(pred_spline)
        preds = pred_spline
        torch.cuda.synchronize()
        preds = preds.cpu().numpy()

        pred_spline = pred_spline.cpu()
        pred_spline = pred_spline.numpy()

        instances = data['instance']
        polys = []
        results = []
        for i, instance in enumerate(instances):
            detection = defaultdict(float)
            poly = preds[i]
            poly = poly * data['patch_w'][i]
            poly[:, 0] += data['starting_point'][i][0]
            poly[:, 1] += data['starting_point'][i][1]
            detection['image_id'] = instance['image_path']
            img_h, img_w = instance['height'], instance['width']

            detection['poly'] = poly
            detection['image_size'] = [img_w, img_h]
            # pred_sp = pred_spline[i]
            # pred_sp = pred_sp * data['patch_w'][i]
            # pred_sp[:, 0] += data['starting_point'][i][0]
            # pred_sp[:, 1] += data['starting_point'][i][1]
            #
            # instance['spline_pos'] = pred_sp.tolist()

            polys.append(poly)

            results.append(detection)


            # if save:

                # predicted_poly = []



                # pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                # utils.draw_poly(pred_mask, poly.astype(np.int))
                # predicted_poly.append(poly.tolist())
                #
                # # gt_mask = utils.get_full_mask_from_instance(
                # #     self.min_area,
                # #     instance)
                #
                # instance['my_predicted_poly'] = predicted_poly
                # # instance_id = instance['instance_id']
                # image_id = instance['image_id']
                #
                # pred_mask_fname = os.path.join(self.output_dir, '{}_pred.png'.format(image_id))
                # instance['pred_mask_fname'] = os.path.relpath(pred_mask_fname, self.output_dir)
                #
                # # gt_mask_fname = os.path.join(self.output_dir, '{}_gt.png'.format(instance_id))
                # # instance['gt_mask_fname'] = os.path.relpath(gt_mask_fname, self.output_dir)
                #
                # instance['n_corrections'] = 0
                #
                # info_fname = os.path.join(self.output_dir, '{}_info.json'.format(image_id))
                #
                # with warnings.catch_warnings():
                #     warnings.simplefilter("ignore")
                #     sio.imsave(pred_mask_fname, pred_mask)
                #     # sio.imsave(gt_mask_fname, gt_mask)
                #
                # # print '==> dumping json'
                # with open(info_fname, 'w') as f:
                #     json.dump(instance, f, indent=2)

        return results, polys

    def test(self):
        print ('Starting testing')
        self.model.eval()

        # Leave train mode
        times = []
        count = 0
        with torch.no_grad():

            detection_img = np.zeros((650, 650, 3), dtype=np.uint8)
            results = []
            polys = []
            #tqdm: for progress bar visualization
            a = self.test_loader
            a = tqdm(a)
            a = enumerate(a)
            for step, data in a:
                # Forward pass

                # if step == self.max_test_step:
                #     break

                # if self.opts['get_point_annotation']:
                #     img = data['img'].to(device)
                #     annotation = data['annotation_prior'].to(device).unsqueeze(1)
                #     img = torch.cat([img, annotation], 1)
                # else:
                img = data['img'].to(device)

                start = timeit.default_timer()
                output = self.model(img,
                                    data['fwd_poly'])
                stop = timeit.default_timer()
                if count > 0:
                    times.append(stop - start)

                # if self.opts['coarse_to_fine_steps'] > 0:

                # output['pred_polys'] = output['pred_polys'][-1]
                output['pred_polys'] = output['pred_polys']

                # Bring everything to cpu/numpy
                for k in output.keys():
                    if k == 'pred_polys': continue
                    if k == 'edge_logits': continue
                    if k == 'vertex_logits': continue
                    output[k] = output[k].cpu().numpy()
                preds = output['pred_polys']
                preds_1 = preds[0].cpu().numpy()
                preds_2 = preds[1].cpu().numpy()
                preds_3 = preds[2].cpu().numpy()

                for i in range(preds_1.shape[0]):
                    detection = defaultdict(float)
                    instance = data['instance'][i]

                    poly = preds_3[i]
                    # poly = data['orig_poly'][i]

                    poly = poly * data['patch_w'][i]
                    poly[:, 0] += data['starting_point'][i][0]
                    poly[:, 1] += data['starting_point'][i][1]
                    detection['image_id'] = instance['image_path']
                    img_h, img_w = instance['height'], instance['width']

                    detection['poly'] = poly
                    detection['image_size'] = [img_w, img_h]
                    results.append(detection)

                del (output)
                if count > 0:
                    print(sum(times) / float(len(times)))
                count = count + 1

            image_list = data['image_list'][0]
            image_names = data['image_names'][0]
            self.visualize(results, image_list, image_names, self.outputpath)

    def visualize(self, results, image_list, image_names, outputpath):

        for k in range(len(image_list)):

            image_id = image_list[k]
            image_name = image_names[k]
            image = skimage.io.imread(image_id)
            image1 = image.copy()

            # out_path = os.path.join(outputpath, image_name + '_mask.tif')
            out_path1 = os.path.join(outputpath, image_name + '_poly.tif')

            flag = 0
            colors = utils.random_colors(300)
            k = 0
            for i, r in enumerate(results):
                mask = np.zeros([image.shape[0], image.shape[1]],
                      dtype=np.uint8)
                img_id = r['image_id']
                poly = r['poly']
                w, h = r['image_size']

                if img_id == image_id:
                    image1 = utils.draw_poly_line(image1, poly)
                    # image = utils.draw_mask(mask, poly, image, color=colors[k])
                    flag = 1
                    k = k +1
            if flag == 1:
                # io.imsave(out_path, image.astype(np.uint8))
                io.imsave(out_path1, image1.astype(np.uint8))
                k = 0

if __name__ == '__main__':
    print('==> Parsing Args')
    args = get_args()
    print('==>load configuration')
    configs = BuildingConfig()
    print('Init Tester')
    tester = Tester(args, configs)
    print('==> Start Loop over trainer')
    tester.test()

