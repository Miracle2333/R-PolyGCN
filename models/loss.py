import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if target_class_ids.size()[0]:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data, indices[:,1].data, :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        alpha = 2
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0].data, :, :]
        y_pred = pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_fp_edge_loss(gt_edges, edge_logits, target_class_ids):
    """
    Edge loss in the first point network

    gt_edges: [batch_size, grid_size, grid_size] of 0/1
    edge_logits: [batch_size, grid_size*grid_size]
    """

    if target_class_ids.size()[0]:

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = gt_edges[indices[:, 0].data, :, :]
        y_pred = edge_logits[indices[:, 0].data, :]
        alpha = 2.5
        y_true_shape = y_true.size()
        y_true = y_true.view(y_true_shape[0], -1)

        #Focal loss
        # BCE_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        #
        # pt = torch.exp(-BCE_loss)
        #
        # loss = alpha*(1 - pt) ** 2 * BCE_loss

        #normal loss
        loss = alpha*F.binary_cross_entropy_with_logits(y_pred, y_true)

    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if edge_logits.is_cuda:
            loss = loss.cuda()
        return loss

    return torch.mean(loss)


def compute_fp_vertex_loss(gt_verts, vertex_logits, target_class_ids):
    """
    Vertex loss in the first point network

    gt_verts: [batch_size, grid_size, grid_size] of 0/1
    vertex_logits: [batch_size, grid_size**2]
    """

    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = gt_verts[indices[:, 0].data, :, :]
        y_pred = vertex_logits[indices[:, 0].data, :]
        alpha = 2.5
        y_true_shape = y_true.size()
        y_true = y_true.view(y_true_shape[0], -1)

        # Focal loss
        # BCE_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        #
        # pt = torch.exp(-BCE_loss)
        #
        # loss = alpha*(1 - pt) ** 2 * BCE_loss

        # normal loss
        loss = alpha * F.binary_cross_entropy_with_logits(y_pred, y_true)

    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if vertex_logits.is_cuda:
            loss = loss.cuda()
        return loss

    return torch.mean(loss)


def compute_poly_mathcing_loss(pnum, pred, gt, target_class_ids, loss_type="L1"):

    if pred.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = gt[indices[:, 0].data, :, :]
        y_pred = pred[indices[:, 0].data, :, :]
        batch_size = y_true.size()[0]
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).cuda()

        # import ipdb;
        # ipdb.set_trace()
        feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), y_true.size(2)).detach()
        gt_expand = torch.gather(y_true, 1, feature_id).view(batch_size, pnum, pnum, 2)

        pred_expand = y_pred.unsqueeze(1)

        dis = pred_expand - gt_expand

        if loss_type == "L2":
            dis = (dis ** 2).sum(3).sqrt().sum(2)
        elif loss_type == "L1":
            dis = torch.abs(dis).sum(3).sum(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
        min_id = torch.from_numpy(min_id.data.cpu().numpy()).cuda()

        min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long(). \
            expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
        gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

        return gt_right_order, torch.mean(min_dis)

    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        order = 0
        if pred.is_cuda:
            loss = loss.cuda()
        return order, loss


def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                                            edge_logits, vertex_logits, pred_polys, target_polys, target_mask, vertex_target_masks, edge_target_masks, config):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    # mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)
    gt_right_order, poly_mathcing_loss = compute_poly_mathcing_loss(pnum=config.PNUM, pred=pred_polys, gt=target_polys,
                                                                    target_class_ids=target_class_ids, loss_type="L1")
    fp_vertex_loss = compute_fp_vertex_loss(gt_verts=vertex_target_masks, vertex_logits=vertex_logits, target_class_ids=target_class_ids)

    fp_edge_loss = compute_fp_edge_loss(gt_edges=edge_target_masks, edge_logits=edge_logits, target_class_ids=target_class_ids)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, gt_right_order, poly_mathcing_loss, fp_vertex_loss, fp_edge_loss]


def compute_losses_no_edge(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, masks, target_mask,
                                            pred_polys, target_polys, vertex_target_masks, edge_target_masks, vertex_logits, edge_logits, config):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, masks)
    fp_vertex_loss = compute_fp_vertex_loss(gt_verts=vertex_target_masks, vertex_logits=vertex_logits,
                                            target_class_ids=target_class_ids)

    fp_edge_loss = compute_fp_edge_loss(gt_edges=edge_target_masks, edge_logits=edge_logits,
                                        target_class_ids=target_class_ids)
    gt_right_order, poly_mathcing_loss = compute_poly_mathcing_loss(pnum=config.PNUM, pred=pred_polys, gt=target_polys,
                                                                    target_class_ids=target_class_ids, loss_type="L1")

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, gt_right_order, poly_mathcing_loss, fp_vertex_loss, fp_edge_loss]
