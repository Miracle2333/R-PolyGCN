from skimage.io import imread
import skimage.color as color
from skimage.draw import polygon
import cv2
import os
import numpy as np
import scipy
import random
import colorsys
import skimage
from scipy.ndimage.morphology import distance_transform_cdt
import copy
from PIL import Image, ImageDraw

EPS = 1e-7


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


# def box_refinement(box, gt_box):
#     """Compute refinement needed to transform box to gt_box.
#     box and gt_box are [N, (y1, x1, y2, x2)]
#     """
#
#     height = box[:, 2] - box[:, 0]
#     width = box[:, 3] - box[:, 1]
#     center_y = box[:, 0] + 0.5 * height
#     center_x = box[:, 1] + 0.5 * width
#
#     gt_height = gt_box[:, 2] - gt_box[:, 0]
#     gt_width = gt_box[:, 3] - gt_box[:, 1]
#     gt_center_y = gt_box[:, 0] + 0.5 * gt_height
#     gt_center_x = gt_box[:, 1] + 0.5 * gt_width
#
#     dy = (gt_center_y - center_y) / height
#     dx = (gt_center_x - center_x) / width
#     dh = torch.log(gt_height / height)
#     dw = torch.log(gt_width / width)
#
#     result = torch.stack([dy, dx, dh, dw], dim=1)
#     return result


def create_folder(path):
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s'%(path))
        print('Experiment folder created at: %s'%(path))


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return nodes[np.argmin(dist_2)]


def closest_node_index(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates

    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]

    return poly


def poly0g_to_poly01(polygon, grid_side):
    """
    [0, grid_side] coordinates to [0, 1].
    Note: we add 0.5 to the vertices so that the points
    lie in the middle of the cell.
    """
    result = (polygon.astype(np.float32) + 0.5)/grid_side

    return result


def get_mini_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    poly_temp = poly.copy()
    # temp = poly[:, 0].copy()
    # poly_temp[:, 0] = poly[:, 1].copy()
    # poly_temp[:, 1] = temp
    poly_temp = poly_temp.astype(np.int32)

    mask[poly_temp[:, 1], poly_temp[:, 0]] = 1.

    return mask


def get_mini_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    poly_temp = poly.copy()
    # temp = poly[:, 0].copy()
    # poly_temp[:, 0] = poly[:, 1].copy()
    # poly_temp[:, 1] = temp
    poly_temp = poly_temp.astype(np.int32)

    cv2.polylines(mask, [poly_temp], True, [1])

    return mask


def get_mini_full_mask(poly, mask):
    """
    Generate edge mask
    """
    poly_temp = poly.copy()
    # temp = poly[:, 0].copy()
    # poly_temp[:, 0] = poly[:, 1].copy()
    # poly_temp[:, 1] = temp
    poly_temp = poly_temp.astype(np.int32)

    cv2.fillPoly(mask, [poly_temp], [1])

    return mask


def get_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    mask[poly[:, 0], poly[:, 1]] = 1.

    return mask


def get_edge_mask1(x, y, mask):
    """
    Generate edge mask
    """
    rr, cc = polygon(y, x)
    mask[rr, cc] = 1
    return mask


def get_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    poly_temp = poly.copy()
    # temp = poly[:, 0].copy()
    # poly_temp[:, 0] = poly_temp[:, 1].copy()
    # poly_temp[:, 1] = temp
    poly_temp = poly_temp.astype(np.int32)
    poly_temp = poly_temp.reshape(-1, 1, 2)
    cv2.polylines(mask, [poly_temp], True, [1])
    return mask


def load_mask(x, y):
    """Load instance masks for the given image.

    Different datasets use different ways to store masks. Override this
    method to load instance masks and return them in the form of am
    array of binary masks of shape [height, width, instances].

    Returns:
        masks: A bool array of shape [height, width, instance count] with
            a binary mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
    """
    # mask = np.empty([0, 0, 0])
    # class_ids = np.empty([0], np.int32)
    # info = self.image_info
    mask = np.zeros([1024, 1024],
                    dtype=np.uint8)
    rr, cc =skimage.draw.polygon(y, x)
    mask[rr, cc] = 1

    return mask


def resize(image, orig_mask, gt_polys, vertex_mask, edge_mask, config, use_mini_mask):
    # Load image and mask
    shape = image.shape
    image, window, scale, padding = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    full_mask = resize_mask(orig_mask, scale, padding)
    vertex_mask = resize_mask(vertex_mask, scale, padding)
    edge_mask = resize_mask(edge_mask, scale, padding)
    gt_polys = resize_points(gt_polys, scale, padding)
    bboxes = extract_box(polys=gt_polys)

    # Random horizontal flips.
    # if augment:
    #     if random.randint(0, 1):
    #         image = np.fliplr(image)
    #         mask = np.fliplr(mask)


    # # Active classes
    # # Different datasets have different classes, so track the
    # # classes supported in the dataset of this image.
    # active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    # active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        # for i in range(mask.shape[-1]):
        #     m = mask[:, :, i]
        #     y1, x1, y2, x2 = bbox[i][:4]
        #     m = m[y1:y2, x1:x2]
        #     if m.size == 0:
        #         a=1
        """can be improvred here"""
        full_mask = minimize_mask(bboxes, full_mask, config.MINI_MASK_SHAPE)
        vertex_mask = minimize_mask(bboxes, vertex_mask, config.MINI_MASK_SHAPE)
        edge_mask = minimize_mask(bboxes, edge_mask, config.MINI_MASK_SHAPE)
        gt_polys = minimize_points(bboxes, gt_polys, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(shape, window)

    return image, image_meta, full_mask, vertex_mask, edge_mask, bboxes, gt_polys


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def resize_points(gt_polys, scale, padding):
    for i in range(0, np.size(gt_polys, axis=2)):
        gt_polys[:, :, i] = gt_polys[:, :, i]*scale + padding[0][0]
    return gt_polys


def resize_point(gt_poly, scale, padding):
    gt_poly[:, :] = gt_poly[:, :]*scale + padding[0][0]
    return gt_poly


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        # mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
        m = skimage.transform.resize(m, mini_shape, order=1, mode="reflect")
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def minimize_single_mask(bbox, mask, mini_shape):
    mini_mask = np.zeros(mini_shape, dtype=np.float32)
    m = mask[:, :]
    y1, x1, y2, x2 = bbox[:4]
    m = m[y1:y2, x1:x2]
    if m.size == 0:
        raise Exception("Invalid bounding box with area of zero")
    # m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
    # mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    m = skimage.transform.resize(m, mini_shape, order=1, mode="constant")
    # mini_mask[:, :] = np.around(m).astype(np.bool)
    mini_mask = m

    return mini_mask




def minimize_points(bboxes, gt_polys, mini_shape):
    """resize the vertex points coordinates into normalized ones in the minimized bounding box"""
    nor_polys = gt_polys.copy().astype(np.float32)
    for i in range(0, np.size(gt_polys, axis=2)):
        h = bboxes[i, 2] - bboxes[i, 0]
        w = bboxes[i, 3] - bboxes[i, 1]
        nor_polys[:, 0, i] = (gt_polys[:, 0, i] - bboxes[i, 0])/h
        nor_polys[:, 1, i] = (gt_polys[:, 1, i] - bboxes[i, 1])/w

        nor_polys[:, 0, i] = np.clip(nor_polys[:, 0, i], 0 + EPS, 1 - EPS)
        nor_polys[:, 1, i] = np.clip(nor_polys[:, 1, i], 0 + EPS, 1 - EPS)
    return nor_polys


def minimize_poly_point(box, gt_poly):
    nor_poly = gt_poly.copy()

    h = box[2] - box[0]
    w = box[3] - box[1]
    nor_poly[:, 0] = (gt_poly[:, 0] - box[1]) / w
    nor_poly[:, 1] = (gt_poly[:, 1] - box[0]) / h

    nor_poly[:, 0] = np.clip(nor_poly[:, 0], 0 + EPS, 1 - EPS)
    nor_poly[:, 1] = np.clip(nor_poly[:, 1], 0 + EPS, 1 - EPS)
    return nor_poly


def get_initial_points(cp_num):
    pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
    for i in range(cp_num):
        thera = 1.0 * i / cp_num * 2 * np.pi - np.pi/4.0
        if thera < 0:
            thera += 2*np.pi
        if thera > 2*np.pi:
            thera -= 2*np.pi
        x = np.cos(thera)
        y = -np.sin(thera)
        pointsnp[i, 0] = x
        pointsnp[i, 1] = y

    fwd_poly = (0.7 * pointsnp + 1) / 2

    arr_fwd_poly = np.ones((cp_num, 2), np.float32) * 0.
    arr_fwd_poly[:, :] = fwd_poly
    return arr_fwd_poly


def extract_box(polys):
    boxes = np.zeros([polys.shape[-1], 4], dtype=np.int32)
    for i in range(polys.shape[-1]):
        p = polys[:, :, i]
        # Bounding box.
        # horizontal_indicies = np.where(np.any(m, axis=0))[0]
        # vertical_indicies = np.where(np.any(m, axis=1))[0]
        if p.shape[0]:
            y1 = (np.min(p[:, 0])).astype(np.int32)
            y2 = (np.max(p[:, 0])).astype(np.int32)
            x1 = (np.min(p[:, 1])).astype(np.int32)
            x2 = (np.max(p[:, 1])).astype(np.int32)
            h = y2 -y1
            w = x2 - x1
            y1 = y1 - h * 0.1 + 0.5
            y2 = y2 + h * 0.1 + 0.5
            x1 = x1 - w * 0.1 + 0.5
            x2 = x2 + w * 0.1 + 0.5
        else:
            x1, x2, y1, y2 = 0, 1, 0, 1

        # if horizontal_indicies.shape[0]:
        #     x1, x2 = horizontal_indicies[[0, -1]]
        #     y1, y2 = vertical_indicies[[0, -1]]
        #     # x2 and y2 should not be part of the box. Increment by 1.
        #     x2 += 1
        #     y2 += 1
        # else:
        #     # No mask for this instance. Might happen due to
        #     # resizing or cropping. Set bbox to zeros
        #     x1, x2, y1, y2 = 0, 1, 0, 1
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def extract_single_box(poly):
    box = np.zeros(4, dtype=np.int32)

    p = poly[:, :]
    # Bounding box.
    # horizontal_indicies = np.where(np.any(m, axis=0))[0]
    # vertical_indicies = np.where(np.ny(m, axis=1))[0]
    if p.shape[0]:
        x1 = (np.min(p[:, 0])).astype(np.int32)
        x2 = (np.max(p[:, 0])).astype(np.int32)
        y1 = (np.min(p[:, 1])).astype(np.int32)
        y2 = (np.max(p[:, 1])).astype(np.int32)
        h = y2 - y1
        w = x2 - x1
        #to expand the box a littile(by 20%)
        y1 = y1 + 0.5
        y2 = y2 + 0.5
        x1 = x1 + 0.5
        x2 = x2 + 0.5
    else:
        x1, x2, y1, y2 = 0, 1, 0, 1

    # if horizontal_indicies.shape[0]:
    #     x1, x2 = horizontal_indicies[[0, -1]]
    #     y1, y2 = vertical_indicies[[0, -1]]
    #     # x2 and y2 should not be part of the box. Increment by 1.
    #     x2 += 1
    #     y2 += 1
    # else:
    #     # No mask for this instance. Might happen due to
    #     # resizing or cropping. Set bbox to zeros
    #     x1, x2, y1, y2 = 0, 1, 0, 1
    box = np.array([y1, x1, y2, x2])
    return box.astype(np.int32)


def crop_single_box_gcn(poly):
    box = np.zeros(4, dtype=np.int32)

    p = poly.copy()
    # Bounding box.
    # horizontal_indicies = np.where(np.any(m, axis=0))[0]
    # vertical_indicies = np.where(np.ny(m, axis=1))[0]
    if p.shape[0]:
        x1 = (np.min(p[:, 0])).astype(np.int32) - 1
        x2 = (np.max(p[:, 0])).astype(np.int32) + 1
        y1 = (np.min(p[:, 1])).astype(np.int32) - 1
        y2 = (np.max(p[:, 1])).astype(np.int32) + 1
        h = y2 -y1
        w =x2 - x1
        # y1 = y1 - h*0.1
        # y2 = y2 + h*0.1
    else:
        x1, x2, y1, y2 = 0, 1, 0, 1
        h = y2 - y1
        w = x2 - x1
    box = np.array([x1, y1, w, h])
    return box.astype(np.int32)


def extract_single_sbox(poly, context_expansion):
    box = np.zeros(4, dtype=np.int32)

    p = poly.copy()
    # Bounding box.
    # horizontal_indicies = np.where(np.any(m, axis=0))[0]
    # vertical_indicies = np.where(np.ny(m, axis=1))[0]
    if p.shape[0]:
        x1 = (np.min(p[:, 0])).astype(np.int32)
        x2 = (np.max(p[:, 0])).astype(np.int32)
        y1 = (np.min(p[:, 1])).astype(np.int32)
        y2 = (np.max(p[:, 1])).astype(np.int32)
        h = y2 - y1
        w = x2 - x1
        # y1 = y1 - h*0.1
        # y2 = y2 + h*0.1
    else:
        x1, x2, y1, y2 = 0, 1, 0, 1
        h = y2 - y1
        w = x2 - x1
    box = np.array([x1, y1, w, h])
    img_shape = [1024, 1024]

    xs = poly[:, 0]
    ys = poly[:, 1]
    x_center = x1 + (1 + w) / 2.
    y_center = y1 + (1 + h) / 2.

    widescreen = True if w > h else False

    if not widescreen:
        # img = img.transpose((1, 0, 2))
        x_center, y_center, w, h = y_center, x_center, h, w
        xs, ys = ys, xs

    x_min = int(np.floor(x_center - w * (1 + context_expansion) / 2.))
    x_max = int(np.ceil(x_center + w * (1 + context_expansion) / 2.))

    x_min = max(0, x_min)
    x_max = min(img_shape[1] - 1, x_max)

    patch_w = x_max - x_min
    # NOTE: Different from before

    y_min = int(np.floor(y_center - patch_w / 2.))
    y_max = y_min + patch_w

    top_margin = max(0, y_min) - y_min

    y_min = max(0, y_min)
    y_max = min(img_shape[0] - 1, y_max)

    # scale_factor = float(self.opts['img_side']) / patch_w
    #
    # patch_img = img[y_min:y_max, x_min:x_max, :]

    # new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
    # new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img
    #
    # new_img = transform.rescale(new_img, scale_factor, order=1,
    #                             preserve_range=True, multichannel=True)
    # new_img = new_img.astype(np.float32)
    # assert new_img.shape == [self.opts['img_side'], self.opts['img_side'], 3]

    starting_point = [x_min, y_min - top_margin]

    patch_h = y_max - y_min

    xs = (xs - x_min) / float(patch_w)
    ys = (ys - y_min) / float(patch_h)

    xs = np.clip(xs, 0 + EPS, 1 - EPS)
    ys = np.clip(ys, 0 + EPS, 1 - EPS)

    if not widescreen:
        # Now that everything is in a square
        # bring things back to original mode
        starting_point = [y_min - top_margin, x_min]

        xs, ys = ys, xs
        x_min, y_min = y_min, x_min
        x_max, y_max = y_max, x_max

    patch_box = np.array([y_min, x_min, y_max, x_max])

    return_dict = {
        'patch_w': patch_w,
        'top_margin': top_margin,
        'starting_point': starting_point,
        'widescreen': widescreen
    }

    poly = np.array([xs, ys]).T

    return patch_box.astype(np.int32), poly, return_dict



def compose_image_meta(image_shape, window):
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
        list(image_shape) +     # size=3
        list(window)           # size=4 (y1, x1, y2, x2) in image cooredinates
    )
    return meta


# def gather_feature(id, feature):
#     feature_id = id.unsqueeze_(2).long().expand(id.size(0),
#                                                 id.size(1),
#                                                 feature.size(2)).detach()
#     cnn_out = torch.FloatTensor()
#     try:
#         cnn_out = torch.gather(feature, 1, feature_id).float()
#     except RuntimeError:
#         print(feature_id.shape, feature.shape, id.shape)
#
#     return cnn_out


# def prepare_gcn_component(pred_polys, grid_sizes, max_poly_len, n_adj=3):
#
#     batch_array_feature_indexs = []
#     for i in range(pred_polys.shape[0]):
#
#         curr_p = pred_polys[i]
#         p_index = []
#         for size in grid_sizes:
#             curr_p_grid_size = np.floor(curr_p * size).astype(np.int32)
#
#             curr_p_grid_size[:, 1] *= size
#             curr_p_index = np.sum(curr_p_grid_size, axis=-1)
#             p_index.append(curr_p_index)
#
#         array_feature_indexs = np.zeros((len(grid_sizes), max_poly_len), np.float32)
#
#         array_feature_indexs[:, :max_poly_len] = np.array(p_index)
#
#         batch_array_feature_indexs.append(array_feature_indexs)
#
#     adj_matrix = create_adjacency_matrix_cat(pred_polys.shape[0], n_adj, max_poly_len)
#
#
#     return {
#             'feature_indexs': torch.Tensor(np.stack(batch_array_feature_indexs, axis=0)),
#             'adj_matrix': torch.Tensor(adj_matrix)
#             }


def create_adjacency_matrix_cat(batch_size, n_adj, n_nodes):
    a = np.zeros([batch_size, n_nodes, n_nodes])
    m = int(-n_adj / 2)
    n = int(n_adj / 2 + 1)

    for t in range(batch_size):
        for i in range(n_nodes):
            for j in range(m, n):
                if j != 0:
                    a[t][i][(i + j) % n_nodes] = 1
                    a[t][(i + j) % n_nodes][i] = 1

    return a.astype(np.float32)


# def sample_point(cps, cp_num, p_num):
#     EPS = 1e-7
#     alpha = 0.5
#     p_num = int(p_num/cp_num)
#     # cp_num = 30
#
#     # Suppose cps is [n_batch, n_cp, 2]
#     cps = torch.cat([cps, cps[:, 0, :].unsqueeze(1)], dim=1)
#     auxillary_cps = torch.zeros(cps.size(0), cps.size(1) + 2, cps.size(2)).to(device)
#     auxillary_cps[:, 1:-1, :] = cps
#
#     l_01 = torch.sqrt(torch.sum(torch.pow(cps[:, 0, :] - cps[:, 1, :], 2), dim=1) + EPS)
#     l_last_01 = torch.sqrt(torch.sum(torch.pow(cps[:, -1, :] - cps[:, -2, :], 2), dim=1) + EPS)
#
#     l_01.detach_().unsqueeze_(1)
#     l_last_01.detach_().unsqueeze_(1)
#
#     # print(l_last_01, l_01)
#
#     auxillary_cps[:, 0, :] = cps[:, 0, :] - l_01 / l_last_01 * (cps[:, -1, :] - cps[:, -2, :])
#     auxillary_cps[:, -1, :] = cps[:, -1, :] + l_last_01 / l_01 * (cps[:, 1, :] - cps[:, 0, :])
#
#     # print(auxillary_cps)
#
#     t = torch.zeros([auxillary_cps.size(0), auxillary_cps.size(1)]).to(device)
#
#     for i in range(1, t.size(1)):
#         t[:, i] = torch.pow(torch.sqrt(torch.sum(torch.pow(auxillary_cps[:, i, :] - auxillary_cps[:, i-1, :], 2),
#                                                 dim=1)), alpha) + t[:, i-1]
#
#     # No need to calculate gradient w.r.t t.
#     t = t.detach()
#     # print(t)
#     lp = 0
#     points = torch.zeros([cps.size(0), p_num * cp_num, cps.size(2)]).to(device)
#     # print(self.device)
#     # print(auxillary_cps.type())
#     # print(t.type())
#
#     for sg in range(1, cp_num+1):
#         v = batch_linspace(t[:, sg], t[:, sg+1], p_num)
#         # print(v.type())
#         # print(v.size())
#         # print(v)
#         t0 = t[:, sg-1].unsqueeze(1)
#         t1 = t[:, sg].unsqueeze(1)
#         t2 = t[:, sg+1].unsqueeze(1)
#         t3 = t[:, sg+2].unsqueeze(1)
#
#         for i in range(p_num):
#
#             tv = v[:, i].unsqueeze(1)
#
#             x01 = (t1-tv)/(t1-t0)*auxillary_cps[:, sg-1, :]+(tv-t0)/(t1-t0)*auxillary_cps[:, sg, :]
#
#             x12 = (t2-tv)/(t2-t1)*auxillary_cps[:, sg, :]+(tv-t1)/(t2-t1)*auxillary_cps[:, sg+1,:]
#
#             x23 = (t3-tv)/(t3-t2)*auxillary_cps[:, sg+1, :]+(tv-t2)/(t3-t2)*auxillary_cps[:, sg+2, :]
#
#             x012 = (t2-tv)/(t2-t0)*x01+(tv-t0)/(t2-t0)*x12
#
#             x123 = (t3-tv)/(t3-t1)*x12+(tv-t1)/(t3-t1)*x23
#
#             points[:, lp] = (t2-tv)/(t2-t1)*x012+(tv-t1)/(t2-t1)*x123
#
#             lp = lp + 1
#
#     return points


# def batch_linspace(start_t, end_t, step_t):
#     step_t = [step_t] * end_t.size(0)
#     batch_arr = map(torch.linspace, start_t, end_t, step_t)
#     batch_arr = [arr.unsqueeze(0) for arr in batch_arr]
#     return torch.cat(batch_arr, dim=0).to(device)


def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou


def iou_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred:
        masks[0] = draw_poly12(masks[0], p)

    for g in gt:
        masks[1] = draw_poly12(masks[1], g)

    return iou_from_mask(masks[0], masks[1]), masks


def draw_poly12(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, [poly], 255)

    return mask


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_mask(mask, poly, image, color):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    # cv2.fillPoly(mask.astype(np.int8), [poly], 255)
    rr, cc = polygon(poly[:, 0], poly[:, 1])
    mask[rr, cc] = 255

    alpha = 0.5

    for c in range(3):
        image[:, :, c] = np.where(mask == 255,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])

    return image


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def draw_poly(mask, poly, image, color):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    poly_temp = poly.copy()
    # temp = poly[:, 0].copy()
    # poly_temp[:, 0] = poly[:, 1].copy()
    # poly_temp[:, 1] = temp
    poly_temp = poly_temp.astype(np.int32)
    poly_temp = poly_temp.reshape(-1, 1, 2)
    cv2.polylines(image, [poly_temp], True, (0, 255, 0), 1)

    for i in range(0, poly_temp.shape[0]):

        cv2.rectangle(image, (poly_temp[i, 0, 0], poly_temp[i, 0, 1]), (poly_temp[i, 0, 0]+1, poly_temp[i, 0, 1]+1), (255, 0, 0), 2)
    # cv2.fillPoly(mask.astype(np.int8), [poly], 255)
    # poly = poly.astype(np.uint8)
    # polys = poly.reshape(1, -1)[0]
    # draw = ImageDraw.Draw(image)
    # draw.polygon(polys, fill=None, outline='#f00')

    return image


def draw_poly_line(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    poly_temp = poly.copy()
    # temp = poly[:, 0].copy()
    # poly_temp[:, 0] = poly[:, 1].copy()
    # poly_temp[:, 1] = temp
    poly_temp = poly_temp.astype(np.int32)
    poly_temp = poly_temp.reshape(-1, 1, 2)
    cv2.polylines(mask, [poly_temp], True, (0, 255, 0))

    for i in range(0, poly_temp.shape[0]):

        cv2.rectangle(mask, (poly_temp[i, 0, 0], poly_temp[i, 0, 1]), (poly_temp[i, 0, 0]+1, poly_temp[i, 0, 1]+1), (255, 0, 0), 1)
    # cv2.fillPoly(mask.astype(np.int8), [poly], 255)
    # poly = poly.astype(np.uint8)
    # polys = poly.reshape(1, -1)[0]
    # draw = ImageDraw.Draw(image)
    # draw.polygon(polys, fill=None, outline='#f00')

    return mask


def draw_box(img, box, color):
    y1, x1, y2, x2 = box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img


def uniformsample(pgtnp_px2, newpnum):

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
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp



