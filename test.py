# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable

from data.config import cfg
from models.mix_net import MixNet
from models.pyramidbox import build_detector, build_enhancer
from utils.augmentations import to_chw_bgr
from utils.get_mAP import get_mAP

parser = argparse.ArgumentParser(description='pyramidbox test')
parser.add_argument('--model',
                    type=str,
                    default='weights/REGDet_pyramidbox.pth',
                    help='Dir to trained model')
parser.add_argument('--attention_type',
                    default='eca',  # none, eca
                    help='attention type')
parser.add_argument('--thresh',
                    default=0.01, type=float,
                    help='Final confidence threshold')
parser.add_argument('--topk',
                    default=5000, type=int,
                    help='Proposal numbers')
parser.add_argument('--pred_data',
                    default='./output_data/REGDet_pyramidbox.json',
                    type=str,
                    help='Dir to output prediction data')
parser.add_argument('--data_root',
                    default=None,
                    help='data root')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

data_root = args.data_root or cfg.FACE.DSET_DIR

cfg.TOP_K = args.topk
cfg.CONF_THRESH = args.thresh


def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        2000 * 2000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]
    # x = x * cfg.scale

    # scaling
    if cfg.rescale:
        x = 1 / 255. * (x + 127.5)

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    detect_results = []

    for i in range(detections.size(1)):
        j = 0
        while j < detections.size(2) and detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            detect_results.append([pt[0], pt[1], pt[2], pt[3], float(score)])
            j += 1

    detect_results = np.array(detect_results)

    if len(detect_results.shape) == 1:
        return np.array([])

    order = detect_results[:, 4].ravel().argsort()[::-1]
    det = detect_results[order, :]

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    return det


if __name__ == '__main__':
    """ Build network
    """
    pyramidbox_net = build_detector('test', cfg)
    brighten_net = build_enhancer(cfg.n_blocks, cfg.stage_num, args.attention_type)
    net = MixNet(pyramidbox_net, brighten_net, 'test')

    """ Load model
    """
    mdata = torch.load(args.model)
    if 'model' not in mdata:
        model_state_dict = mdata
    else:
        model_state_dict = mdata['model']
    out_state_dict = {}
    for k in model_state_dict.keys():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        out_state_dict[new_k] = model_state_dict[k]
    net.load_state_dict(out_state_dict)
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_names = [line.strip().split()[0] for line in
                 open('splits/test.txt', 'r').readlines()]

    pred_data = {}

    with torch.no_grad():
        for idx, img_name in enumerate(img_names):
            print('Processing {}-th image: {}'.format(idx, img_name))

            img_id = os.path.basename(img_name).split('.')[0]

            det = detect(net, os.path.join(data_root, 'images', img_name), args.thresh)

            # avoid too many boxes
            det = det[:750]

            # save detection boxes
            pred_data[img_id] = []
            for i in range(det.shape[0]):
                xmin = det[i][0]
                ymin = det[i][1]
                xmax = det[i][2]
                ymax = det[i][3]
                score = det[i][4]
                bbox = [xmin, ymin, (xmax + 1), (ymax + 1)]
                score = float(score)
                label = 'face'
                pred_data[img_id].append({'box': bbox, 'score': score, 'label': label})

    with open(args.pred_data, 'w') as fid:
        json.dump(pred_data, fid)

    gt_data = json.load(open('splits/gt_test.json'))
    print('Calculating mAP ...')
    mAP = get_mAP(pred_data, gt_data)
    print("mAP = {0:.2f}%".format(mAP * 100))

    print('Done.')
