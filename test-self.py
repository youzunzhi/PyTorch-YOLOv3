from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import collections
import sys
import time
import datetime
import argparse
import tqdm
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Resize, CenterCrop


class NYUv2Loader_origin_color(Dataset):
    """
    return origin and colorized image pair when getitem
    """

    def __init__(
        self,
        root,
        img_size,
    ):
        self.root = root
        self.n_classes = 14
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.gamma = False

        if self.root is not None:
            file_list = recursive_glob(rootdir=self.root + 'test/', suffix="png")
            self.files['origin'] = file_list
            # file_list = recursive_glob(rootdir=self.root + "test_color_0/", suffix="png")
            # self.files['color'] = file_list

    def __len__(self):
        return len(self.files['origin'])

    def __getitem__(self, index):
        origin_img_path = self.files['origin'][index].rstrip()
        color_img_path = origin_img_path.replace('test', 'test_color_0_ori_size')
        # color_img_path = origin_img_path
        origin_img = Image.open(origin_img_path).convert('RGB')
        color_img = Image.open(color_img_path).convert('RGB')
        origin_img = origin_img.resize((self.img_size[0], self.img_size[1]))
        color_img = color_img.resize((self.img_size[0], self.img_size[1]))
        origin_img = transforms.ToTensor()(origin_img)
        color_img = transforms.ToTensor()(color_img)
        # origin_img, _ = pad_to_square(origin_img, 0)
        # color_img, _ = pad_to_square(color_img, 0)
        return origin_img, color_img


def evaluate(model, iou_thres, conf_thres_gt, conf_thres_eval, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    root = '/home/u2263506/data/nyuv2-seg/' if torch.cuda.is_available() else '/Users/youzunzhi/Downloads/nyuv2-seg/'
    dataset = NYUv2Loader_origin_color(root, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (origin_img, color_img) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    # for batch_i, (origin_img, color_img) in enumerate(dataloader):
        if torch.cuda.is_available():
            origin_img = origin_img.cuda()
            color_img = color_img.cuda()

        with torch.no_grad():
            origin_outputs = model(origin_img)
            color_outputs = model(color_img)
            origin_outputs = non_max_suppression(origin_outputs, conf_thres=conf_thres_gt, nms_thres=nms_thres)
            color_outputs = non_max_suppression(color_outputs, conf_thres=conf_thres_eval, nms_thres=nms_thres)
            origin_targets = make_origin_outputs_to_targets(origin_outputs)
            if len(origin_targets):
                labels += origin_targets[:, 1].tolist()

        sample_metrics += get_batch_statistics(color_outputs, origin_targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def make_origin_outputs_to_targets(origin_outputs):
    target = []
    for i in range(len(origin_outputs)):
        if origin_outputs[i] is not None:
            output = np.asarray(origin_outputs[i])
            for box in output:
                # if box[-1] in [0, 56, 57, 59, 60, 61, 62, 71, 72]:
                target.append([i, float(box[-1]), box[0], box[1], box[2], box[3]])
    target = torch.from_numpy(np.asarray(target))
    target = target.type(torch.float32)
    return target




def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres_gt", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--conf_thres_eval", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    experiment_name = f'test-self_{opt.conf_thres_gt}_{opt.conf_thres_eval}_{opt.img_size}'
    logger = setup_logger("YOLOv2", 'runs/', 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        iou_thres=opt.iou_thres,
        conf_thres_gt=opt.conf_thres_gt,
        conf_thres_eval=opt.conf_thres_eval,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    logger.info("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    logger.info(f"mAP: {AP.mean()}")
