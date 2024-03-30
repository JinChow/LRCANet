import math
import os
import sys
from multiprocessing import Pool
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.special import softmax
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils


def merge(eval_path, num_tasks=1, method='prob'):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    # for Ekman-6
    count_true_pred = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

   #for VideoEmotion8
   #count_true_pred = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    print("Reading individual output files")
    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            # chunk_nb = line.split(']')[1].split(' ')[2]
            # split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(
                line.split('[')[1].split(']')[0], dtype=float, sep=',')
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []

            if method == 'prob':
                dict_feats[name].append(softmax(data))
            else:
                dict_feats[name].append(data)

            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(64)
    # [pred, top1, top5, label]
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    number = sum(top1)
    for i in range(len(label)):
        true_label = label[i]
        predicted_label = ans[i][0]
        if true_label == predicted_label:
            count_true_pred[true_label] += 1

    return final_top1 * 100, final_top5 * 100, count_true_pred



def compute_video(lst):

    i, video_id, data, label = lst

    feat = [x for x in data]

    feat = np.mean(feat, axis=0)
    print(feat)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]


# Please confirm the 0.txt is under the following path
final_top1, final_top5, count_true_pred = merge('the/path/to/result', 1)

print(f"Final ACC: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
