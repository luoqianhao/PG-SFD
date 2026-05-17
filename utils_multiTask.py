import torch
from numpy.random import normal
import  random
import logging
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from functools import partial
import math
from tqdm import tqdm
from aug_funcs import rot_img, translation_img, hflip_img, grey_img, rot90_img
import torch.backends.cudnn as cudnn
from adeval import  EvalAccumulatorCuda

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import label_binarize

def save_pap_inputs_images(pr_px: np.ndarray,
                           gt_px: np.ndarray,
                           names=None,
                           save_dir: str = "./px_vis",
                           save_limit: int | None = None,
                           save_heatmap: bool = True,
                           make_pair: bool = True):
    """
    将用于计算 P-AP 的 pr_px (N,H,W) 与 gt_px (N,H,W) 保存为图片。
    - pr：保存灰度图 + 可选JET热力图
    - gt：保存二值mask灰度图
    - pair：左右拼接（左pr热力/灰度，右gt）
    """
    os.makedirs(save_dir, exist_ok=True)
    N = pr_px.shape[0]
    if names is None:
        names = [f"{i:06d}" for i in range(N)]
    if save_limit is not None:
        N = min(N, save_limit)

    for i in range(N):
        name = names[i]
        pred = pr_px[i]       # (H,W), float
        gt   = gt_px[i]       # (H,W), {0,1}或float

        # 预测图归一化到 0-255（逐图 min-max）
        pmin, pmax = float(pred.min()), float(pred.max())
        if pmax > pmin:
            pred8 = ((pred - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)
        else:
            pred8 = np.zeros_like(pred, dtype=np.uint8)

        # GT 二值到 0/255
        gt8 = ((gt > 0.5).astype(np.uint8) * 255)

        # 单图保存
        pr_gray_path = os.path.join(save_dir, f"{i:06d}_{name}_pr.png")
        cv2.imwrite(pr_gray_path, pred8)

        if save_heatmap:
            pr_jet = cv2.applyColorMap(pred8, cv2.COLORMAP_JET)
            pr_jet_path = os.path.join(save_dir, f"{i:06d}_{name}_pr_jet.png")
            cv2.imwrite(pr_jet_path, pr_jet)
        else:
            pr_jet = cv2.cvtColor(pred8, cv2.COLOR_GRAY2BGR)

        gt_path = os.path.join(save_dir, f"{i:06d}_{name}_gt.png")
        cv2.imwrite(gt_path, gt8)

        # 拼接对比（左：pr，右：gt）
        if make_pair:
            left  = pr_jet
            right = cv2.cvtColor(gt8, cv2.COLOR_GRAY2BGR)
            pair  = np.concatenate([left, right], axis=1)
            pair_path = os.path.join(save_dir, f"{i:06d}_{name}_pair.png")
            cv2.imwrite(pair_path, pair)

def evaluation_batch_multi_task(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None,
                                save_px_vis_dir: str | None = None,  # 新增：保存目录（可选）
                                save_px_limit: int | None = None):     # 新增：最多保存多少张（可选）):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    cls_preds = []
    cls_labels = []
    cls_probs = []

    img_name_list = []  # 新增：记录每张图的名字，便于保存时命名

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            label = label.to(device)

            # 记录文件名（去后缀）
            names = [os.path.splitext(os.path.basename(p))[0] for p in img_path]
            img_name_list.extend(names)
            
            # 前向传播
            output = model(img)
            
            # 异常检测部分
            en, de = output['anomaly_features']
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            
            pr_list_sp.append(sp_score)
            
            # 分类评估部分
            cls_logits = output['cls_logits']
            cls_similarities = output['cls_similarities']
            
            cls_pred = torch.argmax(cls_logits, dim=1)
            cls_proba = F.softmax(cls_logits, dim=1)
            
            cls_preds.extend(cls_pred.cpu().numpy())
            cls_labels.extend(label.cpu().numpy())
            cls_probs.extend(cls_proba.cpu().numpy())
        
        # 异常检测指标计算
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        
         # ====== 在这里落盘用于 P-AP 的输入对 ======
        if save_px_vis_dir is not None:
            try:
                save_pap_inputs_images(
                    pr_px=pr_list_px,
                    gt_px=gt_list_px,
                    names=img_name_list,
                    save_dir=save_px_vis_dir,
                    save_limit=save_px_limit,
                    save_heatmap=True,
                    make_pair=True
                )
                print(f"[P-AP输入图] 已保存到：{save_px_vis_dir}")
            except Exception as e:
                print(f"[警告] 保存 P-AP 可视化失败：{e}")
        
        # 异常检测评估
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)
        
        # 分类指标计算
        cls_accuracy = accuracy_score(cls_labels, cls_preds)
        cls_precision = precision_score(cls_labels, cls_preds, average='weighted', zero_division=0)
        cls_recall = recall_score(cls_labels, cls_preds, average='weighted', zero_division=0)
        cls_f1 = f1_score(cls_labels, cls_preds, average='weighted', zero_division=0)
        
        # 多分类的AUC
        if len(np.unique(cls_labels)) > 2:
            try:
                # 将标签转换为one-hot编码
                unique_classes = np.unique(cls_labels)
                cls_auc = roc_auc_score(
                    label_binarize(cls_labels, classes=unique_classes),
                    cls_probs,
                    multi_class='ovr'
                )
            except:
                cls_auc = 0.0
        else:
            try:
                cls_auc = roc_auc_score(cls_labels, [prob[1] for prob in cls_probs])
            except:
                cls_auc = 0.0
        
        # 分类报告和混淆矩阵
        cls_report = classification_report(cls_labels, cls_preds, output_dict=True, zero_division=0)
        confusion_mat = confusion_matrix(cls_labels, cls_preds)

    return {
        'anomaly': {
            'auroc_sp': auroc_sp,
            'ap_sp': ap_sp, 
            'f1_sp': f1_sp,
            'auroc_px': auroc_px,
            'ap_px': ap_px,
            'f1_px': f1_px,
            'aupro_px': aupro_px
        },
        'classification': {
            'accuracy': cls_accuracy,
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1,
            'auc': cls_auc,
            'confusion_matrix': confusion_mat,
            'detailed_report': cls_report
        },
        'raw_data': {
            'cls_preds': cls_preds,
            'cls_labels': cls_labels,
            'cls_probs': cls_probs
        }
    }


# 简化版本（如果只需要主要指标）
def evaluation_batch_simple(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None,
                              save_px_vis_dir: str | None = None, save_px_limit: int | None = None):
    """简化版本，只返回主要指标"""
    results = evaluation_batch_multi_task(model, dataloader, device, _class_, max_ratio, resize_mask,
                                          save_px_vis_dir=save_px_vis_dir, save_px_limit=save_px_limit)
    
    return [
        # 异常检测指标（保持原有顺序）
        results['anomaly']['auroc_sp'],
        results['anomaly']['ap_sp'],
        results['anomaly']['f1_sp'],
        results['anomaly']['auroc_px'],
        results['anomaly']['ap_px'],
        results['anomaly']['f1_px'],
        results['anomaly']['aupro_px'],
        # 分类指标（新增）
        results['classification']['accuracy'],
        results['classification']['f1']
    ]

# #=====debug====
# def _stable_rank_scores(x: np.ndarray, seed: int = 1) -> np.ndarray:
#     # x: 1D float，返回加了极小抖动的副本（不改变 AUC/AP 的“宏观”排序，但打破 ties）
#     rng = np.random.default_rng(seed)
#     eps = rng.standard_normal(x.size).astype(np.float64) * 1e-12
#     y = x.astype(np.float64, copy=True)
#     y += eps
#     return y
# #===============

def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, use_metrics=['I-AUROC', 'I-AP', 'I-F1_max','P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)
        
    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()
    

    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))
    
    metrics = accum.summary()
    metric_results = {}
    
    # 检查是多分类还是二分类
    is_multiclass = len(np.unique(gt_sp)) > 2
    
    # #====debug===
    # flat_gt = gt_px.ravel()
    # flat_pr = pr_px.ravel()
    # flat_pr = _stable_rank_scores(flat_pr, seed=1)
    # #======
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            if is_multiclass:
                # 对于多分类但只有异常分数的情况，我们需要特殊处理
                # 将异常检测视为二分类问题（正常 vs 异常）
                # 或者使用其他适合的评估方式
                try:
                    # 方法1：将多分类转换为二分类（正常=0，异常=1）
                    binary_gt = (gt_sp > 0).astype(int)
                    auroc_sp = roc_auc_score(binary_gt, pr_sp)
                except:
                    auroc_sp = 0.0
            else:
                # 二分类
                auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
            
        elif metric.startswith('I-AP'):
            if is_multiclass:
                try:
                    # 同样将多分类转换为二分类
                    binary_gt = (gt_sp > 0).astype(int)
                    ap_sp = average_precision_score(binary_gt, pr_sp)
                except:
                    ap_sp = 0.0
            else:
                ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
            
        elif metric.startswith('I-F1_max'):
            if is_multiclass:
                try:
                    binary_gt = (gt_sp > 0).astype(int)
                    best_f1_score_sp = f1_score_max(binary_gt, pr_sp)
                except:
                    best_f1_score_sp = 0.0
            else:
                best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
            
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            P_AP_me = average_precision_score(gt_px.ravel(), pr_px.ravel())
            # P_AP_me = average_precision_score(flat_gt, flat_pr)
            metric_results[metric] = P_AP_me
        elif metric.startswith('P-F1_max'):
            # best_f1_score_px = f1_score_max(flat_gt, flat_pr)
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
            
    return list(metric_results.values())

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
        # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
        # hflip img
    flipped_img = hflip_img(img)
    augment_img = torch.cat([augment_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(img)
    augment_img = torch.cat([augment_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(img, angle)
        augment_img = torch.cat([augment_img, rotate90_img], dim=0)
    augment_img = (augment_img[torch.randperm(augment_img.size(0))])
    return augment_img

def modify_grad(x, inds, factor=0.):
    # print(inds.shape)
    inds = inds.expand_as(x)
    # print(x.shape)
    # print(inds.shape)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x

def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        factor = (point_dist/mean_dist)**(y)
        # factor = factor/torch.max(factor)
        # factor = torch.clip(factor, min=min_grad)
        # print(thresh)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr

def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()

def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N

def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x



def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None,
                    save_px_vis_dir: str | None = None,  # 新增：保存目录（可选）
                    save_px_limit: int | None = None):     # 新增：最多保存多少张（可选）)
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []

    img_name_list = []

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)

            # 记录文件名（去后缀）
            names = [os.path.splitext(os.path.basename(p))[0] for p in img_path]
            img_name_list.extend(names)
            
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

         # ====== 在这里落盘用于 P-AP 的输入对 ======
        if save_px_vis_dir is not None:
            try:
                save_pap_inputs_images(
                    pr_px=pr_list_px,
                    gt_px=gt_list_px,
                    names=img_name_list,
                    save_dir=save_px_vis_dir,
                    save_limit=save_px_limit,
                    save_heatmap=True,
                    make_pair=True
                )
                print(f"[P-AP输入图] 已保存到：{save_px_vis_dir}")
            except Exception as e:
                print(f"[警告] 保存 P-AP 可视化失败：{e}")
        
        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])], ignore_index=True)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

# 新增多样性损失
def diversity_loss(proto_attn_maps, temperature=1.0):
    """Encourage prototypes to attend to different regions"""
    if len(proto_attn_maps) < 2:
        return 0.0
    device = proto_attn_maps[0].device
    n = len(proto_attn_maps)
    loss = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            a1 = proto_attn_maps[i].flatten(1)  # [B, H*W]
            a2 = proto_attn_maps[j].flatten(1)
            # Cosine similarity
            sim = F.cosine_similarity(a1, a2, dim=1).mean()
            loss -= sim  # 越不相似越好（负相关）
            count += 1
    return loss / (count + 1e-8) if count > 0 else 0.0

