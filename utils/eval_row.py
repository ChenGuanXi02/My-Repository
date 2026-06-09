import os
import time
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

import numba
from tqdm import tqdm
from utils.metrics import Emeasure, Smeasure, WeightedFmeasure, _cal_mae
from utils.metrics import _prepare_data
from tqdm.contrib.concurrent import thread_map, process_map  # or thread_map, process_map
from PIL import Image


# #####################
SM = Smeasure()
WFM = WeightedFmeasure()
# 计算评估指标的函数



@numba.jit(nopython=True)
def generate_parts_numel_combinations(fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel, gt_fg_numel, gt_size):
    bg_fg_numel = gt_fg_numel - fg_fg_numel
    bg_bg_numel = pred_bg_numel - bg_fg_numel

    parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

    mean_pred_value = pred_fg_numel / gt_size
    mean_gt_value = gt_fg_numel / gt_size

    demeaned_pred_fg_value = 1 - mean_pred_value
    demeaned_pred_bg_value = 0 - mean_pred_value
    demeaned_gt_fg_value = 1 - mean_gt_value
    demeaned_gt_bg_value = 0 - mean_gt_value

    combinations = [
        (demeaned_pred_fg_value, demeaned_gt_fg_value),
        (demeaned_pred_fg_value, demeaned_gt_bg_value),
        (demeaned_pred_bg_value, demeaned_gt_fg_value),
        (demeaned_pred_bg_value, demeaned_gt_bg_value),
    ]
    return parts_numel, combinations

def cal_em_with_cumsumhistogram(pred: np.ndarray, gt: np.ndarray, gt_fg_numel, gt_size) -> np.ndarray:
    pred = (pred * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
    fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
    fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
    fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

    fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
    bg___numel_w_thrs = gt_size - fg___numel_w_thrs

    if gt_fg_numel == 0:
        enhanced_matrix_sum = bg___numel_w_thrs
    elif gt_fg_numel == gt_size:
        enhanced_matrix_sum = fg___numel_w_thrs
    else:
        parts_numel_w_thrs, combinations = generate_parts_numel_combinations(
            fg_fg_numel=fg_fg_numel_w_thrs,
            fg_bg_numel=fg_bg_numel_w_thrs,
            pred_fg_numel=fg___numel_w_thrs,
            pred_bg_numel=bg___numel_w_thrs,
            gt_fg_numel=gt_fg_numel,
            gt_size=gt_size,
        )

        results_parts = np.empty(shape=(4, 256), dtype=np.float64)
        for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
            align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + np.spacing(1))
            )
            enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
            results_parts[i] = enhanced_matrix_value * part_numel
        enhanced_matrix_sum = results_parts.sum(axis=0)

    em = enhanced_matrix_sum / (gt_size - 1 + np.spacing(1))
    return em

def measure_mea(mask_name):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)

    ###############################################################
    # gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    ################################################################
    # 检查扩展名，如果是.tif，使用PIL加载
    _, ext_mask = os.path.splitext(mask_path)
    _, ext_pred = os.path.splitext(pred_path)

    if ext_mask.lower() == '.tif':
        gt = Image.open(mask_path).convert('L')  # 使用PIL加载TIF并转为灰度
        gt = np.array(gt)  # 转换为NumPy数组以便后续处理
    else:
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 使用OpenCV加载其他格式

    if ext_pred.lower() == '.tif':
        pred = Image.open(pred_path).convert('L')  # 使用PIL加载TIF并转为灰度
        pred = np.array(pred)  # 转换为NumPy数组
    else:
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 使用OpenCV加载其他格式
    #############################################################################

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray):
        sm = self.cal_sm(pred, gt)
        return sm

    def cal_em(pred: np.ndarray, gt: np.ndarray):
        # Here we do not use EM() class to avoid multiple process conflict
        gt_fg_numel = np.count_nonzero(gt)
        gt_size = gt.shape[0] * gt.shape[1]
        changeable_em = cal_em_with_cumsumhistogram(pred, gt, gt_fg_numel, gt_size)
        return changeable_em

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray):
        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        return wfm

    pred, gt = _prepare_data(pred=pred, gt=gt)
    sm = cal_sm(SM, pred, gt)
    changeable_em = cal_em(pred, gt)
    wfm = cal_wfm(WFM, pred, gt)
    mae = _cal_mae(pred, gt)
    return sm, changeable_em, wfm, mae



#######################################
def measure_mea_change(mask_name):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)

    ###############################################################
    # gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    ################################################################
    # 检查扩展名，如果是.tif，使用PIL加载
    a, ext_mask = os.path.splitext(mask_path)
    # print(a, ext_mask)
    b, ext_pred = os.path.splitext(pred_path)
    # print(b, ext_pred)

    if ext_mask.lower() == '.tif':
        gt = Image.open(mask_path).convert('L')  # 使用PIL加载TIF并转为灰度
        gt = np.array(gt)  # 转换为NumPy数组以便后续处理
    else:
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 使用OpenCV加载其他格式

    if ext_pred.lower() == '.tif':
        pred = Image.open(pred_path).convert('L')  # 使用PIL加载TIF并转为灰度
        pred = np.array(pred)  # 转换为NumPy数组
    else:
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)  # 使用OpenCV加载其他格式
    #############################################################################



    # 将预测的灰度值转化为概率值（0 到 1之间）
    pred_prob = pred / 255.0  # 假设灰度值在0到255之间，转化为0到1之间的概率值

    # 将概率值转化为二分类标签（0 或 1），阈值为 0.5
    pred_labels = (pred_prob >= 0.5).astype(int)  # 如果大于等于 0.5 预测为 1，否则预测为 0
    gt_labels = (gt > 0).astype(int)  # 假设gt标签为0或255，255代表正类，0代表负类

    # 计算 TP, FP, FN, TN
    TP = ((pred_labels == 1) & (gt_labels == 1)).sum()  # True Positives
    FP = ((pred_labels == 1) & (gt_labels == 0)).sum()  # False Positives
    FN = ((pred_labels == 0) & (gt_labels == 1)).sum()  # False Negatives
    TN = ((pred_labels == 0) & (gt_labels == 0)).sum()  # True Negatives
    # 将 TP, FP, TN, FN 转换为 np.float64 类型，以避免溢出
    TP = np.float64(TP)
    FP = np.float64(FP)
    TN = np.float64(TN)
    FN = np.float64(FN)
    # print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")

    # 计算 Precision, Recall 和 F1 分数
    precision = TP / (TP + FP + 1e-7)  # 防止除以零
    recall = TP / (TP + FN + 1e-7)  # 防止除以零
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)  # 防止除以零

    # print(a, ext_mask, f1_score)

    # 计算 Accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-7)  # 防止除以零

    # 计算 AUC
    try:
        auc_score = roc_auc_score(gt_labels.flatten(), pred_prob.flatten())  # 计算 AUC
    except ValueError:
        auc_score = 0.0  # 如果无法计算（如全是 0 或全是 1），AUC 设为 0

    print(a, ext_mask, auc_score)

    # 计算 IoU
    iou = TP / (TP + FP + FN + 1e-7)  # 防止除以零

    # 计算 FPR
    fpr = FP / (FP + TN + 1e-7)  # 防止除以零

    # 计算 MCC
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # print(denominator)
    # 对分母进行保护，防止溢出和无效值
    if np.any(np.isnan(denominator)) or np.any(denominator == 0):
        mcc = np.nan  # 如果分母为零或无效值，返回 NaN
    else:
        mcc = (TP * TN - FP * FN) / (denominator + 1e-7)  # 防止除以零
    # 如果 MCC 是 NaN，则跳过该条数据
    if np.isnan(mcc):
        return None  # 返回 None，表示这一条数据不再参与计算

    # 计算TPAMI中错误的FPR
    false_FPR = TP / (TP + TN + 1e-7)  # 防止除以零

    return accuracy, f1_score, auc_score, iou, fpr, mcc, false_FPR


# 主评估函数
def eval(mask_path='./Dataset/TestDataset',
         pred_path='./results',
         dataset_name='NIST16_160'):  # 支持多个数据集进行评估
    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160
    global mask_root, pred_root

    # 遍历多个数据集进行评估
    for dataset in [dataset_name]:
        mask_root = os.path.join(mask_path, dataset, 'GT')
        pred_root = os.path.join(pred_path, dataset)
        mask_name_list = sorted(os.listdir(mask_root))
        # 确保读取文件的路径正确
        if not os.path.exists(mask_root):
            print(f"Warning: {mask_root} does not exist.")
            continue
        if not os.path.exists(pred_root):
            print(f"Warning: {pred_root} does not exist.")
            continue
        if len(mask_name_list) == 0:
            print(f"Warning: No ground truth masks found in {mask_root}.")
            continue

        # 调用 thread_map 进行并行评估
        res1 = thread_map(measure_mea_change, mask_name_list, max_workers=8, chunksize=4)

        # 过滤掉返回 None 的数据（即 MCC 为 NaN 的数据）
        res1 = [x for x in res1 if x is not None]

        # 提取评估指标
        accuracies = [x[0] for x in res1]
        f1_scores = [x[1] for x in res1]
        auc_scores = [x[2] for x in res1]
        ious = [x[3] for x in res1]
        fprs = [x[4] for x in res1]
        mccs = [x[5] for x in res1]
        false_FPRs = [x[6] for x in res1]

        # 计算每个指标的平均值
        results1 = {
            "Accuracy": np.mean(np.array(accuracies, dtype=np.float64)),
            "F1_score": np.mean(np.array(f1_scores, dtype=np.float64)),
            "AUC": np.mean(np.array(auc_scores, dtype=np.float64)),
            "IoU": np.mean(np.array(ious, dtype=np.float64)),
            "FPR": np.mean(np.array(fprs, dtype=np.float64)),
            "MCC": np.mean(np.array(mccs, dtype=np.float64)),
            "False_FPR": np.mean(np.array(false_FPRs, dtype=np.float64)),
        }
        # 输出每个数据集的评估结果
        print(f"{dataset}: {results1}")

# ##########################################################
        res = process_map(measure_mea, mask_name_list, max_workers=8, chunksize=4)

        sms = [x[0] for x in res]
        changeable_ems = [x[1] for x in res]
        wfms = [x[2] for x in res]
        maes = [x[3] for x in res]

        results = {
            "Smeasure": np.mean(np.array(sms, dtype=np.float64)),
            "wFmeasure": np.mean(np.array(wfms, dtype=np.float64)),
            "MAE": np.mean(np.array(maes, dtype=np.float64)),
            "meanEm": np.mean(np.array(changeable_ems, dtype=np.float64), axis=0).mean(),
            "maxEm": np.mean(np.array(changeable_ems, dtype=np.float64), axis=0).max(),
        }
        print(dataset_name, ":", results)
        return results
