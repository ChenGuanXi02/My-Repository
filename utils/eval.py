import os
import time
import cv2
import numpy as np
import pandas as pd
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


    # 计算 Accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-7)  # 防止除以零

    # 计算 AUC
    try:
        auc_score = roc_auc_score(gt_labels.flatten(), pred_prob.flatten())  # 计算 AUC
    except ValueError:
        auc_score = 0.0  # 如果无法计算（如全是 0 或全是 1），AUC 设为 0

    # 按要求格式打印（仅文件名、F1、AUC，保留3位小数）
    # print(f"{mask_name} F1:{f1_score:.3f} AUC:{auc_score:.3f}")

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
    # # 如果 MCC 是 NaN，则跳过该条数据
    # if np.isnan(mcc):
    #     return None  # 返回 None，表示这一条数据不再参与计算

    # 计算TPAMI中错误的FPR
    false_FPR = TP / (TP + TN + 1e-7)  # 防止除以零

    return accuracy, f1_score, auc_score, iou, fpr, mcc, false_FPR

#
# 主评估函数
def eval(mask_path='./Dataset/TestDataset',
         pred_path='./results',
         dataset_name='LR'):  # 支持多个数据集进行评估
    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160
    global mask_root, pred_root

    # 遍历多个数据集进行评估
    for dataset in [dataset_name]:

        # 1. 默认的 GT 文件夹路径 (适配大多数情况)
        default_gt_root = os.path.join(mask_path, dataset, 'GT')

        # 2. 手动设置的特殊 Groundtruth 文件夹名称 (适配特殊情况)
        # ⚠️ 这里你可以根据需要改成 'Groundtruth', 'mask', 'masks' 等
        manual_gt_root = '/home/chenzeyu/TrainAndTest/DID_InpDiffusion/LR_GT'

        # 3. 依次判断文件夹是否存在
        if os.path.exists(default_gt_root):
            mask_root = default_gt_root
        elif os.path.exists(manual_gt_root):
            mask_root = manual_gt_root
        else:
            # 都不存在时，在控制台输出错误，并直接抛出异常中断程序
            error_msg = (
                f"\n{'=' * 50}\n"
                f"❌ 错误：找不到数据集 '{dataset}' 的标签文件夹！\n"
                f"代码尝试了以下两个路径，但都不存在：\n"
                f"  1. {default_gt_root}\n"
                f"  2. {manual_gt_root}\n"
                f"请检查你的数据集路径或手动修改 manual_gt_folder_name 参数。\n"
                f"{'=' * 50}\n"
            )
            print(error_msg)
            raise FileNotFoundError(f"找不到数据集 {dataset} 的 GT 目录。")

        # 4. 在控制台上打印出来访问的 GT 地址
        print(f"✅ 成功匹配标签路径，当前正在访问的 GT 文件夹为: {mask_root}")

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

        # # 调用 thread_map 进行并行评估
        # res1 = thread_map(measure_mea_change, mask_name_list, max_workers=8, chunksize=4)
        #
        # # 提取评估指标
        # accuracies = [x[0] for x in res1]
        # f1_scores = [x[1] for x in res1]
        # auc_scores = [x[2] for x in res1]
        # ious = [x[3] for x in res1]
        # fprs = [x[4] for x in res1]
        # mccs = [x[5] for x in res1]
        # false_FPRs = [x[6] for x in res1]
        #
        # # 计算每个指标的平均值
        # results1 = {
        #     "Accuracy": np.nanmean(np.array(accuracies, dtype=np.float64)),
        #     "F1_score": np.nanmean(np.array(f1_scores, dtype=np.float64)),
        #     "AUC": np.nanmean(np.array(auc_scores, dtype=np.float64)),
        #     "IoU": np.nanmean(np.array(ious, dtype=np.float64)),
        #     "FPR": np.nanmean(np.array(fprs, dtype=np.float64)),
        #     "MCC": np.nanmean(np.array(mccs, dtype=np.float64)),  # <--- 使用 np.nanmean
        #     "False_FPR": np.nanmean(np.array(false_FPRs, dtype=np.float64)),
        # }
        # # 输出每个数据集的评估结果
        # print(f"{dataset}: {results1}")

        # 并行计算所有图像的指标
        results_list = thread_map(measure_mea_change, mask_name_list, max_workers=8, chunksize=4)

        # --- 使用 Pandas 进行后续处理 ---

        # 1. 将结果列表转换为 DataFrame
        # 创建一个字典，键是指标名称，值是从结果列表中提取的对应指标
        metrics_dict = {
            "Accuracy": [res[0] for res in results_list],
            "F1_score": [res[1] for res in results_list],
            "AUC": [res[2] for res in results_list],
            "IoU": [res[3] for res in results_list],
            "FPR": [res[4] for res in results_list],
            "MCC": [res[5] for res in results_list],
            "False_FPR": [res[6] for res in results_list],
        }
        df = pd.DataFrame(metrics_dict)

        # 2. 使用 .mean() 计算平均值 (Pandas 默认忽略 NaN)
        # .mean() 返回一个 Pandas Series，我们可以用 .to_dict() 将其转换回字典
        average_results = df.mean().to_dict()

        # 记得删掉
        # --- 新增：统计 MCC 为 NaN 的数量 ---
        # 使用 .isna() 或 .isnull() 找出 NaN 值，然后用 .sum() 计数
        mcc_nan_count = df['MCC'].isna().sum()
        average_results['MCC_nan_count'] = mcc_nan_count

        # 3. 输出结果
        print(f"{dataset}: {average_results}")

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



# # 结果存excel版，记得改excel名
# import os
# import numpy as np
# import pandas as pd  # 新增：用于处理Excel
# from PIL import Image
# import cv2
# from sklearn.metrics import roc_auc_score
# from tqdm.contrib.concurrent import thread_map  # 假设thread_map来自tqdm
#
# def eval(mask_path='./Dataset/TestDataset',
#          pred_path='./results',
#          dataset_name='NIST16',
#          excel_output='NIST16_0730.xlsx'):  # 新增参数：Excel输出路径
#     global mask_root, pred_root
#     # 存储Excel数据的列表（每个元素是一个字典：{'name': ..., 'F1': ..., 'AUC': ...}）
#     excel_data = []
#
#     for dataset in [dataset_name]:
#         mask_root = os.path.join(mask_path, dataset, 'GT')
#         pred_root = os.path.join(pred_path, dataset)
#         mask_name_list = sorted(os.listdir(mask_root))
#
#         # 检查路径有效性
#         if not os.path.exists(mask_root):
#             print(f"Warning: {mask_root} does not exist.")
#             continue
#         if not os.path.exists(pred_root):
#             print(f"Warning: {pred_root} does not exist.")
#             continue
#         if len(mask_name_list) == 0:
#             print(f"Warning: No ground truth masks found in {mask_root}.")
#             continue
#
#         # 并行处理所有mask
#         res1 = thread_map(measure_mea_change, mask_name_list, max_workers=8, chunksize=4)
#
#         # 过滤无效结果（MCC为NaN的情况），并关联对应的mask_name
#         valid_indices = [i for i, res in enumerate(res1) if res is not None]
#         valid_masks = [mask_name_list[i] for i in valid_indices]
#         valid_results = [res1[i] for i in valid_indices]
#
#         # 提取有效结果中的F1和AUC，存入excel_data
#         for mask_name, res in zip(valid_masks, valid_results):
#             f1 = res[1]  # res[1]是f1_score
#             auc = res[2]  # res[2]是auc_score
#             excel_data.append({
#                 'name': mask_name,
#                 'F1': round(f1, 3),  # 保留3位小数
#                 'AUC': round(auc, 3)
#             })
#
#         # 计算并打印平均值
#         accuracies = [x[0] for x in valid_results]
#         f1_scores = [x[1] for x in valid_results]
#         auc_scores = [x[2] for x in valid_results]
#         ious = [x[3] for x in valid_results]
#         fprs = [x[4] for x in valid_results]
#         mccs = [x[5] for x in valid_results]
#         false_FPRs = [x[6] for x in valid_results]
#
#         results1 = {
#             "Accuracy": np.mean(np.array(accuracies, dtype=np.float64)),
#             "F1_score": np.mean(np.array(f1_scores, dtype=np.float64)),
#             "AUC": np.mean(np.array(auc_scores, dtype=np.float64)),
#             "IoU": np.mean(np.array(ious, dtype=np.float64)),
#             "FPR": np.mean(np.array(fprs, dtype=np.float64)),
#             "MCC": np.mean(np.array(mccs, dtype=np.float64)),
#             "False_FPR": np.mean(np.array(false_FPRs, dtype=np.float64)),
#         }
#         print(f"{dataset}: {results1}")
#
#     # 将收集的数据存入Excel
#     if excel_data:
#         df = pd.DataFrame(excel_data)
#         df.to_excel(excel_output, index=False)  # 不保留索引列
#         print(f"结果已保存到Excel：{excel_output}")
#     else:
#         print("没有有效数据可存入Excel")
#
# # ##########################################################
#         res = process_map(measure_mea, mask_name_list, max_workers=8, chunksize=4)
#
#         sms = [x[0] for x in res]
#         changeable_ems = [x[1] for x in res]
#         wfms = [x[2] for x in res]
#         maes = [x[3] for x in res]
#
#         results = {
#             "Smeasure": np.mean(np.array(sms, dtype=np.float64)),
#             "wFmeasure": np.mean(np.array(wfms, dtype=np.float64)),
#             "MAE": np.mean(np.array(maes, dtype=np.float64)),
#             "meanEm": np.mean(np.array(changeable_ems, dtype=np.float64), axis=0).mean(),
#             "maxEm": np.mean(np.array(changeable_ems, dtype=np.float64), axis=0).max(),
#         }
#         print(dataset_name, ":", results)
#         return results