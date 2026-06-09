import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有装 tqdm，就用一个假的，不影响运行
    def tqdm(iterable, **kwargs):
        return iterable


def calculate_auc_for_folders(pred_dir, gt_dir):
    # 支持的图片格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    # 获取预测文件夹中的所有合法图片文件名，并排序保证顺序一致
    img_names = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(valid_extensions)])

    auc_list = []

    print(f"✅ 找到 {len(img_names)} 张预测图片，开始计算 AUC...")

    for img_name in tqdm(img_names, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)

        # 检查对应的 GT 是否存在
        if not os.path.exists(gt_path):
            print(f"\n[警告] 找不到对应的 GT 文件: {gt_path}，已跳过。")
            continue

        # 以灰度模式读取图片
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            print(f"\n[警告] 无法读取图片: {img_name}，已跳过。")
            continue

        # 稳妥起见：如果预测图和 GT 尺寸有微小差异，强行将预测图 resize 到 GT 的尺寸
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 1. 将预测图转为 0~1 的概率值 (必须是 float 类型)
        pred_prob = pred.astype(np.float64) / 255.0

        # 2. 将 GT 转为 0 和 1 的二值绝对标签 (大于 0 的像素即视为真实被篡改的区域)
        gt_labels = (gt > 0).astype(int)

        # 3. 计算本张图片的 AUC
        try:
            # 必须展平 (flatten) 成一维向量，AUC 才能基于像素级别进行排序评分
            auc_score = roc_auc_score(gt_labels.flatten(), pred_prob.flatten())
            auc_list.append(auc_score)
        except ValueError:
            # 【致命异常捕捉】如果 GT 里面全黑 (全 0) 或全白 (全 1)，AUC 的底层数学逻辑是无法计算的！
            # 此时按照你之前 eval.py 的逻辑，强行记为 0.0 (或者你也可以选择跳过 `continue`)
            auc_list.append(0.0)

    # 汇总计算结果
    if len(auc_list) == 0:
        print("\n❌ 没有成功计算出任何图片的 AUC，请检查文件夹路径或图片后缀！")
        return None

    mean_auc = np.mean(auc_list)

    print(f"\n{'=' * 50}")
    print(f"📊 评估报告")
    print(f"{'-' * 50}")
    print(f"文件夹 A (预测): {pred_dir}")
    print(f"文件夹 B (GT)  : {gt_dir}")
    print(f"成功评估数量  : {len(auc_list)} 张")
    print(f"⭐ 最终平均 AUC: {mean_auc:.5f} ({mean_auc * 100:.3f}%)")
    print(f"{'=' * 50}\n")

    return mean_auc


if __name__ == "__main__":
    # ==========================================
    # ⚠️ 在这里修改你的文件夹绝对路径或相对路径
    # ==========================================
    FOLDER_A_PRED = "/home/chenzeyu/pycharmprojects/CamoDiffusion/results/SH971TTA/SH"  # 你的预测结果所在的文件夹 (A)
    FOLDER_B_GT = "/home/chenzeyu/TrainAndTest/DID_InpDiffusion/SH_GT"  # 真实标签所在的文件夹 (B)

    calculate_auc_for_folders(FOLDER_A_PRED, FOLDER_B_GT)