import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ================= 配置区域 =================
# 您可以在这里修改路径，或者通过命令行参数传入
DEFAULT_RESULT_PATH = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/unseen-zeroshot/result"
DEFAULT_GT_PATH = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/GT"
# ===========================================


def cal_kl(pred, gt, eps=1e-12):
    """计算 KL 散度 (KLD), 越低越好"""
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred, gt, eps=1e-12):
    """计算相似度 (SIM), 越高越好"""
    map1 = pred / (pred.sum() + eps)
    map2 = gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def cal_nss(pred, gt):
    """计算 NSS, 越高越好"""
    # 标准化预测图
    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    if pred_std == 0:
        return 0
    pred_norm = (pred - pred_mean) / pred_std

    # 离散化 GT (作为 fixation map)
    # AGD20K 的 GT 是热力图，通常取最大值的一定比例作为注视点
    gt_max = np.max(gt)
    gt_min = np.min(gt)
    if gt_max == gt_min:
        return 0

    gt_norm = (gt - gt_min) / (gt_max - gt_min)
    # 选取 GT 中激活度最高的前 10% 区域或者是阈值大于 0.1 的区域作为正样本
    fixation_map = gt_norm > 0.1

    if np.sum(fixation_map) == 0:
        return 0

    nss = np.mean(pred_norm[fixation_map])
    return nss


def process_prediction(pred_img, target_shape, sigma=15):
    """
    处理 SAM3 的预测结果：
    1. Resize 到目标大小 (224x224)
    2. 高斯平滑 (解决二值 mask 计算 KLD 的问题)
    3. 归一化到 [0, 1]
    """
    # Resize
    if pred_img.shape != target_shape:
        pred_img = cv2.resize(pred_img, (target_shape[1], target_shape[0]))  # cv2 uses (W, H)

    # 转为浮点数
    pred_img = pred_img.astype(np.float32)

    # 高斯平滑 (关键步骤)
    # SAM3 输出通常是 sharp 的，需要平滑来匹配 GT 的分布
    pred_img = gaussian_filter(pred_img, sigma=sigma)

    # 归一化 (Min-Max)
    pred_min = pred_img.min()
    pred_max = pred_img.max()
    if pred_max - pred_min > 1e-12:
        pred_img = (pred_img - pred_min) / (pred_max - pred_min)
    else:
        pred_img = np.zeros_like(pred_img)

    return pred_img


def process_gt(gt_img, target_shape):
    """处理 GT：Resize 并归一化"""
    if gt_img.shape != target_shape:
        gt_img = cv2.resize(gt_img, (target_shape[1], target_shape[0]))

    gt_img = gt_img.astype(np.float32)
    gt_img = gt_img / 255.0  # 假设 GT 读取进来是 0-255
    return gt_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--gt_path", type=str, default=DEFAULT_GT_PATH)
    parser.add_argument("--sigma", type=int, default=15, help="高斯平滑的 sigma 值")
    parser.add_argument("--save_csv", type=str, default="evaluation_results.csv", help="保存结果的文件名")
    args = parser.parse_args()

    # 存储结果的数据结构
    # detailed_scores[key] = {'KLD': [], 'SIM': [], 'NSS': []}
    # key 为 "{affordance}/{object}"
    detailed_scores = {}
    global_scores = {"KLD": [], "SIM": [], "NSS": []}

    print(f"Scanning results from: {args.result_path}")

    # 遍历预测结果文件夹
    # 假设结构是 result_path/affordance/object/image.png
    for root, dirs, files in os.walk(args.result_path):
        for file in tqdm(files, desc="Evaluating"):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # 获取相对路径 (例如: hit/axe/001.png)
            pred_abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(pred_abs_path, args.result_path)

            # 构造 GT 路径
            gt_abs_path = os.path.join(args.gt_path, rel_path)

            # 处理可能的后缀名不一致 (例如 pred 是 .png, gt 是 .jpg)
            if not os.path.exists(gt_abs_path):
                base_path = os.path.splitext(gt_abs_path)[0]
                found = False
                for ext in [".png", ".jpg", ".jpeg"]:
                    if os.path.exists(base_path + ext):
                        gt_abs_path = base_path + ext
                        found = True
                        break
                if not found:
                    # print(f"Warning: GT not found for {rel_path}")
                    continue

            # 读取图片 (灰度模式)
            pred_img = cv2.imread(pred_abs_path, 0)
            gt_img = cv2.imread(gt_abs_path, 0)

            if pred_img is None or gt_img is None:
                continue

            # 解析 affordance 和 object 名称
            path_parts = os.path.normpath(rel_path).split(os.sep)
            if len(path_parts) >= 3:
                affordance = path_parts[0]
                obj_name = path_parts[1]
                group_key = f"{affordance}-{obj_name}"
            else:
                group_key = "unknown"

            # 预处理 (统一为 224x224)
            target_shape = (224, 224)
            pred_processed = process_prediction(pred_img, target_shape, sigma=args.sigma)
            gt_processed = process_gt(gt_img, target_shape)

            # 计算指标
            kld = cal_kl(pred_processed, gt_processed)
            sim = cal_sim(pred_processed, gt_processed)
            nss = cal_nss(pred_processed, gt_processed)

            # 记录数据
            if group_key not in detailed_scores:
                detailed_scores[group_key] = {"KLD": [], "SIM": [], "NSS": []}

            detailed_scores[group_key]["KLD"].append(kld)
            detailed_scores[group_key]["SIM"].append(sim)
            detailed_scores[group_key]["NSS"].append(nss)

            global_scores["KLD"].append(kld)
            global_scores["SIM"].append(sim)
            global_scores["NSS"].append(nss)

    # ================= 结果汇总与输出 =================
    print("\n" + "=" * 50)
    print(" >>> 全局评测结果 (Global Metrics) <<<")
    print("=" * 50)

    if len(global_scores["KLD"]) > 0:
        g_kld = np.mean(global_scores["KLD"])
        g_sim = np.mean(global_scores["SIM"])
        g_nss = np.mean(global_scores["NSS"])
        print(f"Total Images: {len(global_scores['KLD'])}")
        print(f"mKLD : {g_kld:.4f} (Lower is better)")
        print(f"mSIM : {g_sim:.4f} (Higher is better)")
        print(f"mNSS : {g_nss:.4f} (Higher is better)")
    else:
        print("No valid image pairs found!")
        return

    print("\n" + "=" * 50)
    print(" >>> 分组评测结果 (Per Affordance-Object) <<<")
    print(f"{'Group (Aff-Obj)':<30} | {'KLD':<8} | {'SIM':<8} | {'NSS':<8} | {'Count'}")
    print("-" * 70)

    # 准备保存到 CSV 的数据列表
    csv_data = []

    # 按名称排序输出
    for key in sorted(detailed_scores.keys()):
        metrics = detailed_scores[key]
        m_kld = np.mean(metrics["KLD"])
        m_sim = np.mean(metrics["SIM"])
        m_nss = np.mean(metrics["NSS"])
        count = len(metrics["KLD"])

        print(f"{key:<30} | {m_kld:.4f}   | {m_sim:.4f}   | {m_nss:.4f}   | {count}")

        csv_data.append({"Affordance-Object": key, "KLD": m_kld, "SIM": m_sim, "NSS": m_nss, "Count": count})

    # 添加全局平均到 CSV
    csv_data.append(
        {
            "Affordance-Object": "GLOBAL_AVERAGE",
            "KLD": g_kld,
            "SIM": g_sim,
            "NSS": g_nss,
            "Count": len(global_scores["KLD"]),
        }
    )

    # 保存 CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(args.save_csv, index=False)
    print("=" * 50)
    print(f"详细结果已保存至: {args.save_csv}")


if __name__ == "__main__":
    main()
