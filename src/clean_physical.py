import os
import math
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def calculate_adaptive_pixel_threshold(
    target_area_m2=0.32,
    raw_gsd_cm=10.0,
    raw_size=512,
    model_size=352
):
    """
    根据目标最小物理面积和成像分辨率计算连通域面积阈值。

    参数:
        target_area_m2: 目标最小物理投影面积，单位为平方米。
        raw_gsd_cm: 原始图像的地面采样距离，单位为厘米/像素。
        raw_size: 原始图像尺寸。
        model_size: 模型输入尺寸。

    返回:
        final_threshold: 连通域面积阈值，单位为像素。
    """
    effective_gsd_m = (raw_gsd_cm / 100.0) * (raw_size / model_size)
    pixel_threshold = target_area_m2 / (effective_gsd_m ** 2)
    final_threshold = math.ceil(pixel_threshold)

    print(f"[Physical Constraint] target area: {target_area_m2} m^2")
    print(f"[Sensor Setting] raw GSD: {raw_gsd_cm} cm/pixel, input size: {model_size}")
    print(f"[Threshold] effective GSD: {effective_gsd_m:.4f} m/pixel, pixel threshold: {final_threshold}")

    return final_threshold


def clean_physical_priors_adaptive(input_dir, output_dir, area_threshold_pixel):
    """
    对原始物理先验图进行连通域面积过滤。

    处理流程:
        1. 读取原始先验图。
        2. 根据固定阈值提取高响应区域。
        3. 统计每个连通域的像素面积。
        4. 保留面积不小于物理尺度阈值的候选区域。
        5. 将清洗后的先验图保存到输出目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    img_names = [name for name in os.listdir(input_dir) if name.lower().endswith(".png")]

    for name in tqdm(img_names, desc="Cleaning physical priors"):
        img_path = os.path.join(input_dir, name)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thresh,
            connectivity=8
        )

        cleaned_img = np.zeros_like(thresh)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= area_threshold_pixel:
                cleaned_img[labels == label_id] = 255

        out_path = os.path.join(output_dir, name)
        cv2.imencode(".png", cleaned_img)[1].tofile(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean raw physical prior maps using a GSD-based area constraint."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/raw_priors",
        help="Directory of raw physical prior maps."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/priors/cleaned",
        help="Directory for saving cleaned physical prior maps."
    )
    parser.add_argument(
        "--gsd_cm",
        type=float,
        default=10.0,
        help="Ground sampling distance of the raw image, in cm/pixel."
    )
    parser.add_argument(
        "--target_area",
        type=float,
        default=0.317,
        help="Minimum target projection area, in square meters."
    )
    parser.add_argument(
        "--raw_size",
        type=int,
        default=512,
        help="Raw image size used for GSD conversion."
    )
    parser.add_argument(
        "--model_size",
        type=int,
        default=352,
        help="Model input size."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dynamic_tau = calculate_adaptive_pixel_threshold(
        target_area_m2=args.target_area,
        raw_gsd_cm=args.gsd_cm,
        raw_size=args.raw_size,
        model_size=args.model_size
    )

    print(f"Start physical prior cleaning. Pixel threshold: {dynamic_tau}")
    clean_physical_priors_adaptive(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        area_threshold_pixel=dynamic_tau
    )
    print(f"Cleaning finished. Results saved to: {args.output_dir}")
