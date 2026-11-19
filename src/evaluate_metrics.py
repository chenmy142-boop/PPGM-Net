import argparse
import os
import warnings
import cv2
import numpy as np
import py_sod_metrics
from tqdm import tqdm

warnings.filterwarnings("ignore")

def read_gray_image(path):
    """读取灰度图，兼容中文路径"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def get_prediction_filename(img_path):
    """根据测试列表生成预测图文件名"""
    clean_path = img_path.replace("\\", "/")
    lower_path = clean_path.lower()

    if "/dataset/" in lower_path:
        idx = lower_path.index("/dataset/")
        relative_path = clean_path[idx + len("/dataset/"):]
    else:
        parts = clean_path.split("/")
        relative_path = "/".join(parts[parts.index("dataset")+1:]) if "dataset" in parts else clean_path

    file_name = os.path.basename(relative_path).lower()
    path_core = os.path.dirname(relative_path) if file_name in ["vis.png","vis.jpg","vis.jpeg"] else os.path.splitext(relative_path)[0]
    pred_filename = path_core.replace("/", "_") + ".png"
    return pred_filename

def compute_target_recall(gt, pred, pred_threshold=128, overlap_threshold=0.10, min_gt_area=1):
    """按连通域统计目标级召回率"""
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred >= pred_threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)

    total_targets = 0
    detected_targets = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_gt_area:
            continue
        total_targets += 1
        gt_component = (labels == label_id)
        overlap_ratio = np.logical_and(gt_component, pred_bin).sum() / area
        if overlap_ratio >= overlap_threshold:
            detected_targets += 1

    return detected_targets, total_targets

def evaluate_model(test_txt, pred_dir, pred_threshold=128, target_overlap_threshold=0.1, min_gt_area=1):
    """只评价一个数据集（Ours）"""
    FM = py_sod_metrics.Fmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()

    total_detected_targets = 0
    total_gt_targets = 0
    missing_count = 0
    valid_count = 0

    with open(test_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("\nEvaluating model: Ours")

    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue

        items = line.split(" ")
        img_path = items[0]
        gt_path = items[1]

        pred_filename = get_prediction_filename(img_path)
        pred_path = os.path.join(pred_dir, pred_filename)

        if not os.path.exists(pred_path):
            missing_count += 1
            continue

        gt = read_gray_image(gt_path)
        pred = read_gray_image(pred_path)

        if gt is None or pred is None:
            continue

        _, gt = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY)

        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        FM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

        detected_targets, gt_targets = compute_target_recall(
            gt=gt,
            pred=pred,
            pred_threshold=pred_threshold,
            overlap_threshold=target_overlap_threshold,
            min_gt_area=min_gt_area
        )

        total_detected_targets += detected_targets
        total_gt_targets += gt_targets
        valid_count += 1

    if valid_count != sum(bool(line.strip()) for line in lines):
        raise RuntimeError(f"Incomplete evaluation: {valid_count} readable pairs; "
                           f"{missing_count} missing predictions.")
    fm_info = FM.get_results()
    em_info = EM.get_results()
    fm_curve = fm_info["fm"]["curve"]
    best_thresh_idx = np.argmax(fm_curve) if isinstance(fm_curve, (list, np.ndarray)) else 0
    target_recall = total_detected_targets / total_gt_targets if total_gt_targets > 0 else 0.0

    print("Valid images:", valid_count)
    print("Missing predictions:", missing_count)
    print("Detected targets:", total_detected_targets)
    print("Ground-truth targets:", total_gt_targets)

    return {
        "S-measure": SM.get_results()["sm"],
        "max-F": fm_curve[best_thresh_idx] if isinstance(fm_curve, (list, np.ndarray)) else fm_curve,
        "Target-Recall": target_recall,
        "adp-E": em_info["em"]["adp"],
        "MAE": MAE.get_results()["mae"]
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate camouflaged target detection results.")
    parser.add_argument("--test_txt", type=str, default="data/test.txt")
    parser.add_argument("--ours_dir", type=str, required=True)
    parser.add_argument("--pred_threshold", type=int, default=128)
    parser.add_argument("--target_overlap_threshold", type=float, default=0.1)
    parser.add_argument("--min_gt_area", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("="*75)
    ours_metrics = evaluate_model(
        test_txt=args.test_txt,
        pred_dir=args.ours_dir,
        pred_threshold=args.pred_threshold,
        target_overlap_threshold=args.target_overlap_threshold,
        min_gt_area=args.min_gt_area
    )

    print("\nFinal evaluation metrics:")
    for k, v in ours_metrics.items():
        print(f"{k}: {v:.5f}")