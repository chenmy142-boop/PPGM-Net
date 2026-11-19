import os
import argparse
import random

import joblib
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier


BANDS = ["blue", "green", "red", "nir", "eir", "lwir"]


def fast_std(img, size=3):
    """
    计算局部标准差，用于描述局部纹理变化。
    """
    img = img.astype(np.float32)
    avg = ndimage.uniform_filter(img, size)
    avg_sq = ndimage.uniform_filter(img ** 2, size)
    std = np.sqrt(np.maximum(avg_sq - avg ** 2, 0))
    return std


def local_contrast_stretch(img):
    """
    对单通道图像进行局部对比度拉伸，增强弱光照区域的局部响应。
    """
    img = img.astype(np.float32)
    low = ndimage.minimum_filter(img, size=5)
    high = ndimage.maximum_filter(img, size=5)
    stretch = (img - low) / (high - low + 1e-8)
    return stretch * 255.0


def calculate_indices_v4_pro(img_stack):
    """
    构建用于随机森林分类的多光谱物理特征。

    特征组成:
        1. 前五个反射波段的相对反射率。
        2. 长波红外原始响应。
        3. 光谱指数、纹理响应、热梯度、亮度和色相特征。
    """
    raw = img_stack.astype(np.float32)

    nir_enhanced = local_contrast_stretch(raw[:, :, 3])
    eir_enhanced = local_contrast_stretch(raw[:, :, 4])

    ndvi = (raw[:, :, 3] - raw[:, :, 2]) / (raw[:, :, 3] + raw[:, :, 2] + 1e-8)
    ndre = (raw[:, :, 4] - raw[:, :, 2]) / (raw[:, :, 4] + raw[:, :, 2] + 1e-8)
    thermal_ratio = raw[:, :, 5] / (raw[:, :, 1] + 1e-8)
    re_slope = (eir_enhanced - nir_enhanced) / (eir_enhanced + nir_enhanced + 1e-8)
    bndvi = (raw[:, :, 3] - raw[:, :, 0]) / (raw[:, :, 3] + raw[:, :, 0] + 1e-8)

    nir_std = fast_std(raw[:, :, 3], size=3)
    thermal_grad = ndimage.gaussian_gradient_magnitude(raw[:, :, 5], sigma=1)

    log_bright = np.log1p(np.mean(raw[:, :, :3], axis=2))

    rgb_for_hsv = np.stack(
        [raw[:, :, 2], raw[:, :, 1], raw[:, :, 0]],
        axis=2
    ) / 255.0
    hsv = mcolors.rgb_to_hsv(np.clip(rgb_for_hsv, 0, 1))
    hue = hsv[:, :, 0]

    raw_sum = np.sum(raw[:, :, :5], axis=2, keepdims=True) + 1e-8
    raw_relative = raw[:, :, :5] / raw_sum

    lwir_raw = raw[:, :, 5:6]

    indices = np.stack(
        [
            ndvi,
            ndre,
            thermal_ratio,
            re_slope,
            bndvi,
            nir_std,
            thermal_grad,
            log_bright,
            hue
        ],
        axis=2
    )

    return np.concatenate([raw_relative, lwir_raw, indices], axis=2)


def load_features(vis_path, gt_path=None):
    """
    根据可见光图像路径读取同目录下的多光谱波段，并生成像素级特征。
    """
    try:
        channels = []

        for band_name in BANDS:
            band_path = vis_path.replace("vis.png", f"{band_name}.png")
            channels.append(np.array(Image.open(band_path).convert("L")))

        img_stack = np.stack(channels, axis=2)
        features = calculate_indices_v4_pro(img_stack)

        return features.reshape(-1, 15), gt_path, img_stack.shape[:2]

    except Exception as error:
        print(f"Failed to load sample: {vis_path}. Error: {error}")
        return None, None, None


def train_global_model(train_txt_path, model_save_path):
    """
    训练全局随机森林分类器，用于将多光谱像素特征映射为目标概率。
    """
    print("Step 1: Sampling multispectral features for Random Forest training.")

    with open(train_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    x_train = []
    y_train = []

    random.shuffle(lines)

    hard_count = 0
    normal_count = 0

    for line in tqdm(lines):
        vis_path, gt_path = line.split(" ")

        is_hard_season = "01_Spring" in vis_path or "02_Summer" in vis_path
        if not is_hard_season and normal_count > 40:
            continue

        pixels, _, _ = load_features(vis_path)
        if pixels is None:
            continue

        gt = np.array(Image.open(gt_path).convert("L")).reshape(-1)
        labels = (gt > 100).astype(int)

        fg_idx = np.where(labels == 1)[0]
        bg_idx = np.where(labels == 0)[0]

        if len(fg_idx) > 20:
            log_bright_fg = pixels[fg_idx, 13]
            median_bright = np.median(pixels[:, 13])

            shadow_fg_idx = fg_idx[log_bright_fg < median_bright]
            sun_fg_idx = fg_idx[log_bright_fg >= median_bright]

            n_fg = 3000 if is_hard_season else 1500
            n_bg = 2000

            n_shadow = min(len(shadow_fg_idx), n_fg // 2)
            n_sun = min(len(sun_fg_idx), n_fg - n_shadow)

            if n_sun < n_fg - n_shadow:
                n_shadow = min(len(shadow_fg_idx), n_fg - n_sun)

            fg_shadow_samples = (
                np.random.choice(shadow_fg_idx, n_shadow, replace=False)
                if n_shadow > 0 else []
            )
            fg_sun_samples = (
                np.random.choice(sun_fg_idx, n_sun, replace=False)
                if n_sun > 0 else []
            )

            if len(fg_shadow_samples) == 0 and len(fg_sun_samples) == 0:
                continue

            fg_samples = np.concatenate([fg_shadow_samples, fg_sun_samples]).astype(int)
            bg_samples = np.random.choice(bg_idx, min(len(bg_idx), n_bg), replace=False)

            sample_idx = np.concatenate([fg_samples, bg_samples])

            x_train.append(pixels[sample_idx])
            y_train.append(labels[sample_idx])

            if is_hard_season:
                hard_count += 1
            else:
                normal_count += 1

        if hard_count + normal_count >= 150:
            break

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=22,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(x_train, y_train)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    joblib.dump(clf, model_save_path)

    return clf


def process_list(clf, txt_path, save_dir, desc):
    """
    使用训练好的随机森林模型生成物理先验概率图。
    """
    print(f"\nStep 2: Generating prior maps for {desc}.")

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    os.makedirs(save_dir, exist_ok=True)

    for line in tqdm(lines):
        vis_path = line.split(" ")[0]

        pixels, _, shape = load_features(vis_path)
        if pixels is None:
            continue

        prob = clf.predict_proba(pixels)[:, 1]

        p_min, p_max = np.percentile(prob, (1, 99))
        if p_max > p_min:
            prob = np.clip((prob - p_min) / (p_max - p_min + 1e-8), 0, 1)

        prob_map = (prob.reshape(shape) * 255).astype(np.uint8)

        parts = vis_path.replace("\\", "/").split("/")
        save_name = "_".join(parts[-6:-1]) + ".png"

        Image.fromarray(prob_map).save(os.path.join(save_dir, save_name))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate physical prior maps using a Random Forest classifier."
    )

    parser.add_argument(
        "--train_txt",
        type=str,
        default="data/train.txt",
        help="Training split file."
    )
    parser.add_argument(
        "--val_txt",
        type=str,
        default="data/val.txt",
        help="Validation split file."
    )
    parser.add_argument(
        "--test_txt",
        type=str,
        default="data/test.txt",
        help="Test split file."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/raw_priors",
        help="Directory for saving raw prior maps."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="checkpoints/rf_prior_model.pkl",
        help="Path for saving the trained Random Forest model."
    )

    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for new RF pixel sampling runs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = train_global_model(
        train_txt_path=args.train_txt,
        model_save_path=args.model_save_path
    )

    process_list(model, args.train_txt, args.save_dir, "training set")
    process_list(model, args.val_txt, args.save_dir, "validation set")
    process_list(model, args.test_txt, args.save_dir, "test set")
