import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from lib.models import Network


class PriorGuidedTestDataset:
    """
    测试集读取类。

    根据 test.txt 中的 RGB 图像路径，自动匹配对应的精炼物理先验图。
    每个样本返回 RGB 图像、先验图、原始图像尺寸以及输出文件名。
    """

    def __init__(self, txt_path, prior_root, testsize=352):
        with open(txt_path, "r", encoding="utf-8") as f:
            self.rgb_paths = [
                line.strip().split(" ")[0]
                for line in f.readlines()
                if line.strip()
            ]

        self.prior_root = prior_root
        self.testsize = testsize

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.prior_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])

    def _build_sample_name(self, rgb_path):
        """
        根据 RGB 图像路径生成先验图和预测结果的文件名。

        示例:
            data/dataset/01_Spring/a/6/h0/0/vis.png

        生成:
            01_Spring_a_6_h0_0
        """
        clean_path = rgb_path.replace("\\", "/")
        parts = clean_path.split("/")
        lower_parts = [p.lower() for p in parts]

        if "dataset" in lower_parts:
            dataset_idx = lower_parts.index("dataset")
            target_folders = parts[dataset_idx + 1:-1]
            file_name = "_".join(target_folders)
        else:
            start_idx = -1

            for i, part in enumerate(parts):
                if (
                    part.startswith("01_")
                    or part.startswith("02_")
                    or part.startswith("03_")
                    or part.startswith("04_")
                ):
                    start_idx = i
                    break

            target_folders = parts[start_idx:-1] if start_idx != -1 else [parts[-2]]
            file_name = "_".join(target_folders)

        return file_name

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        file_name = self._build_sample_name(rgb_path)

        prior_path = os.path.join(
            self.prior_root,
            file_name + ".png"
        ).replace("\\", "/")

        rgb_img = Image.open(rgb_path).convert("RGB")
        original_size = rgb_img.size

        prior_img = Image.open(prior_path).convert("L")

        rgb_t = self.img_transform(rgb_img)
        prior_t = self.prior_transform(prior_img)

        return rgb_t, prior_t, original_size, file_name

    def __len__(self):
        return len(self.rgb_paths)


def load_model(model, model_weights_path, device):
    """
    加载模型权重。
    """
    try:
        state_dict = torch.load(
            model_weights_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(model_weights_path, map_location=device)

    model.load_state_dict(state_dict)
    return model


def generate_prediction_maps(
    test_txt,
    prior_dir,
    model_weights_path,
    save_dir,
    testsize=352,
    use_cpu=False
):
    """
    生成测试集预测结果图。

    参数:
        test_txt: 测试集划分文件。
        prior_dir: 精炼物理先验图目录。
        model_weights_path: 训练好的模型权重路径。
        save_dir: 预测结果保存目录。
        testsize: 模型输入尺寸。
        use_cpu: 是否使用 CPU 推理。
    """
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

    model = Network(channel=32, imagenet_pretrained=False).to(device)
    model = load_model(model, model_weights_path, device)
    model.eval()

    print(f"Model weights loaded from: {model_weights_path}")
    print(f"Using device: {device}")

    dataset = PriorGuidedTestDataset(test_txt, prior_dir, testsize=testsize)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Test samples: {len(dataset)}")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Generating prediction maps"):
            rgb_t, prior_t, original_size, name = dataset[i]

            rgb_t = rgb_t.unsqueeze(0).to(device)
            prior_t = prior_t.unsqueeze(0).to(device)

            preds = model(rgb_t, prior_t)
            res = preds[0]

            res = F.interpolate(
                res,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False
            )

            res = res.sigmoid().detach().cpu().numpy().squeeze()
            res_img = (res * 255).astype(np.uint8)

            save_filename = name + ".png"
            save_path = os.path.join(save_dir, save_filename)

            Image.fromarray(res_img).save(save_path)

    print(f"Prediction maps saved to: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prediction maps for PPGM-Net."
    )

    parser.add_argument(
        "--test_txt",
        type=str,
        default="data/test.txt",
        help="Path to the test split file."
    )
    parser.add_argument(
        "--prior_dir",
        type=str,
        default="data/priors/cleaned",
        help="Directory of cleaned physical prior maps."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="artifacts/table3/Net_epoch_best.pth",
        help="Path to model weights."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/predictions",
        help="Directory for saving prediction maps."
    )
    parser.add_argument(
        "--testsize",
        type=int,
        default=352,
        help="Input image size."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for inference."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_prediction_maps(
        test_txt=args.test_txt,
        prior_dir=args.prior_dir,
        model_weights_path=args.weights,
        save_dir=args.save_dir,
        testsize=args.testsize,
        use_cpu=args.cpu
    )