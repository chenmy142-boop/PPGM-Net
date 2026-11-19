import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter

from lib.models import Network
from utils.utils import adjust_lr, clip_gradient


def normalize_path(path):
    """将相对路径转换为当前运行目录下的绝对路径。"""
    return os.path.abspath(os.path.expanduser(path))


def set_seed(seed):
    """固定随机种子，减少重复实验中的随机扰动。"""
    print(f"Current random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    """为 DataLoader 的每个 worker 设置独立随机种子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PriorGuidedDataset(data.Dataset):
    """读取 RGB 图像、物理先验图和二值标注图。"""

    def __init__(self, txt_path, prior_root, trainsize=352):
        self.trainsize = trainsize
        self.prior_root = prior_root

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.imgs = []
        self.gts = []

        for line in lines:
            if line.strip():
                items = line.strip().split(' ')
                self.imgs.append(items[0])
                self.gts.append(items[1])

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_prior_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        rgb_path = self.imgs[index]
        gt_path = self.gts[index]

        clean_path = rgb_path.replace('\\', '/')
        parts = clean_path.split('/')

        lower_parts = [p.lower() for p in parts]

        if 'dataset' in lower_parts:
            dataset_idx = lower_parts.index('dataset')
            target_folders = parts[dataset_idx + 1: -1]
            file_name = "_".join(target_folders)
        else:
            start_idx = -1
            for i, p in enumerate(parts):
                if p.startswith('01_') or p.startswith('02_') or p.startswith('03_') or p.startswith('04_'):
                    start_idx = i
                    break

            target_folders = parts[start_idx: -1] if start_idx != -1 else [parts[-2]]
            file_name = "_".join(target_folders)

        prior_path = os.path.join(self.prior_root, file_name + '.png').replace('\\', '/')

        rgb_img = Image.open(rgb_path).convert('RGB')

        # 将标注图统一转换为二值掩码。
        gt_img = Image.open(gt_path).convert('L')
        gt_img = gt_img.point(lambda p: 255 if p > 0 else 0, mode='1').convert('L')

        try:
            prior_img = Image.open(prior_path).convert('L')
        except FileNotFoundError:
            print(f"Warning: prior map not found: {prior_path}")
            print(f"RGB image path: {rgb_path}")
            prior_img = Image.new('L', (rgb_img.size[0], rgb_img.size[1]), 0)

        rgb_tensor = self.img_transform(rgb_img)
        prior_tensor = self.gt_prior_transform(prior_img)
        gt_tensor = self.gt_prior_transform(gt_img)

        return rgb_tensor, prior_tensor, gt_tensor

    def __len__(self):
        return len(self.imgs)


def get_loader(txt_path, prior_root, batch_size, trainsize, shuffle=True, num_workers=0, seed=2024):
    """构建训练或验证数据加载器。"""
    dataset = PriorGuidedDataset(txt_path, prior_root, trainsize=trainsize)

    g = torch.Generator()
    g.manual_seed(seed)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g
    )

    return loader


def structure_loss(pred, mask):
    """结构感知损失，由加权 BCE 与加权 IoU 组成。"""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))

    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer, opt):
    """执行一个 epoch 的训练过程。"""
    global step

    model.train()
    loss_all = 0.0
    epoch_step = 0

    try:
        for i, (rgbs, priors, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            rgbs = rgbs.cuda()
            priors = priors.cuda()
            gts = gts.cuda()

            preds = model(rgbs, priors)

            loss = (
                structure_loss(preds[0], gts) +
                structure_loss(preds[1], gts) +
                structure_loss(preds[2], gts) +
                structure_loss(preds[3], gts)
            )

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == len(train_loader) or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, len(train_loader), loss.item()))

                logging.info(
                    '[Train Info]: Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format(epoch, opt.epoch, i, len(train_loader), loss.item())
                )

        loss_all /= epoch_step
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if epoch % 10 == 0:
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch))
            )

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(save_path, 'Net_epoch_interrupt.pth')
        )
        raise


def val(test_loader, model, epoch, save_path, writer):
    """在验证集上计算 MAE 并保存当前最优模型。"""
    global best_mae, best_epoch

    model.eval()

    with torch.no_grad():
        mae_sum = 0.0

        for i, (rgbs, priors, gts) in enumerate(test_loader):
            rgbs = rgbs.cuda()
            priors = priors.cuda()
            gts = gts.cuda()

            res = model(rgbs, priors)

            res_map = F.interpolate(
                res[0],
                size=gts.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            res_map = res_map.sigmoid()

            mae_sum += torch.abs(res_map - gts).mean().item()

        mae = mae_sum / len(test_loader)

        writer.add_scalar('MAE', mae, global_step=epoch)

        print(
            'Epoch: {}, MAE: {:.6f}, bestMAE: {:.6f}, bestEpoch: {}.'.
            format(epoch, mae, best_mae, best_epoch)
        )

        logging.info(
            '[Val Info]: Epoch: {}, MAE: {:.6f}, bestMAE: {:.6f}, bestEpoch: {}.'.
            format(epoch, mae, best_mae, best_epoch)
        )

        if epoch == 1:
            best_mae = mae
            best_epoch = epoch

            os.makedirs(save_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(save_path, 'Net_epoch_best.pth')
            )
            print('Save best model at epoch 1.')

        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch

                os.makedirs(save_path, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(save_path, 'Net_epoch_best.pth')
                )

                print('Save best model.')


def parse_args():
    parser = argparse.ArgumentParser(description='Train prior-guided camouflaged target detection network.')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=40)
    parser.add_argument('--backbone_weights', type=str, default=None, help='Local Res2Net weights; otherwise download the original weights.')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--train_txt', type=str, default='data/train.txt')
    parser.add_argument('--val_txt', type=str, default='data/val.txt')
    parser.add_argument('--prior_dir', type=str, default='data/priors/cleaned')
    parser.add_argument('--save_path', type=str, default='results/checkpoints')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()

    opt.train_txt = normalize_path(opt.train_txt)
    opt.val_txt = normalize_path(opt.val_txt)
    opt.prior_dir = normalize_path(opt.prior_dir)
    opt.save_path = normalize_path(opt.save_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    set_seed(opt.seed)

    save_path = os.path.join(opt.save_path, f'seed_{opt.seed}')
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_path, 'log.log'),
        level=logging.INFO,
        filemode='a',
        format='%(message)s'
    )

    logging.info('Start training')
    logging.info(f'Seed: {opt.seed}')
    logging.info(f'Save path: {save_path}')

    print('Loading prior-guided dataset...')
    print(f'Seed: {opt.seed}')
    print(f'Save path: {save_path}')

    train_loader = get_loader(
        txt_path=opt.train_txt,
        prior_root=opt.prior_dir,
        batch_size=opt.batchsize,
        trainsize=opt.trainsize,
        shuffle=True,
        num_workers=opt.num_workers,
        seed=opt.seed
    )

    val_loader = get_loader(
        txt_path=opt.val_txt,
        prior_root=opt.prior_dir,
        batch_size=1,
        trainsize=opt.trainsize,
        shuffle=False,
        num_workers=opt.num_workers,
        seed=opt.seed
    )

    model = Network(channel=32, backbone_weights=opt.backbone_weights).cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    writer = SummaryWriter(os.path.join(save_path, 'summary'))

    best_mae = 1.0
    best_epoch = 0
    step = 0

    print('Start training for {} epochs.'.format(opt.epoch))

    for epoch in range(1, opt.epoch + 1):
        cur_lr = adjust_lr(
            optimizer,
            opt.lr,
            epoch,
            opt.decay_rate,
            opt.decay_epoch
        )

        train(train_loader, model, optimizer, epoch, save_path, writer, opt)
        val(val_loader, model, epoch, save_path, writer)

    writer.close()

    print('Training finished.')
    print(f'Best MAE: {best_mae:.6f}')
    print(f'Best Epoch: {best_epoch}')
    print(f'Model saved in: {save_path}')
