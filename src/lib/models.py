import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s


class RF(nn.Module):
    """多分支感受野增强模块，用于提取不同尺度的上下文特征。"""

    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7))
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class NCD(nn.Module):
    """邻接特征聚合解码模块，用于生成初始预测图。"""

    def __init__(self, channel=32):
        super(NCD, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


class GRA(nn.Module):
    """分组反向注意力模块，用于逐级细化目标响应。"""

    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True))
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            temp = []
            for i in range(32):
                temp.extend([xs[i], y])
            x_cat = torch.cat(temp, 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y_out = y + self.score(x)
        return x, y_out, y_out


class ReverseStage(nn.Module):
    """反向引导精修阶段，用于逐层补充目标细节。"""

    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        y = -1 * (torch.sigmoid(y)) + 1
        x, y, score = self.weak_gra(x, y)
        x, y, score = self.medium_gra(x, y)
        _, _, score = self.strong_gra(x, y)
        return score


class Network(nn.Module):
    """物理先验引导的多层调制分割网络。"""

    def __init__(self, channel=32, imagenet_pretrained=True, backbone_weights=None):
        super(Network, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained, weights_path=backbone_weights)

        self.rf1 = RF(256, channel)
        self.rf2 = RF(512, channel)
        self.rf3 = RF(1024, channel)
        self.rf4 = RF(2048, channel)
        self.ncd = NCD(channel)

        self.gra1 = GRA(channel, channel)
        self.gra2 = GRA(channel, channel)
        self.gra3 = GRA(channel, channel)
        self.rs1 = ReverseStage(channel)
        self.rs2 = ReverseStage(channel)
        self.rs3 = ReverseStage(channel)

    def forward(self, rgb, prior_map):
        # 保存输入尺寸，用于将多级预测结果恢复到原始分辨率。
        input_size = rgb.size()[2:]

        # 提取浅层视觉特征。
        x = self.resnet.conv1(rgb)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # 将物理先验缩放到对应特征层尺寸，并进行残差式空间调制。
        x1 = self.resnet.layer1(x)
        p1 = F.interpolate(prior_map, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1 = x1 * (1 + p1)

        x2 = self.resnet.layer2(x1)
        p2 = F.interpolate(prior_map, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x2 = x2 * (1 + p2)

        x3 = self.resnet.layer3(x2)
        p3 = F.interpolate(prior_map, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x3 = x3 * (1 + p3)

        x4 = self.resnet.layer4(x3)
        p4 = F.interpolate(prior_map, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x4 = x4 * (1 + p4)

        # 通过多尺度感受野模块统一通道维度并增强上下文表达。
        x1_rf = self.rf1(x1)
        x2_rf = self.rf2(x2)
        x3_rf = self.rf3(x3)
        x4_rf = self.rf4(x4)

        # 解码高层语义特征，生成初始目标预测。
        init_map = self.ncd(x4_rf, x3_rf, x2_rf)

        # 采用级联反向引导机制逐级细化预测结果。
        guidance1 = F.interpolate(init_map, size=x3_rf.shape[2:], mode='bilinear', align_corners=False)
        rg1 = self.rs1(x3_rf, guidance1)
        x3_gra_feat, _, x3_res = self.gra3(x3_rf, rg1)

        guidance2 = F.interpolate(x3_res, size=x2_rf.shape[2:], mode='bilinear', align_corners=False)
        rg2 = self.rs2(x2_rf, guidance2)
        x2_gra_feat, _, x2_res = self.gra2(x2_rf, rg2)

        guidance3 = F.interpolate(x2_res, size=x1_rf.shape[2:], mode='bilinear', align_corners=False)
        rg3 = self.rs3(x1_rf, guidance3)
        x1_gra_feat, _, x1_res = self.gra1(x1_rf, rg3)

        # 将多级输出统一上采样到输入图像尺寸。
        x1_res = F.interpolate(x1_res, size=input_size, mode='bilinear', align_corners=False)
        x2_res = F.interpolate(x2_res, size=input_size, mode='bilinear', align_corners=False)
        x3_res = F.interpolate(x3_res, size=input_size, mode='bilinear', align_corners=False)
        init_map = F.interpolate(init_map, size=input_size, mode='bilinear', align_corners=False)

        # 返回多级预测图，训练阶段用于深监督损失计算。
        return x1_res, x2_res, x3_res, init_map
