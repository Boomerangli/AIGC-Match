import argparse
import logging
import os
import pprint
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
from dataset.acdc import ACDCDataset
from model.unet import UNet
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.dist_helper import setup_distributed
from util import *


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


colors = np.array([
    [0, 0, 0],    # 黑色
    [255, 0, 0],    # 红色
    [0, 255, 0],    # 绿色,比较这个效果最佳
    [0, 0, 255]     # 蓝色
])

def visualize_and_save(img, mask, pred_mask, save_path):
    """
    可视化图像、真实标签和预测标签并保存到指定路径。
    
    Args:
        img (numpy.ndarray): 原始图像，形状为 (H, W)。
        mask (numpy.ndarray): 真实标签，形状为 (H, W)。
        pred_mask (numpy.ndarray): 预测标签，形状为 (H, W)。
        save_path (str): 保存路径。
    """
    # 定义类别颜色映射
    colors = np.array([
        [0, 0, 0],    # 黑色 (背景)
        [255, 0, 0],  # 红色 (类别 1)
        [0, 255, 0],  # 绿色 (类别 2)
        [0, 0, 255],  # 蓝色 (类别 3)
    ])

    # 创建一个 RGB 图像用于分割结果
    pred_color = colors[pred_mask]  # 将预测结果映射为 RGB 颜色

    # 将原始灰度图转换为 RGB 图像
    img = (img * 255).astype('uint8')  # 将原始图像归一化到 [0, 255] 范围
    img = np.stack([img, img, img], axis=-1)  # 灰度图转为 RGB 图

    # 创建一个叠加图像，仅对分割区域叠加颜色
    overlay = img.copy()
    alpha = 1  # 调整透明度
    mask_indices = pred_mask > 0  # 分割区域索引
    overlay[mask_indices] = (img[mask_indices] * (1 - alpha) + pred_color[mask_indices] * alpha).astype('uint8')

    # 显示真实标签的颜色
    mask_color = colors[mask]
    truth = img.copy()
    truth[mask > 0] = (img[mask > 0] * (1 - alpha) + colors[mask[mask > 0]] * alpha).astype('uint8')
    
    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(truth)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Unimatch")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet(in_chns=1, class_num=cfg['nclass'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    local_rank = int(os.environ["LOCAL_RANK"]) 
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
    state_dict = torch.load(os.path.join('Unimatch.pth'))
    model.load_state_dict(state_dict['model'])
    
    valset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=None)

    model.eval()
    for i, (img, mask) in enumerate(valloader):
        img, mask = img.cuda(), mask.cuda()
        h, w = img.shape[-2:]
        img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)
        img = img.permute(1, 0, 2, 3)
        #print(img.shape)
        #print(mask.shape)
        pred = model(img)
        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
        pred = pred.argmax(dim=1).cpu().numpy().astype('uint8')
        
        img = F.interpolate(img, (h, w), mode='bilinear', align_corners=False)
        #print(pred.shape)
        mask = mask.squeeze()
        for j in range(pred.shape[0]):
            visualize_and_save(
                img[j].squeeze(0).squeeze(0).cpu().float().numpy(),  # 位置参数1
                mask[j].squeeze(0).squeeze(0).cpu().numpy(),  # 位置参数2
                pred[j],  # 位置参数3
                os.path.join('result', f'vis_{i:03d}+{j:03d}.png')  # 位置参数4
            )


if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')
    main()
