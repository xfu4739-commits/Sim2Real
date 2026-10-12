import os
import argparse
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from model.S2RFireTr import FireTr
from dataset.real_dataset import RealFireDataset
from lib.dice_loss import BinaryDiceLoss
from lib.focal_loss import BinaryFocalLoss
from lib.losses import combined_losses
from metrics.metrics import BinaryAUPRCHistogram, SegmentationMetric
from utils import setup_seed, load_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    return device

def load_data(cfg):
    test_dataset = RealFireDataset(cfg['dataset'])
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    return test_dataloader

def initialize_model(cfg, device):
    if not cfg['checkpoint']['checkpoint']:
        raise ValueError(
            'Refusing to evaluate a randomly initialized model: '
            'set checkpoint.checkpoint=true and provide checkpoint_path.'
        )
    checkpoint_path = cfg['checkpoint']['checkpoint_path']
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    model = FireTr(cfg['FireTr']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    print(f'Loaded model from {checkpoint_path}')
    return model

def test(dataloader, model, device, vis_dir, loss_dice, loss_focal):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    metric = SegmentationMetric(2)
    auprc = BinaryAUPRCHistogram(device)

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            (index, input_firesquence, output_firesquence, fuel, vegetation,
             topography, satellite_images, weather_data, time_steps) = data
            inputs = input_firesquence.to(device)
            targets = output_firesquence.to(device)
            fuel = fuel.to(device)
            vegetation = vegetation.to(device)
            topography = topography.to(device)
            satellite_images = satellite_images.to(device)
            weather_data = weather_data.to(device)
            time_steps = time_steps.to(device)
            pred = model(
                inputs, fuel, vegetation, topography,
                satellite_images, weather_data, time_steps
            )
            loss, _, _ = combined_losses(pred, targets, loss_dice, loss_focal)
            loss_sum += loss.item() * inputs.shape[0]
            sample_count += inputs.shape[0]

            pred_sigmoid = torch.sigmoid(pred)
            auprc.update(pred_sigmoid, targets)
            pred_binary = torch.where(pred_sigmoid > 0.5, torch.ones_like(pred_sigmoid), torch.zeros_like(pred_sigmoid))
            pred_sigmoid, pred_binary, targets = pred_sigmoid.cpu().numpy(), pred_binary.cpu().numpy(), targets.cpu().numpy()
            pred_sigmoid, pred_binary, targets = pred_sigmoid.astype(np.float32), pred_binary.astype(np.int32), targets.astype(np.int32)
            metric.addBatch(pred_binary, targets)
            
            b, t, h, w = pred_binary.shape
            inputs_array = inputs.cpu().numpy()
            
            for i in range(b):
                for j in range(t):
                    name = f'{index[i].item()}_gt_{j+1}.png'
                    file_name = os.path.join(vis_dir, name)
                    img_gt = np.uint8(inputs_array[i, j, :, :] * 255)
                    cv2.imwrite(file_name, img_gt)
                    
                    name = f'{index[i].item()}_gt_{j+1+t}.png'
                    file_name = os.path.join(vis_dir, name)
                    img_gt = np.uint8(targets[i, j, :, :] * 255)
                    cv2.imwrite(file_name, img_gt)
                    
                    name = f'{index[i].item()}_pd_{j+1+t}.png'
                    file_name = os.path.join(vis_dir, name)
                    img_pd = pred_binary[i, j, :, :]
                    img_pd = np.maximum(img_pd, 0)
                    img_pd = np.minimum(img_pd, 1)
                    img_pd = np.uint8(img_pd * 255)
                    cv2.imwrite(file_name, img_pd)
            
    mean_epoch_loss = loss_sum / sample_count if sample_count else 0.0
    mean_iou = float(metric.IntersectionOverUnion()[1])
    mean_f1 = float(metric.f1())
    mean_auprc = float(auprc.compute().item())

    print('val_loss: ' + str(mean_epoch_loss))
    print('mean_iou: ' + str(mean_iou))
    print('mean_f1: ' + str(mean_f1))
    print('mean_auprc: ' + str(mean_auprc))

    return mean_iou, mean_f1, mean_auprc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    setup_seed(12345)
    cfg = load_config(args.config)
    device = setup_device()
    model = initialize_model(cfg, device)
    test_dataloader = load_data(cfg)
    vis_dir = cfg['dataset']['vis_dir']
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    loss_dice = BinaryDiceLoss()
    loss_focal = BinaryFocalLoss()
          
    iou, f1, auprc = test(test_dataloader, model, device, vis_dir, loss_dice, loss_focal)

if __name__ == '__main__':
    main()
