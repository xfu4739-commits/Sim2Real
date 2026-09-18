import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import tifffile as tif

from dataset.common import load_environment, resolve_scene_roots
from dataset.satellite_fusion import (
    fuse_satellite_sequence,
    load_scene_base_bgr,
    parse_satellite_fusion_cfg,
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def _parse_sim_split_cfg(cfg):
    split_cfg = cfg.get('sim_split') or {}
    return {
        'enabled': bool(split_cfg.get('enabled', False)),
        'val_ratio': float(split_cfg.get('val_ratio', 0.2)),
        'seed': int(split_cfg.get('seed', 0)),
    }


def _collect_sim_event_keys(root_dirs, fire_dir):
    event_keys = []
    for scene_root in root_dirs:
        fire_root = os.path.join(scene_root, fire_dir)
        if not os.path.isdir(fire_root):
            continue
        for sub_id in sorted(os.listdir(fire_root)):
            sub_path = os.path.join(fire_root, sub_id)
            if os.path.isdir(sub_path):
                event_keys.append(f'{scene_root}_{sub_id}')
    return event_keys


def _select_sim_event_keys(event_keys, split_cfg, training):
    if not split_cfg['enabled']:
        return set(event_keys)

    unique_keys = sorted(set(event_keys))
    if not unique_keys:
        return set()

    rng = random.Random(split_cfg['seed'])
    shuffled = unique_keys.copy()
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        val_keys = set() if training else set(shuffled)
        train_keys = set(shuffled) if training else set()
        return train_keys if training else val_keys

    val_count = max(1, int(round(len(shuffled) * split_cfg['val_ratio'])))
    val_count = min(val_count, len(shuffled) - 1)
    val_keys = set(shuffled[:val_count])
    train_keys = set(shuffled[val_count:])
    return train_keys if training else val_keys


class SimFireDataset(Dataset):

    def __init__(self, cfg, training=True):
        self.sample = cfg['input_length'] * 2
        self.training = training
        self.split_cfg = _parse_sim_split_cfg(cfg)
        self.root_dir = resolve_scene_roots(
            cfg['sim_root_dir'],
            [cfg['train_dir'], cfg['val_dir'], cfg['trainwxs_dir'],
             cfg['valwxs_dir'], cfg['topo_dir'], cfg['vege_dir'], cfg['fuel_dir']],
        )
        if self.split_cfg['enabled']:
            self.fire_dir = cfg['train_dir']
            self.wxs_dir = cfg['trainwxs_dir']
        else:
            self.fire_dir = cfg['train_dir'] if training else cfg['val_dir']
            self.wxs_dir = cfg['trainwxs_dir'] if training else cfg['valwxs_dir']
        self.topo_dir = cfg['topo_dir']
        self.vege_dir = cfg['vege_dir']
        self.fuel_dir = cfg['fuel_dir']
        self.satellite_file = cfg['sim_satellite_file']
        self.reverse = cfg['reverse']
        self.num_workers = cfg['num_workers']
        self.topo = {}
        self.vege = {}
        self.fuel = {}
        self.img_size = cfg['img_size']
        self.fusion_cfg = parse_satellite_fusion_cfg(cfg)
        self.satellite = {}
        self.satellite_base = {}

        valid_root_dir = []
        for rd in self.root_dir:
            satellite_path = os.path.join(rd, self.satellite_file)
            if not os.path.isfile(satellite_path):
                print(f"Warning: skipping scene with missing satellite image: {satellite_path}")
                continue
            if cv2.imread(satellite_path) is None:
                print(f"Warning: skipping scene with unreadable satellite image: {satellite_path}")
                continue

            topo, vege, fuel = load_environment(
                rd, self.topo_dir, self.vege_dir, self.fuel_dir, self.img_size
            )
            self.topo[f'{rd}'] = topo
            self.vege[f'{rd}'] = vege
            self.fuel[f'{rd}'] = fuel
            if self.fusion_cfg.enabled:
                self.satellite_base[f'{rd}'] = load_scene_base_bgr(
                    satellite_path, self.img_size, self.fusion_cfg
                )
            else:
                self.satellite[f'{rd}'] = self.process_image(satellite_path, gray=False).float()
            valid_root_dir.append(rd)

        self.root_dir = valid_root_dir
        if not self.root_dir:
            raise FileNotFoundError(
                "no valid simulation scenes found; each scene needs a readable "
                f"{self.satellite_file}"
            )

        self.fire_path = []
        self.target_path = []
        self.sate_path = []
        self.fire_name = []
        self.wxs_value = []
        self.time_steps = []

        allowed_events = _select_sim_event_keys(
            _collect_sim_event_keys(self.root_dir, self.fire_dir),
            self.split_cfg,
            training,
        )
        wxs_dict = self.read_wxs(self.root_dir, self.wxs_dir)

        for rd in self.root_dir:
            fire_root = os.path.join(rd, self.fire_dir)
            for sub_id in sorted(os.listdir(fire_root)):
                root = os.path.join(fire_root, sub_id)
                if not os.path.isdir(root):
                    continue
                event_key = f'{rd}_{sub_id}'
                if event_key not in allowed_events:
                    continue

                files = [
                    name for name in os.listdir(root)
                    if name.lower().endswith('.jpg')
                ]
                length = len(files)
                if length <= 0:
                    continue

                sequence = self.extract_sequences(1, length, self.sample, reverse=self.reverse)
                for s in sequence:
                    time_steps = self.normalize_timestamps(1, length, s)
                    ind = int(len(s) / 2)
                    input_sequence = s[:ind]
                    output_sequence = s[ind:]
                    input_path = []
                    input_wxs = []
                    out_path = []
                    sate_path = []
                    flag = -1
                    for ind, line_sequence in enumerate(input_sequence):
                        f = os.path.join(root, 'out' + str(line_sequence) + '.jpg')
                        o = os.path.join(root, 'out' + str(output_sequence[ind]) + '.jpg')
                        key = f'{rd}_{sub_id}_{line_sequence}'
                        if wxs_dict[key][0] == '#':
                            flag += 1
                        input_wxs.append(wxs_dict[key])
                        input_path.append(f)
                        out_path.append(o)

                    if flag == -1:
                        self.time_steps.append(time_steps)
                        self.fire_path.append(input_path)
                        self.target_path.append(out_path)
                        self.fire_name.append(f'{rd}')
                        self.wxs_value.append(input_wxs)
                        self.sate_path.append([])
                    flag = -1

        split_name = 'train' if training else 'val'
        if self.split_cfg['enabled']:
            print(
                f"SimFireDataset [{split_name}]: "
                f"{len(allowed_events)} events, {len(self.fire_path)} samples "
                f"(val_ratio={self.split_cfg['val_ratio']}, seed={self.split_cfg['seed']})"
            )
        else:
            print(f"SimFireDataset [{split_name}]: {len(self.fire_path)} samples (no event split)")

    def __len__(self):
        return len(self.fire_path)

    def __getitem__(self, index): 
        fire_path = self.fire_path[index]       # 输入火场序列各时刻的图像路径列表
        target_path = self.target_path[index]   # 目标火场序列各时刻的图像路径列表
        fire_name = self.fire_name[index]       # 场景根目录名，用于索引该场景的静态环境数据
        time_steps = self.time_steps[index]     # 目标时刻的归一化时间戳（相对序列起止时间）
        topo = self.topo[fire_name]
        vege = self.vege[fire_name]
        fuel = self.fuel[fire_name]
        
        wxs_value = self.wxs_value[index]
        for i, wxs in enumerate(wxs_value):
            wxs_value[i] = wxs
            for w, j in enumerate(wxs_value[i]):
                wxs_value[i][w] = float(j)

        wxs = torch.tensor(wxs_value)
        
        input_squence = [self.process_image(file, gray=True) for file in fire_path]
        input_squence = torch.cat(input_squence, dim=0).float()

        output_squence = [self.process_image(file, gray=True) for file in target_path]
        output_squence = torch.cat(output_squence, dim=0).float()

        if self.fusion_cfg.enabled:
            # One base map per scene; fuse it with each input fire mask in fire_path order.
            sate_squence = fuse_satellite_sequence(
                self.satellite_base[fire_name],
                fire_path,
                self.img_size,
                self.fusion_cfg,
                sample_seed=index,
            )
        else:
            sate_squence = self.satellite[fire_name].unsqueeze(0).repeat(
                input_squence.shape[0], 1, 1, 1
            )

        return index, input_squence, output_squence, fuel, vege, topo, sate_squence, wxs, torch.tensor(time_steps)

    def process_image(self, file_path, gray=True):
        ''' 图像预处理函数，负责把磁盘上的 JPG 读进来，转成模型可用的 PyTorch 张量
        gray=True 火场输入输出
        gray=False 卫星图像 保留三通道
        '''
        img = cv2.imread(file_path) # 用 OpenCV 读取，默认是 BGR 三通道，uint8，值域 [0, 255]
        if gray:
            # （H,W,3) 到 (H,W)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换颜色，从BGR道GRAY，返回新的numpy数组
            img[np.where(img > 0)] = 1 # 二值化，非零值设为 1
            # 缩放图像 → 转成 PyTorch 张量 → 增加通道维度
            # interpolation=cv2.INTER_NEAREST 最近邻插值，新像素取决于最近像素值不做加权平均
            img = torch.from_numpy(cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)).unsqueeze(0) # 在第0维增加一个通道维度 
        else:
            # 缩放图像 → 转成 PyTorch 张量 → 调整通道顺序
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def get_landfire_modality(self, root_dir, landfire_dir, means, stds, apply_sin_indices=[], indices=[], one_hot_matrices={}):
        file_names = os.listdir(os.path.join(root_dir, landfire_dir))
        modalities = []
        
        for i, file in enumerate(file_names):
            apply_sin = i in apply_sin_indices
            modalities.append(self.read_and_process_landfire_file(os.path.join(root_dir, landfire_dir), file, means, stds, apply_sin))
        
        modalities = torch.cat(modalities, dim=0).float()

        if indices:
            modalities = self.process_land_cover(modalities, indices, one_hot_matrices)

        return modalities

    def read_and_process_landfire_file(self, path, file, means, stds, apply_sin=False):
        file_path = os.path.join(path, file)
        try:
            lf = tif.imread(file_path)
            lf = cv2.resize(lf, self.img_size, interpolation=cv2.INTER_NEAREST)
            lf = torch.from_numpy(lf).unsqueeze(0).float()

            if apply_sin:
                lf = torch.sin(torch.deg2rad(lf))

            for i in range(lf.shape[0]): 
                lf[i] = (lf[i] - means[i]) / stds[i]

            return lf
        except IOError:
            print(f"Open ERROR!: {file_path}")
            return torch.zeros((1, *self.img_size))

    def process_land_cover(self, lf, indices, one_hot_matrices):
        one_hot_encoded_tensors = []
        for i, index in enumerate(indices):
            new_shape = (lf.shape[1], lf.shape[2], one_hot_matrices[i].shape[0])
            landcover_classes_flattened = lf[index, ...].long().flatten() - 1
            landcover_encoding = one_hot_matrices[i][landcover_classes_flattened].reshape(new_shape).permute(2, 0, 1)
            one_hot_encoded_tensors.append(landcover_encoding)
        all_one_hot_encoded = torch.cat(one_hot_encoded_tensors, dim=0)
        remaining_features = lf[len(indices):, ...] if len(indices) < lf.shape[0] else None
        lf = torch.cat([all_one_hot_encoded, remaining_features], dim=0)
        return lf

    def extract_sequences(self, start, end, sample, u_interval=0.167, reverse=True):
        if end - start <= 0:
            return []
        sequence = []
        num_samples = max(1, int((end - 1) - np.floor(end * u_interval)))
        
        for _ in range(num_samples):
            seq = self.extract_sequences_interval_random(start, end, sample, reverse)
            if seq:
                sequence.append(seq)
        return sequence

    def extract_sequences_interval_random(self, start, end, sample, reverse):
        max_interval = int(np.floor((end - start) * 0.167))
        intervals = np.random.randint(1, max_interval + 1, size=sample - 1)
        intervals = np.insert(intervals, 0, 0).cumsum()
        if intervals[-1] > (end - start):
            return None
        if reverse:
            seq = [end - i for i in intervals[::-1]]
        else:
            seq = [start + i for i in intervals]

        return seq if all(1 <= x <= end for x in seq) else None

    def read_wxs(self, root_dir, wxs_path):
        file_lines_dict = {}
        for rd in root_dir:
            wxs_dir = os.path.join(rd , wxs_path)
            try:
                for filename in os.listdir(wxs_dir):
                    file_path = os.path.join(wxs_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line_number, line_content in enumerate(lines, 1):
                            key = f"{rd}_{filename[:-4]}_{line_number}"
                            file_lines_dict[key] = line_content.strip().split(' ')
            except Exception as e:
                print(f"Error reading {wxs_dir}: {e}")
        return file_lines_dict

    def normalize_timestamps(self, start_time, end_time, timestamps):
        start_time = float(start_time)
        end_time = float(end_time)
        timestamps = np.array(timestamps, dtype=float)
        normalized_timestamps = (timestamps - start_time) / (end_time - start_time)
        normalized_timestamps = np.clip(normalized_timestamps, 0, 1)
        return normalized_timestamps[3:].tolist()


    def compute_means_stds(self, modality_dir, indices_to_exclude):
        all_data = []
        for rd in self.root_dir:
            for root, dirs, files in os.walk(os.path.join(rd, modality_dir)):
                for file in files:
                    if file.endswith(".tif"):
                        file_path = os.path.join(root, file)
                        try:
                            lf = tif.imread(file_path)
                            lf = cv2.resize(lf, self.img_size, interpolation=cv2.INTER_NEAREST)
                            all_data.append(lf)
                        except IOError:
                            print(f"Open ERROR!: {file_path}")

        all_data = np.stack(all_data, axis=0)
        means = np.mean(all_data, axis=(1, 2))
        stds = np.std(all_data, axis=(1, 2))
        means = np.asarray(means)
        stds = np.asarray(stds)
        means[indices_to_exclude] = 0
        stds[indices_to_exclude] = 1

        return torch.tensor(means), torch.tensor(stds)
