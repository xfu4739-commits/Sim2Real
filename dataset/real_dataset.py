import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime

from dataset.common import (
    EnvironmentRasterCache,
    filter_complete_environment_scenes,
    resolve_scene_roots,
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def image_date_token(filename):
    """Return YYYYMMDD from names like 20170718.png or fire_20170718.png."""
    stem = os.path.splitext(filename)[0]
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


class RealFireDataset(Dataset):

    def __init__(self, cfg):
        self.sample = cfg['input_length'] * 2
        self.root_dir = resolve_scene_roots(
            cfg['real_root_dir'],
            [cfg['test_dir'], cfg['testwxs_dir'], cfg['real_satellite_dir'],
             cfg['topo_dir'], cfg['vege_dir'], cfg['fuel_dir']],
        )
        self.fire_dir = cfg['test_dir']
        self.wxs_dir = cfg['testwxs_dir']
        self.topo_dir = cfg['topo_dir']
        self.vege_dir = cfg['vege_dir']
        self.fuel_dir = cfg['fuel_dir']
        self.sate_dir = cfg['real_satellite_dir']
        self.reverse = cfg['reverse']
        self.num_workers = cfg['num_workers']
        self.img_size = cfg['img_size']
        self.root_dir = filter_complete_environment_scenes(
            self.root_dir, self.topo_dir, self.vege_dir, self.fuel_dir
        )
        self.environment = EnvironmentRasterCache(
            self.topo_dir,
            self.vege_dir,
            self.fuel_dir,
            self.img_size,
            max_scenes=cfg.get('environment_cache_size', 8),
        )

        self.fire_path = []
        self.target_path = []
        self.sate_path = []
        self.fire_name = []
        self.wxs_value = []
        self.time_steps = []

        wxs_dict = self.read_wxs(self.root_dir, self.wxs_dir)     
        
        for rd in self.root_dir:
            for root, dirs, files in os.walk(os.path.join(rd, self.fire_dir)):
                if root == os.path.join(rd, self.fire_dir):
                    continue
                else:
                    image_files = [
                        name for name in files
                        if name.lower().endswith(IMAGE_SUFFIXES)
                    ]
                    sorted_files = sorted(
                        image_files,
                        key=lambda name: datetime.strptime(image_date_token(name), '%Y%m%d'),
                    )
                    length = len(sorted_files)
                    sequences = self.extract_sequences(length, self.sample)
                    for s in sequences:
                        timestamps = [
                            datetime.strptime(image_date_token(sorted_files[i]), '%Y%m%d')
                            for i in s
                        ]
                        time_steps = self.normalize_timestamps(timestamps)
                        ind = int(len(s) / 2)
                        input_sequence = s[:ind]
                        output_sequence = s[ind:]
                        input_path = []
                        input_wxs = []
                        out_path = []
                        sate_path = []
                        valid_weather = True
                        for ind, line_sequence in enumerate(input_sequence):
                            f = os.path.join(root, sorted_files[line_sequence])
                            o = os.path.join(root, sorted_files[output_sequence[ind]])
                            s_path = os.path.join(rd, self.sate_dir, os.path.basename(root), sorted_files[line_sequence])                                      
                            date_token = image_date_token(sorted_files[line_sequence])
                            key = f'{rd}_{os.path.basename(root)}_{date_token}'
                            if key not in wxs_dict:
                                valid_weather = False
                                break
                            input_wxs.append(wxs_dict[key])
                            input_path.append(f) 
                            out_path.append(o)
                            sate_path.append(s_path)
                        
                        if not valid_weather:
                            continue

                        self.time_steps.append(time_steps)
                        self.fire_path.append(input_path) 
                        self.target_path.append(out_path)
                        self.fire_name.append(f'{rd}')
                        self.wxs_value.append(input_wxs)
                        self.sate_path.append(sate_path)
                        

    def __len__(self):
        return len(self.fire_path)

    def __getitem__(self, index): 
        fire_path = self.fire_path[index]
        target_path = self.target_path[index]
        sate_path = self.sate_path[index]
        fire_name = self.fire_name[index]
        time_steps = self.time_steps[index]
        topo, vege, fuel = self.environment.get(fire_name)
        
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

        sate_squence = [self.process_image(file, gray=False) for file in sate_path]
        sate_squence = torch.stack(sate_squence, dim=0).float()

        return index, input_squence, output_squence, fuel, vege, topo, sate_squence, wxs, torch.tensor(time_steps)

    def process_image(self, file_path, gray=True):
        img = cv2.imread(file_path)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img[np.where(img > 0)] = 1
            img = torch.from_numpy(
                cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
            ).unsqueeze(0)
        else:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def extract_sequences(self, end, sample):
        sequences = []
        if self.reverse:
            for i in range(end - sample + 1):
                sequences.append(list(range(end - i - 1, end - i - sample - 1, -1)))
        else:
            for i in range(end - sample + 1):
                sequences.append(list(range(i, i + sample)))
        return sequences

    def read_wxs(self, root_dir, wxs_path):
        file_lines_dict = {}
        for rd in root_dir:
            wxs_dir = os.path.join(rd , wxs_path)
            try:
                for filename in os.listdir(wxs_dir):
                    file_path = os.path.join(wxs_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        event_name = os.path.splitext(filename)[0]
                        for line_content in lines:
                            values = line_content.strip().split()
                            if len(values) < 3:
                                continue
                            date_token = f"{int(values[0]):04d}{int(values[1]):02d}{int(values[2]):02d}"
                            key = f"{rd}_{event_name}_{date_token}"
                            file_lines_dict[key] = values
            except Exception as e:
                print(f"Error reading {wxs_dir}: {e}")
        return file_lines_dict

    def normalize_timestamps(self, timestamps):
        # Normalize against chronological bounds rather than sequence order. This
        # preserves direction: forecast timestamps increase, while backtracking
        # timestamps decrease. Using the first/last sequence items as the bounds
        # made both directions increase from 0 to 1 and erased that distinction.
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds()
        if duration == 0:
            normalized_timestamps = [0.0 for _ in timestamps]
        else:
            normalized_timestamps = [
                (timestamp - start_time).total_seconds() / duration
                for timestamp in timestamps
            ]
        return normalized_timestamps[3:]

