#!/usr/bin/env python3
"""
Get eval result of models.
"""
import os
import pickle
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud

from model import Network
from utils import (
    canvas_filter,
    get_depth_map,
    get_radar_map,
    map_pointcloud_to_image)


class conf:
    datasets = {
        'nuscenes': './data/nuscenes_radar_5sweeps_infos_test.pkl',
    }

    default_dataset = 'nuscenes'
    max_depth = 80
    min_depth = 0


def path2label(path):
    return path.rstrip('/').replace('/', '_')


rng = np.random.default_rng()


class Vidar:
    data_root = './data/nuscenes/samples/'
    
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.infos = pickle.loads(f.read())
            
        # self.infos = self.infos[::10][:512]

        self.radar_load_dim = 18 # self.radar_data_conf["radar_load_dim"]
        self.radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17] # [x y z] dyn_prop id [rcs vx vy vx_comp vy_comp] is_quality_valid ambig_state [x_rms y_rms] invalid_state pdh0 [vx_rms vy_rms] + [timestamp_diff]

        self.RADAR_PTS_NUM = 200

        # Todo support multi-view Depth Completion
        # Now we follow the previous research, only use the front Camera and Radar
        self.radar_use_type = 'RADAR_FRONT'
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        
    def set_model(self, model_path):
        self.model_path = model_path
        self.net = self.load_model(self.model_path, self.device)
        
    @staticmethod
    def load_model(model_path, device):
        checkpoint = torch.load(
            model_path, 
            map_location=device
        )
        
        net = Network().to(device)
        net.load_state_dict(checkpoint['network'])
        
        net.eval()
        return net

    def __len__(self):
        return len(self.infos)

    def get_params(self, data):
        params = dict()
        if 'calibrated_sensor' in data.keys():
            params['sensor2ego'] = data['calibrated_sensor']
        else:
            params['sensor2ego'] = dict()
            params['sensor2ego']['translation'] = data['sensor2ego_translation']
            params['sensor2ego']['rotation'] = data['sensor2ego_rotation']

        if 'ego_pose' in data.keys():
            params['ego2global'] = data['ego_pose']
        else:
            params['ego2global'] = dict()
            params['ego2global']['translation'] = data['ego2global_translation']
            params['ego2global']['rotation'] = data['ego2global_rotation']

        return params

    # 这里做了改动, 直接输出Radar的深度图好了, 是在受不了复杂的逻辑
    def __getitem__(self, index):
        data = self.infos[index]

        # get cameras images only for front 
        camera_infos = data['cam_infos'][self.camera_use_type]
        camera_params = self.get_params(camera_infos)
        camera_filename = camera_infos['filename'].split('samples/')[-1]
        img = cv2.imread(os.path.join(self.data_root, camera_filename))
        
        # get radars only for front
        radar_infos = data['radar_infos'][self.radar_use_type][0]
        radar_params = self.get_params(radar_infos)
        path = radar_infos['data_path'].split('samples/')[-1]
        radar_obj = RadarPointCloud.from_file(os.path.join(self.data_root, path))
        radar_all = radar_obj.points.transpose(1,0)[:, self.radar_use_dims]
        radar = np.concatenate((radar_all[:, :3], np.ones([radar_all.shape[0], 1])), axis=1)
        
        # get lidar top
        lidar_infos = data['lidar_infos'][self.lidar_use_type]
        lidar_params = self.get_params(lidar_infos)
        path = lidar_infos['filename'].split('samples/')[-1]
        lidar_obj = LidarPointCloud.from_file(os.path.join(self.data_root, path))
        lidar = lidar_obj.points.transpose(1,0)[:, :3]
        lidar = np.concatenate((lidar, np.ones([lidar.shape[0], 1])), axis=1)
        
        # project lidar and radar to image coordinates
        lidar_pts, lidar = map_pointcloud_to_image(lidar, lidar_params['sensor2ego'], lidar_params['ego2global'],
                                        camera_params['sensor2ego'], camera_params['ego2global'])
        
        radar_pts, radar = map_pointcloud_to_image(radar, radar_params['sensor2ego'], radar_params['ego2global'],
                                        camera_params['sensor2ego'], camera_params['ego2global'])
        
        
        radar_pts = radar_pts[:, :3]
        valid_radar_pts_cnt = radar_pts.shape[0]
        if valid_radar_pts_cnt <= self.RADAR_PTS_NUM:
            padding_radar_pts = np.zeros((self.RADAR_PTS_NUM, 3), dtype=radar_pts.dtype)
            padding_radar_pts[:valid_radar_pts_cnt,:] = radar_pts
        else:
            random_idx = sorted(rng.choice(range(valid_radar_pts_cnt), size=(self.RADAR_PTS_NUM,), replace=False))
            padding_radar_pts = radar_pts[random_idx,:]
        
        lidar = get_depth_map(lidar[:, :3], img.shape[:2])
        
        inds = canvas_filter(radar[:, :2], img.shape[:2]) 
        radar = radar[inds]
        radar = get_radar_map(radar[:, :3], img.shape[:2])
        
        lidar, radar = (np.array(d) for d in (lidar, radar))
        lidar, radar = (d[None] for d in (lidar, radar))
        img = img.transpose(2, 0, 1)
        
        valid_radar_pts_cnt = np.array(valid_radar_pts_cnt)
        return img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar

    def get_error(self, diffs, mask):
        mae  = np.mean(np.abs(diffs[mask]))
        rmse = np.sqrt(np.mean(diffs[mask]**2))
        return mae, rmse

    def eval(self, model_path):
        if model_path is not None:
            self.set_model(model_path)
            
        errors, errors_50, errors_70 = [], [], []
        rmses, rmses_50, rmses_70 = [], [], []
        for ind in tqdm(range(len(self))):
            
            img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar = self[ind]
            img, radar, padding_radar_pts, valid_radar_pts_cnt = (inp[None] for inp in (img, radar, padding_radar_pts, valid_radar_pts_cnt))
            
            with torch.no_grad():
                pred, _ = self.net.forward_test(img, radar, padding_radar_pts, valid_radar_pts_cnt)

            pred, lidar = ( arr.reshape(-1) for arr in (pred, lidar))
            
            mask1 = (lidar > 0) & (lidar <= 80)
            mask2 = (lidar > 0) & (lidar <= 50)
            mask3 = (lidar > 0) & (lidar <= 70)
            
            diff = pred - lidar
            
            diff80, rmse80 = self.get_error(diff, mask1)
            diff50, rmse50 = self.get_error(diff, mask2)
            diff70, rmse70 = self.get_error(diff, mask3)
            
            errors.append(diff80)
            errors_50.append(diff50)
            errors_70.append(diff70)
            rmses.append(rmse80)
            rmses_50.append(rmse50)
            rmses_70.append(rmse70)
        

        result = {
            'epe:0-80':float(np.mean(errors)),    'rmse:0-80':float(np.mean(rmses)),
            'epe:0-50':float(np.mean(errors_50)), 'rmse:0-50':float(np.mean(rmses_50)),
            'epe:0-70':float(np.mean(errors_70)), 'rmse:0-70':float(np.mean(rmses_70)),}
        tqdm.write(', '.join([
            '{}: {:.5}'.format(k, v) for k, v in result.items()]))

        return result

    @staticmethod
    def proj_2d_to_3d(pts, K):
        d = pts[:, 2:]
        pts = np.concatenate((pts[:, :2] * d, d), 1)
        return pts @ np.linalg.inv(K).T

    def proj_2d_map_to_3d(self, pred, mask, K):
        y, x = np.where(mask)
        pts = np.stack((x, y, pred[mask]), 1)
        return self.proj_2d_to_3d(pts, K)


def main():
    parser = argparse.ArgumentParser(
        description='Get eval result for models.')
    parser.add_argument(
        '-d', '--dataset', default=conf.default_dataset,
        help='Dataset name or lovelive dataset id.')
    parser.add_argument(
        '-m', '--model', type=str)

    args = parser.parse_args()

    dataset = Vidar(conf.datasets[args.dataset])
    dataset.eval(args.model)

if __name__ == '__main__':
    main()
