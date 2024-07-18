import os
import cv2
import numpy as np
import pickle
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud

from utils import (
    map_pointcloud_to_image,
    get_depth_map,
    get_radar_map,
    canvas_filter,
)


class conf:
    input_h, input_w = 900, 1600
    max_depth = 80
    min_depth = 0


rng = np.random.default_rng()


class Vidar(torch.utils.data.Dataset):
    path = './data/nuscenes_radar_5sweeps_infos_train.pkl'
    
    data_root = './data/nuscenes/samples/'
    semantic_root = './data/nuscenes/seg_mask/'
    
    def __init__(self):
        with open(self.path, 'rb') as f:
            self.infos = pickle.loads(f.read())
        
        self.radar_load_dim = 18 # self.radar_data_conf["radar_load_dim"]
        self.radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17] # [x y z] dyn_prop id [rcs vx vy vx_comp vy_comp] is_quality_valid ambig_state [x_rms y_rms] invalid_state pdh0 [vx_rms vy_rms] + [timestamp_diff]
        
        self.semantic_mask_used_mask = [0, 1, 4, 12, 20, 32, 80, 83, 93, 127, 102, 116] 
        # {"wall": 0, "building": 1, "sky": 2, "floor": 3, "tree": 4, "ceiling": 5, "road": 6, "bed ": 7, "windowpane": 8, "grass": 9, "cabinet": 10, "sidewalk": 11, "person": 12, "earth": 13, "door": 14, "table": 15,
        # "mountain": 16, "plant": 17, "curtain": 18, "chair": 19, "car": 20, "water": 21, "painting": 22, "sofa": 23, "shelf": 24, "house": 25, "sea": 26, "mirror": 27, "rug": 28, "field": 29, "armchair": 30, "seat": 31, 
        # "fence": 32, "desk": 33, "rock": 34, "wardrobe": 35, "lamp": 36, "bathtub": 37, "railing": 38, "cushion": 39, "base": 40, "box": 41, "column": 42, "signboard": 43, "chest of drawers": 44, "counter": 45, "sand": 46,
        # "sink": 47, "skyscraper": 48, "fireplace": 49, "refrigerator": 50, "grandstand": 51, "path": 52, "stairs": 53, "runway": 54, "case": 55, "pool table": 56, "pillow": 57, "screen door": 58, "stairway": 59, "river": 60,
        # "bridge": 61, "bookcase": 62, "blind": 63, "coffee table": 64, "toilet": 65, "flower": 66, "book": 67, "hill": 68, "bench": 69, "countertop": 70, "stove": 71, "palm": 72, "kitchen island": 73, "computer": 74, "swivel chair": 75,
        # "boat": 76, "bar": 77, "arcade machine": 78, "hovel": 79, "bus": 80, "towel": 81, "light": 82, "truck": 83, "tower": 84, "chandelier": 85, "awning": 86, "streetlight": 87, "booth": 88, "television receiver": 89, "airplane": 90, 
        # "dirt track": 91, "apparel": 92, "pole": 93, "land": 94, "bannister": 95, "escalator": 96, "ottoman": 97, "bottle": 98, "buffet": 99, "poster": 100, "stage": 101, "van": 102, "ship": 103, "fountain": 104, "conveyer belt": 105, 
        # "canopy": 106, "washer": 107, "plaything": 108, "swimming pool": 109, "stool": 110, "barrel": 111, "basket": 112, "waterfall": 113, "tent": 114, "bag": 115, "minibike": 116, "cradle": 117, "oven": 118, "ball": 119, "food": 120,
        # "step": 121, "tank": 122, "trade name": 123, "microwave": 124, "pot": 125, "animal": 126, "bicycle": 127, "lake": 128, "dishwasher": 129, "screen": 130, "blanket": 131, "sculpture": 132, "hood": 133, "sconce": 134, "vase": 135,
        # "traffic light": 136, "tray": 137, "ashcan": 138, "fan": 139, "pier": 140, "crt screen": 141, "plate": 142, "monitor": 143, "bulletin board": 144, "shower": 145, "radiator": 146, "glass": 147, "clock": 148, "flag": 149}
        
        self.RADAR_PTS_NUM = 200
        
        # Todo support multi-view Depth Completion
        # Now we follow the previous research, only use the front Camera and Radar
        self.radar_use_type = 'RADAR_FRONT'
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'

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
        
        # get semantic mask of images
        name = camera_filename.split('/')[-1].replace('.jpg', '.png')
        seg_mask_path = os.path.join(self.semantic_root, name)
        seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
        seg_mask_roi = list()
        for i in self.semantic_mask_used_mask:
            seg_mask_roi.append(np.where(seg_mask==i, 1, 0))
        seg_mask_roi = np.sum(np.stack(seg_mask_roi, axis=0), axis=0)
        
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
            
        inds = (lidar[:, 2] > conf.min_depth) & (lidar[:, 2] < conf.max_depth)
        lidar = lidar[inds]
        lidar = get_depth_map(lidar[:, :3], img.shape[:2])
        
        inds = canvas_filter(radar[:, :2], img.shape[:2])
        radar = radar[inds]
        radar = get_radar_map(radar[:, :3], img.shape[:2])
        
        img        = Image.fromarray(img[...,::-1])  # BGR->RGB
        lidar      = Image.fromarray(lidar.astype('float32'), mode='F')
        radar      = Image.fromarray(radar.astype('float32'), mode='F')
        seg_mask_roi   = torch.from_numpy(seg_mask_roi.astype('float32'))[None]
        
        # Aug
        try:
            img, lidar, radar, seg_mask_roi = augmention(img, lidar, radar, seg_mask_roi)
        except:
            pass
        
        lidar, radar = (np.array(d) for d in (lidar, radar))
        
        lidar_mask, radar_mask = (
            (d > 0).astype(np.uint8) for d in (lidar, radar))
        
        lidar, radar = (d[None] for d in (lidar, radar))
        lidar_mask, radar_mask = (
            d[None] for d in (lidar_mask, radar_mask))
        
        img = np.array(img)[...,::-1] # RGB -> BGR
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        return img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar, lidar_mask, seg_mask_roi


def augmention(img:Image, lidar:Image, radar:Image, seg_mask:torch.Tensor):
    width, height = img.size
    _scale = rng.uniform(1.0, 1.3) # resize scale > 1.0, no info loss
    scale  = int(height * _scale)
    degree = np.random.uniform(-5.0, 5.0)
    flip   = rng.uniform(0.0, 1.0)
    # Horizontal flip
    if flip > 0.5:
        img   = TF.hflip(img)
        lidar = TF.hflip(lidar)
        radar = TF.hflip(radar)
        seg_mask   = TF.hflip(seg_mask)
    
    # Color jitter
    brightness = rng.uniform(0.6, 1.4)
    contrast   = rng.uniform(0.6, 1.4)
    saturation = rng.uniform(0.6, 1.4)

    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    
    # Resize
    img        = TF.resize(img,   scale, interpolation=InterpolationMode.BICUBIC)
    lidar      = TF.resize(lidar, scale, interpolation=InterpolationMode.NEAREST)
    radar      = TF.resize(radar, scale, interpolation=InterpolationMode.NEAREST)
    seg_mask   = TF.resize(seg_mask, scale, interpolation=InterpolationMode.NEAREST)

    # Crop
    width, height = img.size
    ch, cw = conf.input_h, conf.input_w
    h_start = rng.integers(0, height - ch)
    w_start = rng.integers(0, width - cw)

    img          = TF.crop(img,   h_start, w_start, ch, cw)
    lidar        = TF.crop(lidar, h_start, w_start, ch, cw)
    radar        = TF.crop(radar, h_start, w_start, ch, cw)
    seg_mask     = TF.crop(seg_mask, h_start, w_start, ch, cw)

    img     = TF.gaussian_blur(img, kernel_size=3, )

    return img, lidar, radar, seg_mask


if __name__ == '__main__':
    from utils import colorize_depth_map
    dataset = Vidar()
    dataset = iter(dataset)
    img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar, lidar_mask, seg_mask_roi = next(dataset)
    radar = np.clip(radar, 0, 80).astype(np.uint8).squeeze()
    radar = colorize_depth_map(radar/80)
    cv2.imwrite('test_Radar.png', radar)