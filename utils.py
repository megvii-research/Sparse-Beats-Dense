import functools
import time

import scipy
import numpy as np
import torch

from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion
from matplotlib import cm

def tensor_to_frame(output_dict):

    def brighten(inp_srgb):
        return np.clip((100/inp_srgb.mean())*inp_srgb,0,255).astype(np.uint8)

    frame_dict = {}
    for output_type, output_frame in output_dict.items():
        if output_frame is None:
            continue
        if isinstance(output_frame, torch.Tensor):
            output_frame = output_frame.cpu().numpy()
        elif isinstance(output_frame, np.ndarray):
            pass
        if output_type in [
            "img", 
        ]:
            img = output_frame[0,...].transpose((1,2,0))
            img = np.clip(img, 0, 255).astype(np.uint8)
            frame_dict[output_type] = img
    
        elif output_type in [
            'pred', 
            "label",
            "radar",
        ]:
            img = output_frame.squeeze()
            img = np.clip(img, 0, 80).astype(np.uint8)            
            frame_dict[output_type] = colorize_depth_map(img/80)
        elif output_type in [
            'pred_mask', 
            "label_mask",
            "radar_mask",
            "valid_label",
            "valid_label_mask",
        ]:
            img = output_frame.squeeze()
            frame_dict[output_type] = img

    return frame_dict


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """From vod.frame without rounding to int"""

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    # uvs = np.round(uvs).astype(np.int)

    return uvs


def map_pointcloud1_to_pointcloud2(
    lidar_points,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):
    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    
    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)
    
    points = lidar_points.points.transpose((1, 0))
    return points


def map_pointcloud_to_image(
    lidar_points,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):
    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    points = map_pointcloud1_to_pointcloud2(lidar_points, lidar_calibrated_sensor, lidar_ego_pose,
                                            cam_calibrated_sensor, cam_ego_pose, min_dist)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    uvs = project_3d_to_2d(points[:, :3], np.array(cam_calibrated_sensor['camera_intrinsic']))

    return points, np.concatenate((uvs, points[:, 2:3]), 1)


def canvas_filter(data, shape):
    return np.all((data >= 0) & (data < shape[1::-1]), 1)


def _scale_pts(data, out_shape, input_shape):
    data[:, :2] *= (np.array(out_shape[::-1]) / input_shape[1::-1])
    return data


def get_depth_map(data, shape, input_shape=None):
    if input_shape is not None:
        data = _scale_pts(data.copy(), shape, input_shape)

    depth = np.zeros(shape + (data.shape[1] - 2, ), dtype=np.float32)
    if np.any(data[:, :2].max(0) >= shape[1::-1]) or data[:, :2].min() < 0:
        inds = canvas_filter(data[:, :2], shape)
        data = data[inds]
    depth[data[:, 1].astype(int), data[:, 0].astype(int)] = data[:, 2:]
    return depth.squeeze()


def get_radar_vert_map(radar, out_shape, input_shape=None):
    if input_shape is not None:
        radar = _scale_pts(radar.copy(), out_shape, input_shape)

    radar_map = np.full(out_shape + (9, ), 10000, dtype=np.float32)
    radar_map[radar[:, 1].astype(int), radar[:, 0].astype(int)] = radar[:, 3:]
    radar_map = scipy.ndimage.minimum_filter1d(radar_map, 3, 1)
    radar_map[radar_map == 10000] = 0
    return radar_map


def get_radar_map(data, shape, input_shape=None):
    if input_shape is not None:
        data = _scale_pts(data.copy(), shape, input_shape)

    depth = np.zeros(shape + (data.shape[1] - 2, ), dtype=np.float32)
    if np.any(data[:, :2].max(0) >= shape[1::-1]) or data[:, :2].min() < 0:
        inds = canvas_filter(data[:, :2], shape)
        data = data[inds]
    depth[:, data[:, 0].astype(int)] = data[:, 2:]
    return depth.squeeze()


def extend_height(cam_depth, camera_intrinsic, origin_dims, h0=0.25, h1=1.5):    
    camera_intrinsic = camera_intrinsic 
    H, W = origin_dims
    def getRelSize(camera_intrinsic, d, w=0.5, h=1.5):
        v = (h*camera_intrinsic[0][0])/d
        u = (w*camera_intrinsic[1][1])/d    
        return u, v #int(u),int(v)
    ret = cam_depth.copy()
    for depth in cam_depth:
        x,y,d = depth 
        _,v1 = getRelSize(camera_intrinsic, d, 0, h1)
        _,v0 = getRelSize(camera_intrinsic, d, 0, h0)
        y_list = np.arange(start=max(y-v0,0),stop=min(y+v1,H),step=1)
        ptsnum_after_extend = len(y_list)
        x = np.stack((np.array([x]*ptsnum_after_extend),y_list,np.array([d]*ptsnum_after_extend)),axis=1)
        ret = np.concatenate((ret,x),axis=0)
    return ret


def colorize_depth_map(data, mask=None, norm=False):
    if mask is None:
        mask = data > 0
    elif mask.dtype != bool:
        mask = (mask > 0).astype(bool)

    # data = np.exp(-data / 72.136)
    if norm:
        min_val = data[mask].min()
        data = (data - min_val) / (data.max() - min_val)
    else:
        data = np.clip(data, 0, 1)

    data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    # data = cv2.applyColorMap(data, cv2.COLORMAP_JET)
    data = cm.jet(data)
    data = (data[..., :3] * 255).astype(np.uint8)

    mask = np.stack([mask] * 3, 2)
    data[~mask] = 0
    return data


def log_rate_limited(min_interval=1):
    def decorator(should_record):
        last = 0

        @functools.wraps(should_record)
        def wrapper(*args, **kwargs):
            nonlocal last
            if time.time() - last < min_interval:
                return False
            ret = should_record(*args, **kwargs)
            last = time.time()
            return ret

        return wrapper

    return decorator


class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {"epoch": self.epoch, "minibatch": self.minibatch, "step": self.step}

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict["epoch"]
        self.minibatch = clock_dict["minibatch"]
        self.step = clock_dict["step"]