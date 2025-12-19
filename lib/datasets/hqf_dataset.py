import itertools
import os

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

import pdb

video_map = {
        'bike_bay_hdr': 0,
        'boxes': 1,
        'desk': 2,
        'desk_fast': 3,
        'desk_hand_only': 4,
        'desk_slow': 5,
        'engineering_posters': 6,
        'high_texture_plants': 7,
        'poster_pillar_1': 8,
        'poster_pillar_2': 9,
        'reflective_materials': 10,
        'slow_and_fast_desk': 11,
        'slow_hand': 12,
        'still_life': 13
        }

class HQFDataset(Dataset):
    def __init__(self,
                 hdf5_name='/home/yj/Img_Rebuild/DataSet/DeblurSR/HQF/blurry_hdf5/bike_bay_hdr.hdf5'):
        super(HQFDataset, self).__init__()
        self.data = h5py.File(hdf5_name, 'r')
        self.video_idx = video_map[os.path.basename(hdf5_name[:-5])]

    def __len__(self):
        return len(self.data['blurry_frame'])

    def __getitem__(self, idx):
        blurry_frame = self.data['blurry_frame'][idx]
        event_map = self.data['event_map'][idx]
        keypoints = self.data['keypoints'][idx]
        sharp_frame = self.data['sharp_frame'][idx]
        timestamps = np.linspace(-1, 1, sharp_frame.shape[0], dtype=np.float32)
        item = {
                'video_idx': self.video_idx, # scalar
                'frame_idx': idx, # scalar
                'blurry_frame': blurry_frame, # (1, 180, 240)
                'event_map': event_map, # (26, 180, 240)
                'keypoints': keypoints, # (10, 180, 240)
                'sharp_frame_lr': sharp_frame, # (14, 180, 240)
                'timestamps': timestamps # (14,)
                }
        return item

if __name__ == '__main__':
    dataset = HQFDataset()
    item = dataset[1]['event_map'].shape
    pdb.set_trace()
    print(123)


# if __name__ == '__main__':
#
#     block = EfficientAttention(in_channels=64, key_channels=128, head_count=4, value_channels=128)
#     input = torch.rand(8, 512, 12, 15)  # 输入 B C H W
#     output = block(input)
#     print(input.size())
#     print(output.size())



# import os
# import cv2
# import h5py
# import numpy as np
# from torch.utils.data import Dataset
# import random
#
# video_map = {
#     'bike_bay_hdr': 0,
#     'boxes': 1,
#     'desk': 2,
#     'desk_fast': 3,
#     'desk_hand_only': 4,
#     'desk_slow': 5,
#     'engineering_posters': 6,
#     'high_texture_plants': 7,
#     'poster_pillar_1': 8,
#     'poster_pillar_2': 9,
#     'reflective_materials': 10,
#     'slow_and_fast_desk': 11,
#     'slow_hand': 12,
#     'still_life': 13
# }
#
# class HQFDataset(Dataset):
#     def __init__(self, hdf5_name='/home/yj/Img_Rebuild/DataSet/DeblurSR/HQF/blurry_hdf5/bike_bay_hdr.hdf5', crop_size_h=128,crop_size_w=224):
#         super(HQFDataset, self).__init__()
#         self.data = h5py.File(hdf5_name, 'r')
#         self.video_idx = video_map[os.path.basename(hdf5_name[:-5])]
#         self.crop_size_h = crop_size_h
#         self.crop_size_w = crop_size_w
#
#     def __len__(self):
#         return len(self.data['blurry_frame'])
#
#     def random_crop(self, *arrays):
#         """
#         Randomly crop all input arrays at the same location
#         Args:
#             arrays: list of numpy arrays with shape (..., H, W)
#         Returns:
#             list of cropped arrays
#         """
#         h, w = arrays[0].shape[-2:]  # Get height and width from first array
#
#         # Calculate valid ranges for the crop
#         h_start_max = h - self.crop_size_h
#         w_start_max = w - self.crop_size_w
#
#         # Generate random crop coordinates
#         h_start = random.randint(0, h_start_max)
#         w_start = random.randint(0, w_start_max)
#
#         cropped_arrays = []
#         for array in arrays:
#             if len(array.shape) == 2:  # For 2D arrays
#                 crop = array[h_start:h_start+self.crop_size_h,
#                            w_start:w_start+self.crop_size_w]
#             else:  # For 3D arrays
#                 crop = array[..., h_start:h_start+self.crop_size_h,
#                            w_start:w_start+self.crop_size_w]
#             cropped_arrays.append(crop)
#
#         return cropped_arrays
#
#     def __getitem__(self, idx):
#         blurry_frame = self.data['blurry_frame'][idx]  # (1, 180, 240)
#         event_map = self.data['event_map'][idx]        # (26, 180, 240)
#         keypoints = self.data['keypoints'][idx]        # (10, 180, 240)
#         sharp_frame = self.data['sharp_frame'][idx]    # (14, 180, 240)
#         timestamps = np.linspace(-1, 1, sharp_frame.shape[0], dtype=np.float32)
#
#         # Perform random cropping on all spatial arrays
#         blurry_frame_crop, event_map_crop, keypoints_crop, sharp_frame_crop = \
#             self.random_crop(blurry_frame, event_map, keypoints, sharp_frame)
#
#         item = {
#             'video_idx': self.video_idx,      # scalar
#             'frame_idx': idx,                 # scalar
#             'blurry_frame': blurry_frame_crop,  # (1, 128, 224)
#             'event_map': event_map_crop,        # (26, 128, 224)
#             'keypoints': keypoints_crop,        # (10, 128, 224)
#             'sharp_frame_lr': sharp_frame_crop, # (14, 128, 224)
#             'timestamps': timestamps            # (14,)
#         }
#
#         return item
#
# if __name__ == '__main__':
#     dataset = HQFDataset()
#     item = dataset[1]
#     print("Cropped event map shape:", item['event_map'].shape)
#     print("Cropped blurry frame shape:", item['blurry_frame'].shape)