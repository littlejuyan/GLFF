from networks.utils.indices2coordinates import indices2coordinates
from networks.utils.compute_window_nums import compute_window_nums
import numpy as np


stride = 32
# channels = 2048
input_size = 224

# # The pth path of pretrained model
# pretrain_path = './models/pretrained/resnet50-19c8e357.pth'

N_list = [3, 3]
proposalN = sum(N_list)  # proposal window num
iou_threshs = [0.25, 0.25]
ratios = [[3, 3], [2, 2]]


# '''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
#print(window_nums) # [25,36]
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
# if set == 'CUB':
window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]

