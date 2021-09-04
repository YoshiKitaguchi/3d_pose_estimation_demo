from PIL.Image import NONE
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from torch.utils.data.dataset import ConcatDataset
import copy


def transform (dots_batch, scale, num_dim = 3):
    # x_max, x_min, y_max, y_min, z_max, z_min
    max_min_arr = [0,4096,0,4096,0,10000]
    num_points = int(len(dots_batch)/num_dim)

    for i in range (num_points):
        for j in range(num_dim):
            max_min_arr[j*2] = max(max_min_arr[j*2], dots_batch[i*num_dim + j])
            max_min_arr[j*2 + 1] = min(max_min_arr[j*2+1], dots_batch[i*num_dim + j])

    my_depth = 960 / scale

    # width, height, ori_depth
    size_arr = [max_min_arr[i*2] - max_min_arr[i*2+1] for i in range (num_dim)]

    # box_width, box_height, my_depth
    box_arr = [size_arr[0], size_arr[1], my_depth]

    box_center = [ (max_min_arr[2*i+1] + size_arr[i]/2) for i in range(num_dim)]
    # print (box_center)
    if (size_arr[0] / size_arr[1] > 4/3): box_arr[1] = size_arr[0] * 3/4
    else: box_arr[0] = size_arr[1] * 4/3

    box_bottomLeft = [(box_center[i] - box_arr[i]/2) for i in range(num_dim)]

    extension_offset = 0.2
    extension_arr = [box_bottomLeft[0] * extension_offset, box_bottomLeft[1] * extension_offset, 0]
    offset = [(box_bottomLeft[i] - extension_arr[i]) for i in range(num_dim)]
    transformed_arr = []
    for i in range (num_points):
        transformed_arr += [(dots_batch[i*num_dim+j] - offset[j]) for j in range(num_dim)]

    return torch.tensor(transformed_arr)

def init_fig ():
    fig = plt.figure()
    ax = Axes3D(fig)
    return fig, ax

def set_ax_env(ax, _max_range):
    ax.set_xlim3d(0, _max_range)
    ax.set_ylim3d(0, _max_range)
    ax.set_zlim3d(0, _max_range)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

def update_points (ax, tar=[],pred=[],num_dim=3, bias=0):
    if len(tar) > 0: num_points = int(len(tar[0])/num_dim)
    elif len(pred) > 0: num_points = int(len(pred[0])/num_dim)
    else: return
    for i in range (num_points):
        if (num_dim == 2) :
            if len(pred) > 0:
                ax.scatter(pred[0][i*num_dim], 0, pred[0][i*num_dim + 1], marker='o', c ='b', s= 1)
                drawLine(pred[0],ax, 'b',num_dim=2)
            if len(tar) > 0:
                ax.scatter(tar[0][i*num_dim]+bias, 0, tar[0][i*num_dim + 1], marker='o', c ='r', s= 1)
                drawLine(tar[0],ax, 'r',num_dim=2)
        else :
            if len(pred) > 0:
                ax.scatter(pred[0][i*num_dim], pred[0][i*num_dim + 2], pred[0][i*num_dim + 1], marker='o', c ='b', s= 1)
                drawLine(pred[0],ax, 'b')
            if len(tar) > 0:
                ax.scatter(tar[0][i*num_dim]+bias, tar[0][i*num_dim + 2], tar[0][i*num_dim + 1], marker='o', c ='r', s= 1)
                drawLine(tar[0],ax, 'r')

def view_dataset (tar= [], pred= [], _max_range = 1500, num_dim = 3, shift = True, given_fig = None, given_ax = None):
    if len(tar) == 0 and len(pred) == 0: return
    bias = 0
    if shift:
        bias = _max_range
        _max_range *= 2
    fig, ax = init_fig()
    set_ax_env(ax, _max_range)
    update_points(ax, tar, pred,num_dim,bias)
    plt.show()

def drawLine(points, ax, color, data='mpi', num_dim = 3):
    if data == 'mpi':
        edges = [[0,1],[1,2],[3,4],[4,5],[0,6],[6,7],[7,8],[3,9],[9,10],[10,11],[6,9],[0,3],[0,12],[3,12]]
    for edge in edges:
        x = [points[edge[0]*num_dim+0],points[edge[1]*num_dim+0]]
        z = [points[edge[0]*num_dim+1],points[edge[1]*num_dim+1]]
        y = [0,0]
        if num_dim == 3:
            y = [points[edge[0]*num_dim+2],points[edge[1]*num_dim+2]]
        ax.plot(x,y,z, color+'-')

def removePoints (dots, joints_to_remove, num_dim = 3):
    cleand_arr = []
    n = len(dots)//num_dim
    
    for i in range (n):
        if (i in joints_to_remove):
            continue
        cleand_arr += [dots[i*num_dim+x] for x in range(num_dim)]
    removed_tensor = torch.as_tensor(cleand_arr)
    return removed_tensor

def setXYorigin(dots, num_dim = 3):
    mins = [float('inf') for i in range(num_dim)]
    for i in range(len(dots)//num_dim):
        for x in range(num_dim): 
            mins[x] = copy.deepcopy(min(dots[i*num_dim + x], mins[x]))
    for i in range(len(dots)//num_dim):
        for x in range(num_dim): 
            dots[i*num_dim + x] = dots[i*num_dim + x] - mins[x]
    return dots
        
def coco_to_mpi(pts, num_dim = 2):
    edges = [[2,0],[4,1],[6,2],[1,3],[3,4],[5,5],[8,6],[10,7],[12,8],[7,9],[9,10],[11,11],[0,12]]
    changed = np.copy(pts)
    for edge in edges:
        changed[:,edge[1] * num_dim] = pts[:,edge[0] * num_dim]
        changed[:,edge[1] * num_dim+1] = pts[:,edge[0] * num_dim+1] 
    return changed


class depthDatasetMemory(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.annot = h5py.File(folder+'/annot_data.mat', 'r')
        self.scale = 200
        self.bias = 2048

    def __getitem__(self, idx):
        image = cv2.imread(self.folder+"/imageSequence/img_"+str(idx+1).zfill(6)+".jpg")
        image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        _2d_sample = self.annot['annot2'][idx][0]
        _3d_sample = self.annot['univ_annot3'][idx][0]
        
        num_dots = torch.from_numpy(self.annot['annot2'][idx]).shape[1]
        _2D_dots = torch.zeros(num_dots*2)
        _3D_dots = torch.zeros(num_dots*3)
        
        for i in range (num_dots):
            _2D_dots[i*2] = _2d_sample[i][0]
            _2D_dots[i*2 +1] = self.bias-_2d_sample[i][1]
            _3D_dots[i*3 + 0] = (_3d_sample[i][0] + self.bias) / self.scale #xs
            _3D_dots[i*3 + 1] = (self.bias - _3d_sample[i][1]) / self.scale #zs
            _3D_dots[i*3 + 2] = (_3d_sample[i][2] + self.bias) / self.scale #ys

        _2D_dots = removePoints (_2D_dots, [0,1,14,15], 2)
        _3D_dots = removePoints (_3D_dots, [0,1,14,15], 3)
        _2D_dots = setXYorigin(_2D_dots,2)
        _3D_dots = setXYorigin(_3D_dots,3)
        _2D_dots = transform(_2D_dots, self.scale, 2)
        _3D_dots = transform(_3D_dots, self.scale, 3)

        _3d_sample = {'images': image,'_2d_dots': _2D_dots, '_3d_dots': _3D_dots}
        return _3d_sample

    def __len__(self):
        return len(self.annot['univ_annot3'])

def getTestingData(batch_size, folders, shuffle=True):
    return DataLoader(ConcatDataset([depthDatasetMemory(folder) for folder in folders]), batch_size, shuffle=shuffle) 





if __name__ == "__main__":
    loader = getTestingData(1, ["TS1"], False)
    for sample in loader:
        image = sample['image']
        _2d_dots = sample['_2d_dots']
        _3d_dots = sample['_3d_dots']
        view_dataset(_3d_dots,_max_range=10)
        break