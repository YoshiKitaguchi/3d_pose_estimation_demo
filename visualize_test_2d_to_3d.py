import torch
import matplotlib.pyplot as plt
from SimpleHigherHRNet import SimpleHigherHRNet
from model_2d_to_3d import LinearModel, weight_init
from dataloader import getTestingData, view_dataset, coco_to_mpi
hhrnet = SimpleHigherHRNet(
    c=48, nof_joints=17, checkpoint_path='./weights/pose_higher_hrnet_w48_640.pth.tar',
    resolution=640, device='cuda', filter_redundant_poses=False
)
print(">>> loading SimpleHigherHRNet")
model_2d_to_3d = LinearModel()
model_2d_to_3d = model_2d_to_3d.cuda()
model_2d_to_3d.apply(weight_init)
model_2d_to_3d_path = "weights/2d_to_3d.pth.tar"
print(">>> loading ckpt of 2d_to_3d model from '{}'".format(model_2d_to_3d_path))
ckpt_2d_to_3d = torch.load(model_2d_to_3d_path)
model_2d_to_3d.load_state_dict(ckpt_2d_to_3d['state_dict'])
model_2d_to_3d.eval()

loader = getTestingData(1, ["dataset/TS1"], True)
for sample in loader:
    images = sample['images'].cpu().detach().numpy()
    _2d_dots = sample['_2d_dots'].cuda()
    _3d_dots = sample['_3d_dots'].cuda()
    plt.imshow(images[0])
    plt.show()
    _2d_pred = hhrnet.predict(images)
    _2d_pred = coco_to_mpi(_2d_pred)
    _3d_pred = model_2d_to_3d(torch.tensor(_2d_pred).cuda()).cpu().detach().numpy()
    view_dataset(_2d_dots.cpu().detach().numpy(),_2d_pred,1000,2,False)
    view_dataset(_3d_dots.cpu().detach().numpy(),_3d_pred,_max_range=10,num_dim=3,shift=False)
    # break