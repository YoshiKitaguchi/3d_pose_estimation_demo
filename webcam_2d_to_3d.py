import cv2
import torch
import time
import matplotlib.pyplot as plt
from SimpleHigherHRNet import SimpleHigherHRNet
from model_2d_to_3d import LinearModel, weight_init
from dataloader import init_fig, set_ax_env, update_points, coco_to_mpi

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


vid = cv2.VideoCapture(0)

plt.ion()
fig, ax = init_fig()
while (1):
    ret, frame = vid.read()
    image = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = time.time()
    _2d_pred = hhrnet.predict(image)
    if _2d_pred.shape[1] == 0:
        continue
    _2d_pred = coco_to_mpi(_2d_pred)
    _3d_pred = model_2d_to_3d(torch.tensor(_2d_pred).cuda()).cpu().detach().numpy()
    print((time.time() - start),"s")
    set_ax_env(ax, _max_range=10)
    update_points(ax,_3d_pred)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.cla() 