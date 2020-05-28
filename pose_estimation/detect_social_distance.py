import torch
import cv2
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from PIL import Image , ImageDraw
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_final_preds
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import *
import models
from math import hypot
from itertools import combinations
import numpy as np
import heapq

def distance(p1,p2):
    """Euclidean distance between two points."""
    x1,y1 = p1
    x2,y2 = p2
    return hypot(x2 - x1, y2 - y1)



def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    
    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)
PATH=f"model.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = torch.load(PATH)
fps_time = 0

update_config("../experiments/coco/resnet152/384x288_d256x3_adam_lr1e-3.yaml")

cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED
model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False )
model.load_state_dict(torch.load("pose_resnet_152_384x288.pth.tar"))

model=model.cuda()

model.eval()

net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # 1 class (person) + background
in_features =  net.roi_heads.box_predictor.cls_score.in_features
net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

net.load_state_dict(checkpoint)
net.to(device)
net.eval()


cap = cv2.VideoCapture("/media/apptech/golden/dataset/harbournorth_video/AXISP322_192.168.100.103 - 1920 x 1080 - 15fps_20191118_144455.avi")
fps_time=0
while True:
    ret, image = cap.read()  
    image1 = Image.fromarray(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB))  
    image_tensor=transform1(image1)

    image_tensor=image_tensor.unsqueeze(0).to(device)

    outputs = net(image_tensor)
    scores=outputs[0]["scores"].data
    bb_box=outputs[0]["boxes"]
    class_=outputs[0]["labels"].data
    num_people=0

    for n,q in enumerate(class_):
        if scores[n]>0.99:
            if q ==1:
                num_people+=1

    points_lists=[]
    for i,s in enumerate(scores):
        if s >=0.9:
            bb_box=outputs[0]["boxes"].data[i]
            bbox = [bb_box[0],bb_box[1],bb_box[2]-bb_box[0],bb_box[3]-bb_box[1]]
            x_mean = int((bb_box[2]+bb_box[0])//2)
            y_mean = int((bb_box[1]+bb_box[3])//2)
            points_lists.append((x_mean,y_mean))

            c, s = _box2cs(bbox, image.shape[0], image.shape[1])
            r = 0
            trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
            input = cv2.warpAffine(
            image,
            trans,
            (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
            input = transform(input).unsqueeze(0).cuda()
            with torch.no_grad():
                output = model(input)
                preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))



                # plot
                image = image.copy()

                for mat in preds[0]:
                    x, y = int(mat[0]), int(mat[1])
                    cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

            cv2.rectangle(image,(bb_box[2],bb_box[1]),(bb_box[0],bb_box[3]),(0,255,0),2)

    for i in points_lists:
        x_mean = i[0]
        y_mean = i[1]
        distance_list= [distance(i,a) for a in points_lists]
        min_num_index_list = map(distance_list.index, heapq.nsmallest(2, distance_list))
        second_small_index = list(min_num_index_list)[-1]
        second_small_coors=points_lists[second_small_index]
        if distance_list[second_small_index]<250:
            cv2.line(image, (x_mean, y_mean), (second_small_coors[0], second_small_coors[1]), (0, 0, 255), thickness=2)


        cv2.circle(image, (x_mean,y_mean), 2, (0, 255, 0), 2)




    cv2.putText(image, f"num of people: {num_people}", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(image, f"Count Mode", (960, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(image,  "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    fps_time = time.time()
    cv2.imshow('Skeleton', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
