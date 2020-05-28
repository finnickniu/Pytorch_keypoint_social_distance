# Simple Baselines for Human Pose Estimation and Tracking

## News
- Our new work [High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514) is available at [HRNet](https://github.com/HRNet). Our HRNet has been applied to a wide range of vision tasks, such as [image classification](https://github.com/HRNet/HRNet-Image-Classification), [objection detection](https://github.com/HRNet/HRNet-Object-Detection), [semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) and [facial landmark](https://github.com/HRNet/HRNet-Facial-Landmark-Detection).
- Our new work [Deep High-Resolution Representation Learning for Human Pose Estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) has already been released at <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch>. The best single HRNet can obtain an **AP of 77.0** on COCO test-dev2017 dataset and **92.3% of PCKh@0.5** on MPII test set. The new repositoty also support the SimpleBaseline method, and you are welcomed to try it.<br>
- Our entry using this repo has won the winner of [PoseTrack2018 Multi-person Pose Tracking Challenge](https://posetrack.net/workshops/eccv2018/posetrack_eccv_2018_results.html)!<br>
- Our entry using this repo ranked 2nd place in the [keypoint detection task of COCO 2018](http://cocodataset.org/#keypoints-leaderboard)!

## Introduction
This is an official pytorch implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). This work provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field. State-of-the-art results are achieved on challenging benchmarks. On COCO keypoints valid dataset, our best **single model** achieves **74.3 of mAP**. You can reproduce our results using this repo. All models are provided for research purpose.    </br>

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
2. Disable cudnn for batch_norm:
   ```
   # PYTORCH=/path/to/pytorch
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```
   Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick a path where you'd like to have pytorch installed  and then set an environment variable (PYTORCH in this case) accordingly.
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.

4. Download  pretrained models from [GoogleDrive](https://drive.google.com/open?id=1-yRaMjMh7t0S8MUYndq-w6Rc3hPA1knu). Please download them under pose_estimation.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```
   
### Run
    Modiy the video path in pose_estimation/detect_social_distance.py

   ```
    python pose_estimation/detect_social_distance.py
   ```
### Rsult
<img src="./result.png.gif" alt="output"/>



