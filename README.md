# Simple Human Keypoints Detection and Social Distance Detection



## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using NVIDIA 2080TI. Other platforms or GPU cards are not fully tested.

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

4. Download  pretrained models from [GoogleDrive](https://drive.google.com/open?id=1-yRaMjMh7t0S8MUYndq-w6Rc3hPA1knu). Please download them under /pose_estimation.

   
### Run
    Modiy the video path in pose_estimation/detect_social_distance.py

   ```
    python pose_estimation/detect_social_distance.py
   ```
### Result

![Demo](result.gif)





