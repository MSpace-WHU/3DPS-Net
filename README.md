# 3DPS-Net

## Overview

Our **3D Promptable Segmentation Network (3DPS-Net)** is a novel approach for instance segmentation of individual trees
in 3D forest point clouds.
It leverages prompt points to generate object masks, enabling the extraction of individual trees.
3DPS-Net offers two testing modes:

1. **Prompt Testing**: Interactive manual tree segmentation.
2. **Automatic Testing**: Autonomous segmentation of individual trees.

## Requirements

- Python 3.x
- PyTorch 1.12
- CUDA 9.2 or higher

The installation guide suppose ``pyhon=3.7`` ``pytorch=1.12`` ``CUDA=11.3``. You may change them according to your
system.

## Installation

1. **Create a conda virtual environment and activate it**:
   ```
   conda create -n 3dpsnet python=3.7
   conda activate 3dpsnet
   ```
2. **Clone the repository**:
   ```
   git clone https://github.com/MSpace-WHU/3DPS-Net.git
   ```
3. **Install the dependencies**:
   ```
   pip install -r requirements.txt
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
   cd cpp_wrappers
   sh compile_wrappers.sh 
   cd ..
   ```
   
## Data
The forest point cloud data can be in ``.txt`` or ``.ply`` format.

The format follows the structure of ``[x, y, z, r, g, b, sem_label, ins_label].``

   - `x`, `y`, `z`: The spatial coordinates of each point.
   - `r`, `g`, `b`: The RGB color values of each point.
   - `sem_label`: The semantic label.
   - `ins_label`: The instance label.

Among these, r, g, b, and semantic label are not utilized. 

If the data format is different, modifications to the ``data_acquire`` function in ``dataset.py`` will be required.

## Usage

1. **Prepare the dataset and perform down-sampling**: 

The raw point cloud data can be in ``.ply``, ``.txt`` or ``.las`` formats. It is also acceptable to have various types of data within the data files. 

Use ``sample.py`` for down-sampling, and the sampled data will be stored in the data folder with the following structure:

   ```
   ├── data/               
   │   ├── forest/            # dataset name
   │   │   ├── train/         # training set
   │   │   ├── test/          # test set
   │   │   ├── val/           # validation set
   ```

2. **Training**:
Start training the network by running:
   ```
   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./train.py --dataset_name=forest
   ```

3. **Evaluation**:
After training, evaluate the model using:
   ```
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./test.py --dataset_name=forest --pretrained=/home/user/Desktop/pointSAM/PointSAM2/params/forinstance/epoch_83_0.99046.pth
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.