LAED <img src="/README/OIG.png" width="280px" align="right" />
===========
### Leukemia Assessment via End-to-end Deep Learning
LAED is an end-to-end deep learning approach for predicting leukemia type. It helps hematologists diagnose leukemia more accurately and efficiently by automatically identifying and classifying the different types of cells in marrow smears.

## Requirements: 
* Windows on 64-bit x86 
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090)
* Python 3.10 

## Quick start: 
To reproduce the experiments in our paper, please down the dataset from [here](https://figshare.com/articles/dataset/single_cell_dataset/19787371).  
The following example data are stored under a folder named DATA_DIRECTORY. Data from same patients assigned to one folder, stored under the folder of the according leukemia type (ALL, AML, APL, CLL, CML).  
```bash
DATA_DIRECTORY/train/
	├── ALL
		├── Patinent-a
			├── Leukocyte 1
			├── Leukocyte 2
			├── ....
		├── Patinent-b
			├── Leukocyte 1
			├── Leukocyte 2
			├── ...
		├── ...
	├── AML
		├── Patinent-c
			├── Leukocyte 1
			├── Leukocyte 2
			├── ....
		├── Patinent-d
			├── Leukocyte 1
			├── Leukocyte 2
			├── ...
		├── ...
	├── APL
	├── CLL
	└── CML
DATA_DIRECTORY/test/
	├── ALL
	├── AML
	├── APL
	├── CLL
	└── CML
```
Data in one hospital are used to train model, three other hospitals are used to test model's performance. Moreover, the microscopy instruments used in these hospitals differed as well.  
Before training the model, please download pre-trained CNN backbone from [here](https://figshare.com/articles/dataset/Trained_model/19787464).  
For quick testing, you can downlaod pre-trained LAED from [here](https://figshare.com/articles/dataset/Trained_model/19787464).  
```bash
python main.py --data_path DATA_DIRECTORY --backbone_checkpoint BACKBONE_DIRECTORY --train_bag_length 128 --epochs 30
```

## Custom dataset
We recommend users to find the ROIs from WSIs and save the images at first. We also provide the pre-trained Mask-RCNN model to segment leukocytes based on [Detectron2](https://github.com/facebookresearch/detectron2), you can download from [here](https://figshare.com/articles/dataset/Trained_model/19787464). For image segmentation. After the construction of single cell dataset for each patient, store the data as the format in **Quick start**.
![segmentation](/README/figure_2.png)
In order to achieve the high leukemia type prediction accuracy, the more samples the better. Based on our experience, it is better to have an average of 300 or more leukocytes for each patient.  
For [Detectron2](https://github.com/facebookresearch/detectron2) installing  on the Windos system, please use the released v0.6, and follow the instructions below to deploy.  
Open /detectron2-0.6/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu with Notepad.  
Replace the code:  
```bash
// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#ifdef WITH_CUDA
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#endif
// TODO avoid this when pytorch supports "same directory" hipification
#ifdef WITH_HIP
#include "box_iou_rotated/box_iou_rotated_utils.h"
#endif
```
with:
```bash
// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
/*#ifdef WITH_CUDA
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#endif
// TODO avoid this when pytorch supports "same directory" hipification
#ifdef WITH_HIP
#include "box_iou_rotated/box_iou_rotated_utils.h"
#endif*/
#include "box_iou_rotated/box_iou_rotated_utils.h"
```
Install antlr4 using pip:
```bash
antlr4-python3-runtime==4.9.3
```
Enter the command in the terminal command line:
```bash
python setup.py build develop
```

## Visualization
To check the predicted results, users can visalize the attention score of each leukocyte. Most instances of high attention scores denoted atypical cell types relevant to pathological conditions. In such cases, the attention score could guide hematologists toward the leukocytes relevant to leukemia diagnosis. For the visualizing attention scores, [Detectron2](https://github.com/facebookresearch/detectron2) is required.
![segmentation](/README/atten.png)
