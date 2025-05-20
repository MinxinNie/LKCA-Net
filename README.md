# LKCA-NET
This repository contains official implementation for the paper titled "LKCA-Net: Hybrid Large Kernel and Clustering Attention Network for Medical Image Segmentation" 

> **Architecture**
![Method](figure/net.jpg)

## 1. Prepare data

- [Synapse multi-organ segmentation] The Synapse datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). 
- [ACDC cardiac segmentation  Download the preprocessed ACDC dataset from [Google Drive of MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) ] 
- [The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}.]
  - After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)
  - './data/isic17/'
    - train
      - images
        - .png
      - masks
        - .png
    - val
      - images
        - .png
      - masks
        - .png

## 2. Environment
- We recommend an evironment with python >= 3.8, and then install the following dependencies:
```
pip install -r requirements.txt
```

- We recommend to install **Defomrable Convolution** manually for compatability issues:
    - [**Deformable Convolution**] There are many implementation of deformable convolution:
        - [**tvdcn**] We recommend the implementation in **tvdcn** (https://github.com/inspiros/tvdcn), as it provides CUDA implementation of both 2D/3D deformable convolution (The 2D implementation of deformable convolution in tvdcn should be the same as that provided by PyTorch) [**Note: We used tvdcn for our experiments**]
        For example, we can install latest tvdcn with Pytorch >= 2.1 and CUDA >= 12.1 with
        ```
        pip install tvdcn
        ```
    
- **Final Takeaway:** We suggest installing PyTorch >= 2.1, CUDA >= 12.1 for better compatability of all pacakges (especially tvdcn and natten). It is also possible to install those two packages with lower PyTorch and CUDA version, but they may need to be built from source. 





## 3. Synapse Dataset & ACDC Dataset

### a. Modify the dataset path configuration in train.py and select the dataset to be trained.



### b. Run the script
```
python train.py --cfg [config_file in configs]
```
For example, for training Synapse tiny model, run the following command:
```
python train.py --cfg configs/lalk_base.yaml
```

Run the below code to test the Net on the Synapse dataset.

```
python test.py --cfg configs/lalk_base_synapse_pretrained.yaml
```



## ISIC 2017  & ISIC2017

Run the code below to train Net on the ISIC dataset.

```\
python train_ISIC.py --cfg configs/lalk_small_isic.yaml
```

Run the below code to test the Net on the ISIC dataset.

```
python test_isic.py --cfg configs/lalk_small_isic_pretrained.yaml
```




## Acknowledgements

This code is built on the top of [AgileFormer](https://github.com/sotiraslab/AgileFormer), [DLKA](https://github.com/xmindflow/deformableLKA) and [MALUNet](https://github.com/JCruan519/MALUNet/tree/main), we thank to their efficient and neat codebase. 

