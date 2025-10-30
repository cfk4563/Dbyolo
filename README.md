# DBYOLO: Dual-Backbone YOLO Network for Lunar Crater Detection

Craters are among the most prominent and significant geomorphological features on the lunar surface. The complex and variable environment of the lunar surface, which is characterized by diverse textures, lighting conditions, and terrain variations, poses significant challenges to existing crater detection methods. To address these challenges, this study introduces DBYOLO, an innovative deep learning framework designed for lunar crater detection, leveraging a dual-backbone feature fusion network, with two key innovations. The first innovation is a lightweight dual-backbone network that processes Lunar Reconnaissance Orbiter Camera (LROC) CCD images and Digital Terrain Model (DTM) data separately, extracting texture and edge features from CCD images and terrain depth features from DTM data. The second innovation is a feature fusion module with attention mechanisms that is used to dynamically integrate multi-source data, enabling the efficient extraction of complementary information from both CCD images and DTM data, enhancing crater detection performance in complex lunar surface environments. Experimental results demonstrate that DBYOLO, with only 3.6 million parameters, achieves a precision of 77.2%, recall of 70.3%, mAP50 of 79.4%, and mAP50-95 of 50.4%, representing improvements of 3.1%, 1.8%, 3.1%, and 2.6%, respectively, over the baseline model before modifications. This showcases an overall performance enhancement, providing a new solution for lunar crater detection and offering significant support for future lunar exploration efforts.

## 数据集结构

用于训练、验证和测试双主干网络的数据集由配对的地形数据（DTM）和光学图像数据（CCD）组成，并附有陨石坑位置及属性标注。数据集结构如下：
通过网盘分享的文件：数据集.zip
链接: https://pan.baidu.com/s/1cqNbKh10AL2vGSRnEInMCA 提取码: 29ny
#### LROC 数据集结构

```
LROC/
├── gray/                   # CCD灰度图像数据
│   ├── train/
│   │   ├── img_0.jpg
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img_4.jpg
│   │   ├── img_5.jpg
│   │   └── ...
│   ├── test/
│   │   ├── img_8.jpg
│   │   ├── img_9.jpg
│   │   └── ...
├── dtm/                    # DTM地形数据
│   ├── train/
│   │   ├── img_0.jpg
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img_4.jpg
│   │   ├── img_5.jpg
│   │   └── ...
│   ├── test/
│   │   ├── img_8.jpg
│   │   ├── img_9.jpg
│   │   └── ...
├── txt/                    # 标签数据
│   ├── train/
│   │   ├── img_0.txt
│   │   ├── img_1.txt
│   │   └── ...
│   ├── val/
│   │   ├── img_4.txt
│   │   ├── img_5.txt
│   │   └── ...
│   ├── test/
│   │   ├── img_8.txt
│   │   ├── img_9.txt
│   │   └── ...
├── train.txt               # 训练集文件列表
├── val.txt                 # 验证集文件列表
├── test.txt                # 测试集文件列表
```
#### LROC.yaml文件
## 数据集配置文件（用于 YOLO）

```yaml
path: D:\Desktop\DByolo\datasets\LROC  # 数据路径
train: train.txt      # txt文件里面包含训练图片路径
val: val.txt          # txt文件里面包含验证图片路径
test: test.txt        # txt文件里面包含测试图片路径

# Classes
names:
  0: crater
```
## 训练
```python
python train.py
```
## 验证
```python
python test.py
```
## 预测
```python
python heatmap.py
```
## Citation
Please cite our paper if you find the work useful:
```
@Article{rs17193377,
AUTHOR = {Liu, Yawen and Chen, Fukang and Qiu, Denggao and Liu, Wei and Yan, Jianguo},
TITLE = {DBYOLO: Dual-Backbone YOLO Network for Lunar Crater Detection},
JOURNAL = {Remote Sensing},
VOLUME = {17},
YEAR = {2025},
NUMBER = {19},
ARTICLE-NUMBER = {3377},
URL = {https://www.mdpi.com/2072-4292/17/19/3377},
ISSN = {2072-4292},
DOI = {10.3390/rs17193377}
}

```
