# 双主干网络用于月球陨石坑预测

种基于双主干特征融合网络的月球陨石坑识别模型DBYOLO（Dual-Backbone YOLO Network for Lunar Crater Detection），具有两个关键创新点，第一个创新点是提出双主干网络分别处理来自月球侦察轨道器相机获取CCD影像和和DTM影像，分别学习CCD影像的纹理边缘特征和DEM影像的地形深度特征，第二个创新点是提出了基于注意力机制的特征融合模块，实现了多源数据特征的动态融合，有效提取月球表面CCD影像和DEM数据的互补信息，提升了月球复杂表面环境下的检测性能。

## 数据集结构

用于训练、验证和测试双主干网络的数据集由配对的地形数据（DTM）和光学图像数据（CCD）组成，并附有陨石坑位置及属性标注。数据集结构如下：

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
val: val.txt          # txt文件里面包含训练图片路径
test: test.txt        # txt文件里面包含训练图片路径

# Classes
names:
  0: crater
```
