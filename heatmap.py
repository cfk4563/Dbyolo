import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from pathlib import Path
from ultralytics.utils import ops
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.plotting import output_to_target , Annotator,Colors

colors = Colors()

def predict(model, x1, x2):
    save = model.save
    model = model.model
    x = None
    y = []  # outputs
    for m in model:
        if m.part == "backbone":
            x1 = m(x1)
            y.append(x1)
        elif m.part == "backbone2":
            x2 = m(x2)
            y.append(x2)
        else:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x)  # save output
    return x, y

def plot_images(
    images,
    batch_idx,
    cls,
    bboxes = np.zeros(0, dtype=np.float32),
    confs = None,
    paths = None,
    fname: str = "images.jpg",
    names = None,
    max_size = 1920,
    max_subplots = 16,
    save = True,
    conf_thres = 0.25,
) :

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color, rotated=is_obb)

    annotator.im.save(fname)  # save


def draw_heatmap(feature_maps):
    """从特征图列表生成热力图"""

    heatmap = feature_maps.sum(dim=1).squeeze().cpu().numpy()  # 应为 (80,80)
    # 应用 ReLU，确保非负值
    heatmap = np.maximum(heatmap, 0)
    # 归一化到 [0,1]，避免除以零
    heatmap_max = heatmap.max()
    heatmap = heatmap / (heatmap_max + 1e-6)  # 添加小值防止除以零
    heatmap = heatmap.astype(np.float32)
    heatmap = cv2.resize(heatmap, (608, 608))
    
    # 创建彩色热力图
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap_color

def main():
    # 从权重文件加载状态字典
    weights_path = "D:\\Desktop\\DByolo\\runs\\detect\\train5\\weights\\best.pt"
    state_dict = torch.load(weights_path)
    model = state_dict['model'].to("cuda")

    # 验证权重是否加载
    print("验证权重加载情况：")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, 前5个值: {param.data.flatten()[:5]}")
        break  # 只打印第一个参数

    # 加载图像并预处理
    ccd_path = 'D:\\Desktop\\DByolo\\datasets\\LROC\\gray\\train\\img_224.jpg'
    dem_path = 'D:\\Desktop\\DByolo\\datasets\\LROC\\dem\\train\\img_224.jpg'
    ccd = cv2.imread(ccd_path)
    dem = cv2.imread(dem_path)

    def preprocess(image):
        image = cv2.resize(image, (640, 640))  # 确保大小与训练时一致
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = image / 255.0  # 归一化到[0, 1]
        return torch.from_numpy(image).float().unsqueeze(0)  # 添加批次维度：[1, C, H, W]

    ccd_processed = preprocess(ccd).half().to("cuda")
    dem_processed = preprocess(dem).half().to("cuda")

    # 设置评估模式并预测
    model.eval()
    with torch.no_grad():
        preds, feature_maps= predict(model, ccd_processed, dem_processed)

    preds = non_max_suppression(
        preds,
        0.01,
        0.75,
        labels=[],
        nc=1,
        multi_label=True,
        agnostic=True,
        max_det=300,
        end2end=False,
        rotated=False,
    )
 
    plot_images(
        ccd_processed,
        *output_to_target(preds, max_det=300),
        paths=ccd_path,
        fname="pred.jpg",
        names={0:"crater"},
        )
    return feature_maps, ccd, dem

feature_maps, ccd, dem = main()

for i in [18,19,20,27,30,33]:
# 绘制热力图
    heatmap = draw_heatmap(feature_maps[i])

    # 将热力图与原图叠加
    overlay = cv2.addWeighted(ccd, 0.6, heatmap, 0.4, 0)

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Heatmap')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存结果
    cv2.imwrite(f'./heatmap/heatmap{i}.jpg', heatmap)
    cv2.imwrite(f'./heatmap/overlay{i}.jpg', overlay)