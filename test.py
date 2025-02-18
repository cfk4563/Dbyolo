from ultralytics import YOLO
import os
import re


def extract_layers(model_str):
    # pattern_with_num = r'\((\d+)\):\s+(\w+)\('
    pattern = r'\((\d+)\):\s*(\w+)'
    # 使用 re.findall 找到所有匹配的编号和层类型
    matches = re.findall(pattern, model_str)
    # 创建一个列表，存储编号和层类型
    layers = []
    for num, layer_type in matches:
        if layer_type not in ["Bottleneck", "Sequential"]:
            layers.append(f"{layer_type}")

    return layers


if __name__ == '__main__':
    #
    # a = "D:/Desktop/ultralytics/runs/detect/train/weights/best.pt"
    # list = []
    # list.append(a)
    # for i in range(2, 7):
    #     tmp = "train" + str(i)
    #     b = a.replace("train", tmp)
    #     list.append(b)
    # for train in list:
    #     try:
    #         print(train)
    #         if os.path.exists(train):
    #             model = YOLO(train)
    #             print(extract_layers(str(model)))
    #             model.val(data="LROC.yaml", split="test", epochs=200, imgsz=640)
    #     except Exception as e:
    #         # 如果抛出异常，打印异常信息并跳到下一个循环
    #         print(f"Error with {train}: {e}")
    #         continue  # 跳到下一个循环
    #     print("\n\n")
    model = YOLO("D:/Desktop/ultralytics/runs/detect/train11/weights/best.pt")
    model.val(data="LROC.yaml", split="test", epochs=200, imgsz=640)