from ultralytics import YOLO
import os

if __name__ == '__main__':

    model = [
            # 'DByolo',
            # 'DByolo_1_AaN',
            # 'DByolo_1_BiFPN',
            # 'DByolo_1_CaC',
            # 'DByolo_1_DFM',
            #  'DByolo_1_MDFM',
            #  'DByolo_1_TFAM',
            #  'DByolo_BnoFocus',
            #  'DByolo_BnoHWD',
        'DByolo_MnoAttention',
        'DByolo_MnoGamma',
        'DByolo_MonlyAttention',
        'DByolo_MonlyResnet',

             ]

    dir = r"D:\Dbyolo\runs\detect"

    for name in model:
        f = os.path.join(dir, name,r"weights\best.pt")

        model = YOLO(f)
        model.val(data="LROC.yaml", split="test", epochs=200, imgsz= 640)