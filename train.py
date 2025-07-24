# from ultralytics import YOLO
#
#
# if __name__ == '__main__':
#
#     model = YOLO("YOLOFusion.yaml")
#     results = model.train(data="LROC.yaml",epochs=200, batch=16, imgsz=640, pretrained=False)

from ultralytics import YOLO


if __name__ == '__main__':
    yaml = [
        # 'DByolo.yaml',
        # 'DByolo_1_AaN.yaml',
        # 'DByolo_1_BiFPN.yaml',
        # 'DByolo_1_CaC.yaml',
        # 'DByolo_1_DFM.yaml',
        # 'DByolo_1_MDFM.yaml',
        'DByolo_1_TFAM.yaml',
        'DByolo_BnoFocus.yaml',
        # 'DByolo_BnoHWD.yaml',
        'DByolo_MnoAttention.yaml',
        'DByolo_MnoGamma.yaml',
        'DByolo_MonlyAttention.yaml',
        'DByolo_MonlyResnet.yaml',
        # 'DEyolo.yaml',
        # 'yoloDB5.yaml',
        # 'yoloDB8.yaml',
        # 'yoloDB10.yaml',
        # 'yoloDB11.yaml',
        # 'yoloDB12.yaml',
        # 'YOLOFusion.yaml',
    ]
    for item in yaml:
        model = YOLO(item)
        name = item.split(".")[0]
        results = model.train(data="LROC.yaml",
                              epochs=200,
                              batch=16,
                              imgsz=640,
                              pretrained=False,
                              workers=4,
                              name = name)
