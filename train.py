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
        "DByolo.yaml",
        "DEyolo.yaml",
        # "yoloDB10.yaml",
        # "yoloDB11.yaml",
        # "yoloDB12.yaml",
    ]
    for item in yaml:
        model = YOLO(item).load("yolov8n.pt")
        name = item.split(".")[0]
        results = model.train(data="LROC.yaml",
                              epochs=200,
                              batch=16,
                              imgsz=640,
                              pretrained=False,
                              workers=4,
                              name = name)
