from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("yolov8n.yaml")
    results = model.train(data="LROC.yaml",epochs=200, batch=16, imgsz=640, pretrained=False)

