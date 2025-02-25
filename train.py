from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("yolov8n.yaml").load('yolov8n.pt')
    results = model.train(data="LROC.yaml",epochs=200, batch=8, imgsz=640)

