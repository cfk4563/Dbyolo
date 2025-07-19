from ultralytics import YOLO



if __name__ == '__main__':

    model = YOLO(r"D:\Dbyolo\runs\detect\DEyolo2\weights\last.pt")
    model.val(data="LROC.yaml", split="test", epochs=200, imgsz=  640)