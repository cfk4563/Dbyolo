from ultralytics import YOLO



if __name__ == '__main__':

    model = YOLO("D:\\Desktop\\DByolo\\runs\\detect\\train5\\weights\\best.pt")
    model.val(data="LROC.yaml", split="test", epochs=200, imgsz=  640)