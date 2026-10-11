from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-cls.pt")
    
    results = model.train(
        data="datasets/tiles",
        epochs=50,
        patience=15,
        device=0,
        imgsz=224,
        batch=64,
        workers=0,
        name="tiles_v1"
    )
    
    

if __name__ == "__main__":
    main()