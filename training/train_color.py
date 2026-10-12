from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-cls.pt")
    
    results = model.train(
        data="datasets/colors",
        epochs=50,
        patience=15,
        device=0,
        imgsz=128,
        batch=64,
        workers=0,
        name="colors_v4"
    )
    
    

if __name__ == "__main__":
    main()