from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-cls.pt")
    
    results = model.train(
        fliplr = 0.0,
        data="datasets/tokens",
        epochs=50,
        patience=15,
        device=0,
        imgsz=96,
        batch=64,
        workers=0,
        name="tokens_v2"
    )
    
    

if __name__ == "__main__":
    main()