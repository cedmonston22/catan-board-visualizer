from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")
    
    results = model.train(
        data="datasets/corners/data.yaml",
        epochs=100,
        patience=25,
        device=0,
        imgsz=1024,
        batch=4,
        workers=0,
        name="corners_v2"
    )
    
    

if __name__ == "__main__":
    main()