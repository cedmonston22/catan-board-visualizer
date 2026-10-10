from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/catan_v1/weights/best.pt")

    metrics = model.val(
        data="datasets/catan/data.yaml",
        device=0,
        imgsz=1024,
        batch=4,
        workers=0,
        name="catan_v1_cleanval"
    )

    print(f"Overall mAP50: {metrics.box.map50:.3f} Overall mAP50-95: {metrics.box.map:.3f}")
    for i, name in metrics.names.items():
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{name:<12} P: {p:.3f}  R: {r:.3f}  mAP50: {ap50:.3f}  mAP50-95: {ap:.3f}")

if __name__ == "__main__":
    main()
