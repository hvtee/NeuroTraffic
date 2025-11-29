from ultralytics import YOLO


def create_model():
    path_to_data = 'src/cars.v3-our-photos.yolov11/cars.v3-our-photos.yolov11.yaml'

    model = YOLO("yolo11s.yaml")  # build a new model from YAML
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=path_to_data, epochs=100, imgsz=640)


create_model()
