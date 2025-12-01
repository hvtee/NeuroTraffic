from ultralytics import YOLO


def check_model():
    path_to_model = 'models/NeuroTrafficV2.pt'
    path_to_test_img = 'check_img/test_img_2.png'
    path_to_output = "check_model/output"

    try:
        model = YOLO(path_to_model)
    except Exception as e:
        print(f"Couldn't load model: {e}")
        return

    try:
        results = model(path_to_test_img)
    except Exception as e:
        print(f"Couldn't process img: {e}")
        return

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename=path_to_output + "result.jpg")  # save to disk


check_model()
