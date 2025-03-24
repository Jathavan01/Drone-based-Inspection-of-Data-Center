from ultralytics import YOLO

# Define the path to your dataset YAML configuration file
dataset_yaml = './dataset/custom_dataset.yaml'

# Define the path to the pre-trained weights you want to use (optional)
weights = 'yolo11n.pt'  # You can use a pre-trained model (YOLOv5 small)

# Create a YOLO model object
model = YOLO(weights)

# Start the training process
model.train(
    data="C:\\Users\\Costco Markham East\\Desktop\\pythonProject\\datasets\\data.yaml",  # Path to dataset YAML
    epochs=50,           # Number of epochs
    batch=16,            # Batch size
    imgsz=640,           # Image size
)
