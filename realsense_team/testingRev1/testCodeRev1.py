import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from torchvision import transforms

# C:\Users\danie\iCloudDrive\Repos\computer_vision\realsense_team\testingRev1\best.pt
model_path = 'realsense_team/testingRev1/best.pt'  # Specify the correct path to your model file
model = torch.load(model_path)  # Load your model using the correct path

# Example transformation. Adjust according to your model's requirements.
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),  # Adjust size according to your model's requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Define a function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        # Extract box coordinates
        x1, y1, x2, y2 = box

        # Draw bounding box rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Annotate with label and confidence score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Perform post-processing for object detection
def post_process(output):
    # Assuming output contains bounding boxes, class labels, and confidence scores
    boxes = output['boxes']
    labels = output['labels']
    scores = output['scores']

    # Convert tensor results to numpy arrays
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    return boxes, labels, scores

# Move the model to GPU for speed if available
if torch.cuda.is_available():
    model.to('cuda')

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply preprocessing to the color image
        input_tensor = preprocess(color_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Move input batch to GPU if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

# Perform inference
        with torch.no_grad():
            output = model(input_batch)

        # Perform post-processing
        boxes, labels, scores = post_process(output)


        # Draw bounding boxes on the original color image
        draw_boxes(color_image, boxes, labels, scores)

        # Show images
        cv2.imshow('Object Detection', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
