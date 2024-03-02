import pyrealsense2 as rs
import numpy as np
import cv2
import torch

# Initialize the YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'custom', path='best.pt')  # Ensure 'best.pt' is the correct path to your model

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

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

        # Apply colormap on depth image (optional, if you want to display depth information)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally (optional, if you want to display depth information)
        # images = np.hstack((color_image, depth_colormap))

        # Perform inference
        results = model(color_image)
        
        # Render results on the color image
        for *box, conf, cls in results.xyxy[0]:  # xyxy, conf, cls
            label = f'{results.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(color_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(color_image, label, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Show images
        cv2.imshow('RealSense', color_image)  # Updated to show only color_image with detections
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
