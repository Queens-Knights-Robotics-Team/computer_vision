import pyrealsense2 as rs
import numpy as np
import cv2
import torch

# Load the TorchScript model
model = torch.jit.load('realsense_team/best.torchscript')

# Verify that the model is in evaluation mode
model.eval()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Use the device manager to check if the device supports the desired configuration
try:
    # This will throw if the device does not support the configuration
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
except Exception as e:
    print(f"An error occurred while configuring the stream: {e}")
    exit()

cam = cv2.VideoCapture(1)

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

        # Apply colormap on depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Convert the color image to the format expected by the model
        input_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = torch.from_numpy(input_image).unsqueeze(0)

        # Perform inference using the model
        try:
            with torch.no_grad():
                output = model(input_image)
        except Exception as e:
            print(f"Error during model inference: {e}")
            output = None

        # Print output shape if inference was successful
        if output is not None:
            print("Output Shape:", output.shape)
        else:
            print("Model inference failed, check for errors above.")

        # Process the model output as needed
        # For example, you can print the output or use it to make decisions

        # Show images
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        # Check for the 'q' key
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

