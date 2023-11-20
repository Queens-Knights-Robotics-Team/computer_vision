import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

while True:
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    # Convert the color image to the HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color
    lower_red = np.array([0, 100, 150])  # Lower bound for red in HSV
    upper_red = np.array([5, 255, 255])  # Upper bound for red in HSV

    # Create a mask for the red color
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    kernel = np.ones((8, 8), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    # Apply inverse binary thresholding
    _, red_mask_inverse = cv2.threshold(red_mask, 1, 255, cv2.THRESH_BINARY_INV)

    # Apply the inverse mask to the original color image
    non_red_color_image = cv2.bitwise_and(color_image, color_image, mask=red_mask_inverse)

    cv2.imshow('rgb', non_red_color_image)
    cv2.imshow('depth', depth_cm)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
cv2.destroyAllWindows()
