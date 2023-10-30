import pyrealsense2 as rs
import numpy as np
import cv2

def red_mask(hsv_image):
    lower_red = np.array([0, 100, 150])
    upper_red = np.array([5, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    kernel = np.ones((9, 9), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    return red_mask

def blue_mask(hsv_image):
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([100, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    kernel = np.ones((9, 9), np.uint8)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
    return blue_mask

# Function to find and draw a bounding box around the red object
def find_and_draw_object(image, colour_of_leds):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if colour_of_leds == 'blue':
        colour_mask = blue_mask(hsv_image)
    if colour_of_leds == 'red':
        colour_mask = red_mask(hsv_image)


    # Find contours in the mask
    contours, _ = cv2.findContours(colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw a bounding box around the largest red object (if found)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green bounding box
    return colour_mask
# Create the RealSense pipeline and start it
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

    colour_mask = find_and_draw_object(color_image, "blue")  # Call the function to find and draw the bounding box
    masked_color_image = cv2.bitwise_and(color_image, color_image, mask=colour_mask)

    cv2.imshow('rgb', color_image)
    cv2.imshow('depth', depth_cm)
    cv2.imshow('rgb', masked_color_image)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
cv2.destroyAllWindows()