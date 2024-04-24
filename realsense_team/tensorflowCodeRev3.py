import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

W = 848
H = 480

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)


print("[INFO] start streaming...")
pipeline.start(config)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

print("[INFO] loading model...")
#PATH_TO_CKPT = "/realsense_team/models/frozen_inference_graph.pb"
#PATH_TO_CKPT = "/home/daniel/QKRT2024/computer_vision/realsense_team/models/frozen_inference_graph.pb"


# Load the Tensorflow model into memory.
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)


# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/

while True:
    frames = pipeline.wait_for_frames()
    frames = aligned_stream.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    points = point_cloud.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    scaled_size = (int(W), int(H))
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image_expanded = np.expand_dims(color_image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_expanded})

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    #print("[INFO] drawing bounding box on detected objects...")
    #print("[INFO] each detected object has a unique color")

    for idx in range(int(num[0])):
        class_ = classes[idx]
        score = scores[idx]
        box = boxes[idx]
        #print(" [DEBUG] class : ", class_, "idx : ", idx, "num : ", num)

        if score > 0.8 and class_ == 1: # 1 for human
            left = box[1] * W   # Normalize left boundary to pixel position
            top = box[0] * H    # Normalize top boundary to pixel position
            right = box[3] * W  # Normalize right boundary to pixel position
            bottom = box[2] * H # Normalize bottom boundary to pixel position

            width = right - left
            height = bottom - top
            bbox = (int(left), int(top), int(width), int(height))

            #p1 = (int(left), int(top))       # Top-left corner
            #p2 = (int(right), int(bottom))   # Bottom-right corner
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))


            # draw box
            cv2.rectangle(color_image, p1, p2, (255,0,0), 2, 1)
            # Insert the new code snippet here
            points_3d = []
            step = 2

            for y in range(int(top), int(bottom), step):
                for x in range(int(left), int(right), step):
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:  # Valid depth data
                        point_3d = rs.rs2_deproject_pixel_to_point(
                            depth_frame.profile.as_video_stream_profile().intrinsics, [x, y], depth)
                        points_3d.append(point_3d)

            if points_3d:
                points_3d = np.array(points_3d)
                center_3d = np.mean(points_3d, axis=0) #this is the actual 3d center in real world
                center_2d = rs.rs2_project_point_to_pixel(
                    depth_frame.profile.as_video_stream_profile().intrinsics, center_3d)
                center_2d = (int(center_2d[0]), int(center_2d[1]))
                cv2.circle(color_image, center_2d, 5, (0, 255, 0), -1)  # Draw center

                # Optional: Display 3D center coordinates
                cv2.putText(color_image, "3D Center: {:.2f}, {:.2f}, {:.2f}".format(*center_3d),
                            (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.waitKey(1)

# Stop streaming
pipeline.stop()