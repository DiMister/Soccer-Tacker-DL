# How tracking works

- YOLO Model (Detection): First, the YOLO neural network processes the frame to find all objects, outputting bounding boxes and confidence scores. It has no memory of the past.
- Tracker (Prediction): Simultaneously, the tracking algorithm (using a Kalman Filter) predicts where it expects the objects from the previous frame to be in the current frame.
- Tracker (Association): Finally, the tracker takes the brand-new detections from YOLO and matches them to its predictions to assign the correct IDs.

The tracking algorithm tries to remeber objects even if they disapear of the screen or are lost briefly. It will contiune to predict the objects movement for 30 frames (by default can be changed) to see if it shows up again before forgetting the object.

## Args
- persist (bool = True): Crucial for tracking. This tells the model that the images are part of a continuous sequence (a video). It ensures the tracker keeps its memory active between frames to link past detections to current ones and assign consistent IDs.
- conf (float): The minimum confidence threshold for object detection (set by CONF, default 0.25). If the YOLO model is less than 25% sure it sees a valid object, the detection is discarded before the tracker even sees it.
- iou (float): The Intersection over Union (IoU) threshold used for Non-Maximum Suppression (NMS) (set by IOU, default 0.5). If the model draws multiple overlapping bounding boxes around the exact same object, this setting helps filter out the duplicates.