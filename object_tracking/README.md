# How tracking works

- YOLO Model (Detection): First, the YOLO neural network processes the frame to find all objects, outputting bounding boxes and confidence scores. It has no memory of the past.
- Tracker (Prediction): Simultaneously, the tracking algorithm (using a Kalman Filter) predicts where it expects the objects from the previous frame to be in the current frame.
- Tracker (Association): Finally, the tracker takes the brand-new detections from YOLO and matches them to its predictions to assign the correct IDs.

The tracking algorithm tries to remeber objects even if they disapear of the screen or are lost briefly. It will contiune to predict the objects movement for 30 frames (by default can be changed) to see if it shows up again before forgetting the object.

## Args

- persist (bool = True): Crucial for tracking. This tells the model that the images are part of a continuous sequence (a video). It ensures the tracker keeps its memory active between frames to link past detections to current ones and assign consistent IDs.
- conf (float): The minimum confidence threshold for object detection (set by CONF, default 0.25). If the YOLO model is less than 25% sure it sees a valid object, the detection is discarded before the tracker even sees it.
- iou (float): The Intersection over Union (IoU) threshold used for Non-Maximum Suppression (NMS) (set by IOU, default 0.5). If the model draws multiple overlapping bounding boxes around the exact same object, this setting helps filter out the duplicates.

## HOTA evaluator outputs

When running `object_tracking/HotaEvaluator.py`, you will see output like this:

```
=== HOTA Evaluation (Validation Set) ===
Model: fine_tuned_models/mosaic_tuned_yolo/weights/best.pt
Tracker: botsort.yaml
Sequences evaluated: N
HOTA: 0.xxxx
DetA: 0.xxxx
AssA: 0.xxxx
Prediction files: runs/tracking/hota_eval/<model_name>/labels
```

### What each value means

- HOTA: Main tracking quality score (higher is better). It balances detection quality and association quality.
- DetA: Detection Accuracy. Measures how well objects are detected frame-by-frame (true positives vs false positives and false negatives).
- AssA: Association Accuracy. Measures how consistently track IDs stay matched to the same object over time.
- Sequences evaluated: Number of validation sequences that had valid ground-truth files and were included in the evaluation.

### Prediction files

- A `.txt` file is written per evaluated sequence in MOT format.
- Location: `runs/tracking/hota_eval/<model_name>/labels/`
- Each line has: `frame,track_id,x,y,w,h,score,-1,-1,-1`
