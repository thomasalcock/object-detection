import cv2 as cv
import numpy as np
from cvlib.object_detection import draw_bbox, detect_common_objects

HEIGHT = 960
WIDTH = 1280

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# this is slow af for some reason
#cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
#cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # this results in a wayyyy faster feed
    frame = cv.resize(frame, (WIDTH, HEIGHT))
    bbox, label, conf = detect_common_objects(frame, confidence=0.25, model="yolov4-tiny")
    frame = draw_bbox(frame, bbox, label, conf)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
