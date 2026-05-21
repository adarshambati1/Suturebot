"""
Live RGB feed from an Intel RealSense with OpenCV bounding boxes around
strongly red and strongly blue regions.

Useful for sanity-checking that the camera sees what you think it sees,
and as a placeholder perception primitive (e.g. mark the needle tip red,
the entry point blue, watch the boxes move).

Run:
    python3 perception/color_bbox_live.py

Press 'q' in the OpenCV window to quit.

Deps:
    pip install pyrealsense2 opencv-python numpy
"""

import cv2
import numpy as np
import pyrealsense2 as rs


# HSV color ranges (OpenCV: H in [0, 179], S/V in [0, 255]).
# Tune these to your lighting + the actual red/blue you use.
RED_RANGES = [
    (np.array([0,   120, 80]),  np.array([10,  255, 255])),   # red wraps around hue 0
    (np.array([170, 120, 80]),  np.array([179, 255, 255])),
]
BLUE_RANGES = [
    (np.array([100, 120, 80]),  np.array([130, 255, 255])),
]

MIN_AREA_PX = 500    # ignore blobs smaller than this


def mask_for(hsv_frame, ranges):
    """OR-combine inRange masks across all (low, high) tuples."""
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv_frame, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    # Clean up speckle and fill small holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def boxes_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_AREA_PX]


def draw_boxes(frame, boxes, color, label):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("RealSense started. Press 'q' in the window to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            red_boxes  = boxes_from_mask(mask_for(hsv, RED_RANGES))
            blue_boxes = boxes_from_mask(mask_for(hsv, BLUE_RANGES))

            draw_boxes(frame, red_boxes,  (0, 0, 255), f"red x{len(red_boxes)}")
            draw_boxes(frame, blue_boxes, (255, 0, 0), f"blue x{len(blue_boxes)}")

            cv2.imshow("RealSense - red/blue bboxes (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
