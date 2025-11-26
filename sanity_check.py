# sanity_check.py
import cv2

video_path = "data/input/reel.mp4"

cap = cv2.VideoCapture(video_path)  # type: ignore
print("Opened:", cap.isOpened())

ret, frame = cap.read()
print("First frame read ok?:", ret)

if ret:
    print("Frame shape:", frame.shape)

cap.release()
