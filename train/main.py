import cv2 as cv
import mediapipe as mp
import time
from log import HandDetector

log_det = HandDetector(model_path='data/hand_landmarker.task')

cap = cv.VideoCapture(0)
cap.set(3, 720)
cap.set(4,1280)

with mp.tasks.vision.HandLandmarker.create_from_options(log_det.options) as landmarker:
    while cap.isOpened():
        suc, img = cap.read()
        if not suc:
            break

        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        timestamp_ms = int(time.time() * 1000)
        
        landmarker.detect_async(mp_image, timestamp_ms)
        
        if log_det.latest_result is not None:
            annotated_rgb = log_det.draw_landmarks_on_image(rgb_img, log_det.latest_result)
            img = cv.cvtColor(annotated_rgb, cv.COLOR_RGB2BGR)

        cv.imshow("camera", img)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or cv.getWindowProperty("camera", cv.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv.destroyAllWindows()