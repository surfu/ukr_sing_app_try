import cv2 as cv
import mediapipe as mp
import time

# Класи та утиліти MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (0, 0, 0) 

class HandDetector:
    def __init__(self, model_path='data/hand_landmarker.task'):
        self.latest_result = None
        
        # Налаштування опцій
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self._print_result
        )

    def _print_result(self, result, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        annotated_image = rgb_image.copy()
        
        if detection_result and detection_result.hand_landmarks:
            hand_landmarks_list = detection_result.hand_landmarks
            handedness_list = detection_result.handedness

            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Малюємо скелет рук
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Визначаємо координати для тексту (ліва/права рука)
                height, width, _ = annotated_image.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN

                cv.putText(annotated_image, f"{handedness[0].category_name}",
                            (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

                index_tip = hand_landmarks[8]
                finger_x = int(index_tip.x * width)
                finger_y = int(index_tip.y * height)
                if finger_x < 300 and finger_y < 300:
                    cv.putText(annotated_image, "ZONE A ACTIVE!", (50, 50), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                
                thumb_tip = hand_landmarks[12]
                thumb_mcp = hand_landmarks[9]
                index_mcp = hand_landmarks[5]
                middle_tip = hand_landmarks[16]
                middle_mcp = hand_landmarks[13]

                is_thumb_up = thumb_tip.y < thumb_mcp.y
                are_other_fingers_folded = (index_tip.y > index_mcp.y) and (middle_tip.y > middle_mcp.y)
                if is_thumb_up and are_other_fingers_folded:
                    cv.putText(annotated_image, "F*CK!", (50, 80), 
                               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)
                

        return annotated_image