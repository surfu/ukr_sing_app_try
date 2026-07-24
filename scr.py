import cv2 as cv
import mediapipe as mp
import time, math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
latest_result = None
history, sentence = [],[]


def save_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='data/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=save_result
    )
def draw_bar(image, words_list):
    cv.rectangle(image, (0, 0), (image.shape[1], 40), (255, 120, 0), -1)
    text = " ".join(words_list[-5:])
    cv.putText(image, text, (15, 28), 
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
def point(ft, sec, w, h):
    x1,y1 = int(ft.x *w),int(ft.y *h)
    x2,y2 =int(sec.x *w), int(sec.y *h)
    l = math.hypot(x2 - x1, y2 - y1)
    return l
def get_similarity(current_angle, target_angle=110):
    diff = abs(current_angle - target_angle)
    similarity = max(0, 100 - (diff * 2)) 
    return int(similarity)
def c_angle(a, b, c):
    radians_ba = math.atan2(a.y - b.y, a.x - b.x)
    radians_bc = math.atan2(c.y - b.y, c.x - b.x)
    angle_rad = radians_bc - radians_ba
    angle_deg = math.degrees(angle_rad)
    angle_deg = abs(angle_deg)
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg
    return angle_deg
def A(tip, mcp, pip, w, h):
    fingers_down = all(tip[i].y > mcp[i].y for i in range(1, 5))
    fingers_folded = all(point(tip[i], mcp[i], w, h) < 60 for i in range(1, 5))    
    if fingers_down and fingers_folded:
        dist_to_pinky = point(tip[0], pip[4], w, h)
        similarity = max(0, 100 - int(dist_to_pinky * 1.5))
        return True, min(100, similarity)
    return False, 0
def B(tip, mcp, pip, w, h):
    angles = [c_angle(tip[i], pip[i], mcp[i]) for i in range(1, 5)]
    fingers_straight = all(a > 140 for a in angles)
    upward_fingers = all(tip[i].y < mcp[i].y for i in range(1, 5))
    thumb_dist = point(tip[0], mcp[2], w, h)
    thumb_folded = thumb_dist < 60
    dist_index_middle = point(tip[1], tip[2], w, h)
    dist_middle_ring = point(tip[2], tip[3], w, h)
    fingers_together = (dist_index_middle < 35 and dist_middle_ring < 35)
    if fingers_straight and upward_fingers and thumb_folded and fingers_together:
        avg_angle = sum(angles) / len(angles)
        similarity = get_similarity(avg_angle, target_angle=175)
        return True, similarity
    return False, 0
def L(tip, mcp, pip, dip, w, h):
    angle_index = c_angle(tip[1], pip[1], mcp[1])
    angle_middle = c_angle(tip[2], pip[2], mcp[2])
    two_fingers_straight = (angle_index > 130 and angle_middle > 130)
    pointing_down = (tip[1].y > mcp[1].y) and (tip[2].y > mcp[2].y)
    ring_folded = point(tip[3], mcp[3], w, h) < 60
    pinky_folded = point(tip[4], mcp[4], w, h) < 60
    
    thumb_dist = point(tip[0], pip[1], w, h)
    
    if two_fingers_straight and pointing_down and ring_folded and pinky_folded and thumb_dist > 30:
        avg_angle = (angle_index + angle_middle) / 2
        return True, get_similarity(avg_angle, target_angle=175)
    return False, 0
def T(tip, mcp, pip, dip, w, h):
    angle_index  = c_angle(tip[1], pip[1], mcp[1])
    angle_middle = c_angle(tip[2], pip[2], mcp[2])
    angle_ring   = c_angle(tip[3], pip[3], mcp[3])    
    three_fingers_straight = (angle_index > 130 and angle_middle > 130 and angle_ring > 130)
    pointing_down = (tip[1].y > mcp[1].y) and (tip[2].y > mcp[2].y) and (tip[3].y > mcp[3].y)
    pinky_folded = point(tip[4], mcp[4], w, h) < 60
    if three_fingers_straight and pointing_down and pinky_folded:
        avg_angle = (angle_index + angle_middle + angle_ring) / 3
        return True, get_similarity(avg_angle, target_angle=175)
    return False, 0
def N(tip, mcp, pip, dip, w, h):
    index_up  = tip[1].y < mcp[1].y
    middle_up = tip[2].y < mcp[2].y
    pinky_up  = tip[4].y < mcp[4].y
    three_upward = index_up and middle_up and pinky_up
    ring_folded = tip[3].y > pip[3].y or point(tip[3], mcp[3], w, h) < 65
    if three_upward and ring_folded:
        angle_index  = c_angle(tip[1], pip[1], mcp[1])
        angle_middle = c_angle(tip[2], pip[2], mcp[2])
        angle_pinky  = c_angle(tip[4], pip[4], mcp[4])
        avg_angle = (angle_index + angle_middle + angle_pinky) / 3
        return True, get_similarity(avg_angle, target_angle=175)
    return False, 0
def P(tip, mcp, pip, dip, w, h):
    angle_index = c_angle(tip[1], pip[1], mcp[1])
    angle_middle = c_angle(tip[2], pip[2], mcp[2])
    two_fingers_straight = (angle_index > 130 and angle_middle > 130)
    pointing_down = (tip[1].y > mcp[1].y) and (tip[2].y > mcp[2].y)
    ring_folded = point(tip[3], mcp[3], w, h) < 60
    pinky_folded = point(tip[4], mcp[4], w, h) < 60
    thumb_dist = point(tip[0], pip[1], w, h)

    if two_fingers_straight and pointing_down and ring_folded and pinky_folded and thumb_dist <= 30:
        avg_angle = (angle_index + angle_middle) / 2
        return True, get_similarity(avg_angle, target_angle=175)
    return False, 0
def SoftSign(tip, mcp, pip, dip, w, h):
    middle_folded = point(tip[2], mcp[2], w, h) < 55
    ring_folded   = point(tip[3], mcp[3], w, h) < 55
    pinky_folded  = point(tip[4], mcp[4], w, h) < 55    
    if not (middle_folded and ring_folded and pinky_folded):
        return False, 0
    index_dist = point(tip[1], mcp[1], w, h)
    is_hook_dist = 30 < index_dist < 75
    if is_hook_dist:
        return True, 90
    return False, 0
def O(tip, mcp, pip, dip, w, h):
    thumb_index_dist = point(tip[0], tip[1], w, h)
    is_ring_closed = thumb_index_dist < 40
    middle_straight = c_angle(tip[2], pip[2], mcp[2]) > 120
    ring_straight   = c_angle(tip[3], pip[3], mcp[3]) > 120
    pinky_straight  = c_angle(tip[4], pip[4], w, h) > 120 if False else point(tip[4], mcp[4], w, h) > 40
    other_fingers_up = middle_straight and ring_straight
    if is_ring_closed and other_fingers_up:
        similarity = get_similarity(thumb_index_dist, target_angle=10)
        return True, max(70, 100 - int(thumb_index_dist * 2))
    return False, 0
def I(tip,mcp,pip,dip,w,h):
    pinky_angle = c_angle(tip[4], pip[4], mcp[4])
    pinky_straight = pinky_angle > 140
    pinky_upward = tip[4].y < mcp[4].y

    # 2. Перевіряємо, що інші 3 пальці (1, 2, 3) зігнуті в кулак
    index_folded  = point(tip[1], mcp[1], w, h) < 55
    middle_folded = point(tip[2], mcp[2], w, h) < 55
    ring_folded   = point(tip[3], mcp[3], w, h) < 55

    # 3. Великий палець (0) притиснутий до кулака
    thumb_folded  = point(tip[0], mcp[2], w, h) < 60

    if pinky_straight and pinky_upward and index_folded and middle_folded and ring_folded and thumb_folded:
        similarity = get_similarity(pinky_angle, target_angle=175)
        return True, similarity

    return False, 0
def Ya(tip, mcp, pip, dip, w, h):
    angle_index = c_angle(tip[1], pip[1], mcp[1])
    angle_middle = c_angle(tip[2], pip[2], mcp[2])
    two_fingers_straight = (angle_index > 130 and angle_middle > 130)
    fingers_up = (tip[1].y < mcp[1].y) and (tip[2].y < mcp[2].y)
    ring_folded = (tip[3].y > mcp[3].y) or (point(tip[3], mcp[3], w, h) < 50)
    pinky_folded = (tip[4].y > mcp[4].y) or (point(tip[4], mcp[4], w, h) < 50)
    fingers_close = point(tip[1], tip[2], w, h) < 35
    if two_fingers_straight and fingers_up and ring_folded and pinky_folded and fingers_close:
        avg_angle = (angle_index + angle_middle) / 2
        return True, get_similarity(avg_angle, target_angle=175)
    return False, 0

def get_gesture(tip, dip, pip, mcp, w, h):
    ok, sim = T(tip, mcp, pip, dip, w, h)
    if ok and sim > 30:
        return "Т", sim

    ok, sim = P(tip, mcp, pip, dip, w, h)
    if ok and sim > 30:
        return "П", sim

    ok, sim = L(tip, mcp, pip, dip, w, h)
    if ok and sim > 30:
        return "Л", sim

    ok, sim = N(tip, dip, pip, mcp, w, h)
    if ok and sim > 30: 
        return "Н", sim

    ok, sim = B(tip, mcp, pip, w, h)
    if ok and sim > 30:
        return "В", sim

    ok, sim = Ya(tip, mcp, pip, dip, w, h)
    if ok and sim > 30:
        return "Я", sim

    ok, sim = I(tip, mcp, pip, dip, w, h)
    if ok and sim > 30:
        return "І", sim

    ok, sim = A(tip, mcp, pip, w, h)
    if ok and sim > 30:
        return "А", sim

    ok, sim = O(tip, mcp, pip, dip, w, h)
    if ok and sim > 30:
        return "О", sim

    ok, sim = SoftSign(tip, mcp, pip, dip, w, h)
    if ok and sim > 30:
        return "Ь", sim

    return None, 0
def draw(rgb, res):
    global history, sentence
    an_img = rgb.copy()
    if res and res.hand_landmarks:        
        for hand_landmarks in res.hand_landmarks:
            mp_drawing.draw_landmarks(
                an_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            h,w, _ = an_img.shape
            tip = [hand_landmarks[i] for i in (4, 8, 12, 16, 20)]       # Всі кінчики
            dip = [hand_landmarks[i] for i in (3, 7, 11, 15, 19)]       # Верхні суглоби
            pip = [hand_landmarks[i] for i in (2, 6, 10, 14, 18)]       # Середні суглоби
            mcp = [hand_landmarks[i] for i in (1, 5, 9, 13, 17)]        # Нижні суглоби (основа)
            current_gesture , similarity = get_gesture(tip, dip, pip, mcp, w, h)
            if current_gesture is not None:
                cv.putText(an_img, f"{current_gesture.upper()}: {similarity}%", (15, 75), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                history.append(current_gesture)
                if len(history) > 10:
                    history.pop(0)
                if current_gesture != "" and history.count(current_gesture) >= 10:
                    if len(sentence) == 0 or sentence[-1] != current_gesture:
                        sentence.append(current_gesture)
        draw_bar(an_img, sentence)     
    return an_img

cap = cv.VideoCapture(0)
cap.set(4,480)
cap.set(3,360)

def init(opt = options, cap = cap ):
    global latest_result
    with HandLandmarker.create_from_options(opt) as det:
        while cap.isOpened():
            s, frame = cap.read()
            if not s:
                break
            fliped = cv.flip(frame,1)
            display_frame = fliped.copy()
            rgb = cv.cvtColor(fliped, cv.COLOR_BGR2RGB)  
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

            timestamp_ms = int(time.time() * 1000)
            det.detect_async(mp_img, timestamp_ms)

            if latest_result is not None and latest_result.hand_landmarks:
                fliped = draw(rgb, latest_result)
                display_frame = cv.cvtColor(fliped, cv.COLOR_RGB2BGR)
            
            cv.imshow('test', display_frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or cv.getWindowProperty("test", cv.WND_PROP_VISIBLE) < 1:
                break
    cap.release()
    cv.destroyAllWindows()