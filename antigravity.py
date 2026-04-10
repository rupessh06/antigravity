import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
face_det = mp_face.FaceDetection(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

MODE = 'strings'  # or 'elastic'
t = 0

def get_index_tip(hand_landmarks, w, h):
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return int(tip.x * w), int(tip.y * h)

def draw_strings(frame, p1, p2, t, n=8):
    tension = 0.5
    colors = [(126, 179, 255), (167, 139, 250), (52, 211, 153)]
    for s, col in enumerate(colors):
        offset = (s - 1) * 15
        pts = []
        for i in range(n + 1):
            frac = i / n
            mx = int(p1[0] + (p2[0] - p1[0]) * frac)
            my = int(p1[1] + (p2[1] - p1[1]) * frac)
            sag = int(np.sin(np.pi * frac) * (1 - tension) * 60)
            wave = int(np.sin(t * 0.06 + frac * np.pi * 2 + s) * 6 * (1 - tension))
            pts.append((mx, my + sag + wave + offset))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i+1], col, 2, cv2.LINE_AA)
        for pt in pts[1:-1]:
            cv2.circle(frame, pt, 4, (126, 179, 255), -1)

def draw_elastic(frame, p1, p2):
    d = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    stretch = max(0, (d - 180) / 180)
    col = (80, 100, 245) if stretch > 0.5 else (80, 200, 120)
    n = 20
    pts_top, pts_bot = [], []
    width = int(12 + stretch * 20)
    ang = np.arctan2(p2[1]-p1[1], p2[0]-p1[0]) + np.pi/2
    for i in range(n + 1):
        frac = i / n
        mx = p1[0] + (p2[0]-p1[0]) * frac
        my = p1[1] + (p2[1]-p1[1]) * frac
        sag = np.sin(np.pi * frac) * max(0, 40 - stretch * 80)
        perp = np.sin(i / n * np.pi) * width
        pts_top.append((int(mx + np.cos(ang)*perp), int(my + sag + np.sin(ang)*perp)))
        pts_bot.append((int(mx - np.cos(ang)*perp), int(my + sag - np.sin(ang)*perp)))
    poly = np.array(pts_top + pts_bot[::-1], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], col)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.polylines(frame, [poly], True, col, 1, cv2.LINE_AA)
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2 + 30)
    cv2.putText(frame, f"stretch: {int(stretch*100)}%", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face_det.process(rgb)
    if face_res.detections:
        for det in face_res.detections:
            bb = det.location_data.relative_bounding_box
            x1,y1 = int(bb.xmin*w), int(bb.ymin*h)
            x2,y2 = x1+int(bb.width*w), y1+int(bb.height*h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (80,200,120), 1)
            cv2.putText(frame, "FACE", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,200,120), 1)

    hand_res = hands.process(rgb)
    fingertips = []
    if hand_res.multi_hand_landmarks:
        for lm in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(126,179,255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(167,139,250), thickness=2))
            tip = get_index_tip(lm, w, h)
            fingertips.append(tip)
            cv2.circle(frame, tip, 10, (126,179,255), -1)
            cv2.circle(frame, tip, 16, (126,179,255), 1)

    if len(fingertips) == 2:
        if MODE == 'strings':
            draw_strings(frame, fingertips[0], fingertips[1], t)
        else:
            draw_elastic(frame, fingertips[0], fingertips[1])

    cv2.putText(frame, f"MODE: {MODE.upper()} | Press S=strings E=elastic Q=quit", (12,28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (126,179,255), 1)
    cv2.putText(frame, "ANTIGRAVITY v0.1", (12,52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,200,120), 1)

    t += 1
    cv2.imshow("ANTIGRAVITY - Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('s'): MODE = 'strings'
    if key == ord('e'): MODE = 'elastic'

cap.release()
cv2.destroyAllWindows()