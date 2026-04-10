import cv2
import mediapipe as mp
import numpy as np
import time

# ── MediaPipe setup ──────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_detection
mp_draw  = mp.solutions.drawing_utils

hands    = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)
face_det = mp_face.FaceDetection(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── Finger tip landmark IDs ──────────────────────────────────────
FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]
FINGER_BASES = [
    mp_hands.HandLandmark.THUMB_IP,
    mp_hands.HandLandmark.INDEX_FINGER_MCP,
    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    mp_hands.HandLandmark.RING_FINGER_MCP,
    mp_hands.HandLandmark.PINKY_MCP,
]
FINGER_NAMES = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']

# ── Colors (BGR) ─────────────────────────────────────────────────
COLS = [
    (255, 180,  80),   # thumb   – amber
    (255, 100, 130),   # index   – pink
    ( 80, 200, 255),   # middle  – cyan
    (130, 255, 130),   # ring    – green
    (200, 130, 255),   # pinky   – purple
]

# ── Physics string nodes ─────────────────────────────────────────
class StringNode:
    def __init__(self, x, y):
        self.x  = float(x)
        self.y  = float(y)
        self.px = float(x)   # previous position (Verlet)
        self.py = float(y)
        self.pinned = False

GRAVITY   = 0.4
DAMPING   = 0.96
N_NODES   = 10          # nodes per string

class PhysicsString:
    def __init__(self, n=N_NODES):
        self.nodes = [StringNode(0, 0) for _ in range(n)]

    def update(self, p1, p2):
        n = len(self.nodes)
        # pin endpoints to fingertips
        self.nodes[0].x,  self.nodes[0].y  = p1
        self.nodes[-1].x, self.nodes[-1].y = p2

        for i in range(1, n - 1):
            nd = self.nodes[i]
            vx = (nd.x - nd.px) * DAMPING
            vy = (nd.y - nd.py) * DAMPING
            nd.px, nd.py = nd.x, nd.y
            nd.x += vx
            nd.y += vy + GRAVITY

        # constraint iterations
        seg_len = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / (n - 1)
        for _ in range(8):
            self.nodes[0].x, self.nodes[0].y  = p1
            self.nodes[-1].x, self.nodes[-1].y = p2
            for i in range(n - 1):
                a, b = self.nodes[i], self.nodes[i+1]
                dx = b.x - a.x
                dy = b.y - a.y
                d  = max(np.sqrt(dx*dx + dy*dy), 0.001)
                diff = (seg_len - d) / d * 0.5
                if i > 0:
                    a.x -= dx * diff
                    a.y -= dy * diff
                if i < n - 2:
                    b.x += dx * diff
                    b.y += dy * diff

    def draw(self, frame, col, thickness=2, glow=True):
        pts = [(int(nd.x), int(nd.y)) for nd in self.nodes]
        # glow pass
        if glow:
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1], col, thickness+4, cv2.LINE_AA)
        # main line
        bright = tuple(min(255, int(c*1.4)) for c in col)
        for i in range(len(pts)-1):
            cv2.line(frame, pts[i], pts[i+1], bright, thickness, cv2.LINE_AA)
        # nodes
        for pt in pts[1:-1]:
            cv2.circle(frame, pt, 4, bright, -1, cv2.LINE_AA)
            if glow:
                cv2.circle(frame, pt, 7, col, 1, cv2.LINE_AA)

# ── Gesture detector ─────────────────────────────────────────────
def get_landmarks(lm, w, h):
    return [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h))
            for i in range(21)]

def finger_up(lm, tip_id, base_id, w, h):
    ty = lm.landmark[tip_id].y
    by = lm.landmark[base_id].y
    return ty < by

def detect_gesture(lm, w, h):
    up = [finger_up(lm, FINGER_TIPS[i], FINGER_BASES[i], w, h)
          for i in range(5)]
    n_up = sum(up)

    # pinch: thumb + index close
    tx = lm.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w
    ty = lm.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h
    ix = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
    iy = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
    pinch_dist = np.sqrt((tx-ix)**2 + (ty-iy)**2)

    if pinch_dist < 40:
        return 'PINCH', up, pinch_dist
    if n_up == 0:
        return 'FIST', up, pinch_dist
    if n_up == 5:
        return 'OPEN', up, pinch_dist
    if up[1] and not up[2] and not up[3] and not up[4]:
        return 'POINT', up, pinch_dist
    if up[1] and up[2] and not up[3] and not up[4]:
        return 'PEACE', up, pinch_dist
    return 'CUSTOM', up, pinch_dist

# ── Gesture action state ──────────────────────────────────────────
gesture_log  = []          # last N gesture events
snap_anim    = []          # list of snap particle bursts
fist_release = False

def add_snap(x, y, col):
    for _ in range(18):
        ang = np.random.uniform(0, 2*np.pi)
        spd = np.random.uniform(3, 10)
        snap_anim.append({
            'x': float(x), 'y': float(y),
            'vx': np.cos(ang)*spd, 'vy': np.sin(ang)*spd,
            'life': 1.0, 'col': col
        })

def update_snap(frame):
    dead = []
    for p in snap_anim:
        p['x']  += p['vx']
        p['y']  += p['vy']
        p['vy'] += 0.3
        p['vx'] *= 0.93
        p['life'] -= 0.04
        if p['life'] <= 0:
            dead.append(p)
            continue
        c  = tuple(int(v * p['life']) for v in p['col'])
        r  = max(1, int(4 * p['life']))
        cv2.circle(frame, (int(p['x']), int(p['y'])), r, c, -1, cv2.LINE_AA)
    for d in dead:
        snap_anim.remove(d)

# ── HUD overlay ───────────────────────────────────────────────────
def draw_hud(frame, gestures, fps, h, w):
    # top-left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 160), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (10, 10), (320, 160), (80, 179, 126), 1)

    cv2.putText(frame, 'ANTIGRAVITY  v1.0', (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 120), 1)
    cv2.putText(frame, f'FPS: {fps:.0f}', (240, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1)

    labels = {'OPEN':  (80, 255, 130),
              'FIST':  (80, 130, 255),
              'PINCH': (255, 180, 80),
              'POINT': (80, 200, 255),
              'PEACE': (200, 130, 255),
              'CUSTOM':(180, 180, 180)}

    for idx, (hand_label, g) in enumerate(gestures.items()):
        name, up, pd = g
        col = labels.get(name, (200, 200, 200))
        y = 65 + idx * 45
        cv2.putText(frame, f'{hand_label}: {name}', (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        finger_str = ' '.join(['|' if u else '.' for u in up])
        cv2.putText(frame, finger_str, (20, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # bottom instruction bar
    cv2.rectangle(frame, (0, h-36), (w, h), (10,10,20), -1)
    cv2.putText(frame, 'Gestures: OPEN=web  FIST+release=snap  PINCH=pull  PEACE=split  Q=quit',
                (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,200), 1)

# ── Main loop ─────────────────────────────────────────────────────
strings   = [PhysicsString() for _ in range(5)]
prev_gestures = {}
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # dark tint for overlay feel
    dark = np.zeros_like(frame)
    cv2.addWeighted(frame, 0.75, dark, 0.25, 0, frame)

    # ── Face detection ────────────────────────────────────────────
    face_res = face_det.process(rgb)
    if face_res.detections:
        for det in face_res.detections:
            bb = det.location_data.relative_bounding_box
            x1 = int(bb.xmin * W);       y1 = int(bb.ymin * H)
            x2 = x1 + int(bb.width * W); y2 = y1 + int(bb.height * H)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (80,220,120), 1, cv2.LINE_AA)
            cv2.putText(frame, 'FACE', (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,220,120), 1)

    # ── Hand detection ────────────────────────────────────────────
    hand_res  = hands.process(rgb)
    hand_data = []   # list of (label, landmarks, gesture, tips)
    gestures_hud = {}

    if hand_res.multi_hand_landmarks:
        for idx, lm in enumerate(hand_res.multi_hand_landmarks):
            label = 'LEFT' if idx == 0 else 'RIGHT'

            # skeleton
            mp_draw.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=COLS[idx*2 % 5], thickness=1, circle_radius=3),
                mp_draw.DrawingSpec(color=(80,80,120), thickness=1))

            gesture, up, pd = detect_gesture(lm, W, H)
            tips = [(int(lm.landmark[FINGER_TIPS[f]].x * W),
                     int(lm.landmark[FINGER_TIPS[f]].y * H))
                    for f in range(5)]

            hand_data.append((label, lm, gesture, tips, up, pd))
            gestures_hud[label] = (gesture, up, pd)

            # highlight fingertips
            for fi, tip in enumerate(tips):
                col = COLS[fi]
                cv2.circle(frame, tip, 8,  col, -1,  cv2.LINE_AA)
                cv2.circle(frame, tip, 12, col,  1,  cv2.LINE_AA)

    # ── String physics between TWO hands ─────────────────────────
    if len(hand_data) == 2:
        h0_label, lm0, g0, tips0, up0, pd0 = hand_data[0]
        h1_label, lm1, g1, tips1, up1, pd1 = hand_data[1]

        # determine active fingers: fingers that are UP on BOTH hands
        for fi in range(5):
            active = up0[fi] and up1[fi]

            if g0 == 'FIST' or g1 == 'FIST':
                active = False   # collapse all strings on fist

            if active:
                strings[fi].update(tips0[fi], tips1[fi])
                strings[fi].draw(frame, COLS[fi], thickness=2, glow=True)
            else:
                # reset node positions to midpoint so they respawn clean
                mid = ((tips0[fi][0]+tips1[fi][0])//2,
                       (tips0[fi][1]+tips1[fi][1])//2)
                for nd in strings[fi].nodes:
                    nd.x = nd.px = mid[0]
                    nd.y = nd.py = mid[1]

        # ── FIST release → snap burst ─────────────────────────────
        both_fist = g0 == 'FIST' and g1 == 'FIST'
        was_fist  = prev_gestures.get('both_fist', False)
        if was_fist and not both_fist:
            cx = (tips0[1][0] + tips1[1][0]) // 2
            cy = (tips0[1][1] + tips1[1][1]) // 2
            for fi in range(5):
                add_snap(tips0[fi][0], tips0[fi][1], COLS[fi])
                add_snap(tips1[fi][0], tips1[fi][1], COLS[fi])
        prev_gestures['both_fist'] = both_fist

        # ── PINCH = pull string toward fingertip ──────────────────
        for hi, (label, lm, g, tips, up, pd) in enumerate(hand_data):
            if g == 'PINCH':
                px = int(lm.landmark[mp_hands.HandLandmark.THUMB_TIP].x * W)
                py = int(lm.landmark[mp_hands.HandLandmark.THUMB_TIP].y * H)
                for fi in range(5):
                    # attract middle nodes toward pinch point
                    for nd in strings[fi].nodes[2:-2]:
                        dx = px - nd.x
                        dy = py - nd.y
                        nd.x += dx * 0.18
                        nd.y += dy * 0.18

        # ── OPEN gesture: web between same-hand fingers ───────────
        for label, lm, g, tips, up, pd in hand_data:
            if g == 'OPEN':
                for fi in range(4):
                    p1 = tips[fi]
                    p2 = tips[fi+1]
                    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2 - 20)
                    col = COLS[fi]
                    pts = []
                    for t in range(11):
                        frac = t / 10
                        mx = int(p1[0] + (p2[0]-p1[0])*frac)
                        my = int(p1[1] + (p2[1]-p1[1])*frac)
                        sag = int(np.sin(np.pi*frac) * 30)
                        pts.append((mx, my - sag))
                    for i in range(len(pts)-1):
                        cv2.line(frame, pts[i], pts[i+1], col, 1, cv2.LINE_AA)
                    # fill with translucent web
                    poly = np.array([p1] + pts + [p2], np.int32)
                    ov   = frame.copy()
                    cv2.fillPoly(ov, [poly], col)
                    cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    # ── Snap particle update ──────────────────────────────────────
    update_snap(frame)

    # ── HUD ───────────────────────────────────────────────────────
    now = time.time()
    fps = 1.0 / max(now - prev_time, 0.001)
    prev_time = now
    draw_hud(frame, gestures_hud, fps, H, W)

    cv2.imshow('ANTIGRAVITY v1.0 — Hand Tracking', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()