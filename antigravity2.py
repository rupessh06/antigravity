import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ── MediaPipe setup ──────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_face     = mp.solutions.face_detection
mp_face_mesh= mp.solutions.face_mesh
mp_draw     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

hands     = mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.6)
face_det  = mp_face.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.6,
                                   min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── Finger tip IDs ───────────────────────────────────────────────
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]
FINGER_NAMES = ['THUMB','INDEX','MIDDLE','RING','PINKY']
COLS_BGR = [
    ( 80, 180, 255),
    (130, 100, 255),
    ( 80, 220, 180),
    ( 80, 255, 130),
    (200, 130, 255),
]

# ── Face mesh landmark indices ───────────────────────────────────
# Eyes
L_EYE_TOP, L_EYE_BOT = 159, 145
L_EYE_L,   L_EYE_R   = 33,  133
R_EYE_TOP, R_EYE_BOT = 386, 374
R_EYE_L,   R_EYE_R   = 362, 263
# Iris (refine_landmarks=True gives these)
L_IRIS_CENTER = 468
R_IRIS_CENTER = 473
# Mouth
MOUTH_TOP, MOUTH_BOT  = 13, 14
MOUTH_L,   MOUTH_R    = 61, 291
# Nose tip
NOSE_TIP = 1
# Head pose reference points
HEAD_POSE_IDS = [1, 33, 263, 61, 291, 199]

# ── Smoothing buffers ────────────────────────────────────────────
class Smoother:
    def __init__(self, size=8):
        self.buf  = []
        self.size = size
    def update(self, val):
        self.buf.append(val)
        if len(self.buf) > self.size:
            self.buf.pop(0)
        return sum(self.buf) / len(self.buf)

blink_smooth   = Smoother(6)
mouth_smooth   = Smoother(6)
gaze_x_smooth  = Smoother(10)
gaze_y_smooth  = Smoother(10)
attention_smooth = Smoother(20)

# ── State tracking ───────────────────────────────────────────────
blink_count    = 0
blink_history  = []   # timestamps
emotion_history= []
prev_eye_closed= False
session_start  = time.time()
prev_time      = time.time()
fps_history    = []

# ── Eye aspect ratio (blink detection) ──────────────────────────
def eye_aspect_ratio(top, bot, left, right):
    h = math.dist(top, bot)
    w = math.dist(left, right)
    return h / (w + 1e-6)

# ── Gaze direction ───────────────────────────────────────────────
def get_gaze(iris_pt, eye_l, eye_r, eye_top, eye_bot):
    eye_w = math.dist(eye_l, eye_r)
    eye_h = math.dist(eye_top, eye_bot)
    if eye_w < 1 or eye_h < 1:
        return 0.0, 0.0
    gx = (iris_pt[0] - eye_l[0]) / eye_w - 0.5
    gy = (iris_pt[1] - eye_top[1]) / eye_h - 0.5
    return gx, gy

# ── Head pose estimation ─────────────────────────────────────────
def get_head_pose(lm_list, W, H):
    model_pts = np.array([
        (0.0,    0.0,    0.0),
        (-30.0, -30.0, -30.0),
        ( 30.0, -30.0, -30.0),
        (-25.0,  20.0, -20.0),
        ( 25.0,  20.0, -20.0),
        (0.0,   50.0, -30.0),
    ], dtype=np.float64)
    img_pts = np.array([lm_list[i] for i in HEAD_POSE_IDS], dtype=np.float64)
    focal   = W
    cam_mat = np.array([[focal,0,W/2],[0,focal,H/2],[0,0,1]], dtype=np.float64)
    dist_co = np.zeros((4,1))
    ok, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam_mat, dist_co,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    pitch = math.degrees(math.atan2(-rmat[2,0], sy))
    yaw   = math.degrees(math.atan2( rmat[2,1], rmat[2,2]))
    roll  = math.degrees(math.atan2( rmat[1,0], rmat[0,0]))
    return pitch, yaw, roll

# ── Emotion estimation (geometry-based) ─────────────────────────
def estimate_emotion(lm, W, H):
    def pt(i): return (lm.landmark[i].x*W, lm.landmark[i].y*H)
    mouth_w  = math.dist(pt(61),  pt(291))
    mouth_h  = math.dist(pt(13),  pt(14))
    brow_l   = pt(65)[1]  - pt(159)[1]
    brow_r   = pt(295)[1] - pt(386)[1]
    brow_avg = (brow_l + brow_r) / 2
    eye_open_l = math.dist(pt(159), pt(145))
    eye_open_r = math.dist(pt(386), pt(374))
    eye_avg    = (eye_open_l + eye_open_r) / 2
    mouth_ratio = mouth_h / (mouth_w + 1e-6)

    if mouth_ratio > 0.35 and eye_avg > 12:
        return 'SURPRISED', (80,200,255)
    if mouth_w > 55 and mouth_ratio > 0.08:
        return 'HAPPY', (80,255,130)
    if brow_avg < -5:
        return 'ANGRY', (80,80,255)
    if mouth_ratio < 0.02 and brow_avg > 3:
        return 'SAD', (200,130,130)
    if eye_avg < 6:
        return 'SLEEPY', (180,130,80)
    return 'NEUTRAL', (180,180,180)

# ── Age / Gender (heuristic from face size + proportions) ────────
# NOTE: True age/gender needs a deep learning model.
# This uses geometric proxies as a placeholder that updates live.
def estimate_age_gender(bb, W, H):
    face_area = bb.width * W * bb.height * H
    # placeholder bands — replace with cv2.dnn model for accuracy
    if face_area > 80000:
        age_band = '20-30'
    elif face_area > 50000:
        age_band = '25-35'
    else:
        age_band = '18-40'
    return age_band, 'DETECTING'

# ── Attention score ──────────────────────────────────────────────
def calc_attention(pitch, yaw, blink_rate, mouth_open):
    yaw_pen   = min(abs(yaw)   / 30.0, 1.0)
    pitch_pen = min(abs(pitch) / 25.0, 1.0)
    blink_pen = min(blink_rate / 25.0, 1.0)
    mouth_pen = 0.3 if mouth_open else 0.0
    score = 100 - (yaw_pen*40 + pitch_pen*20 + blink_pen*20 + mouth_pen*20)
    return max(0, min(100, score))

# ── Physics string (same as v1) ──────────────────────────────────
GRAVITY = 0.35
DAMPING = 0.96

class StringNode:
    def __init__(self, x, y):
        self.x=float(x); self.y=float(y)
        self.px=float(x); self.py=float(y)

class PhysicsString:
    def __init__(self, n=12):
        self.nodes=[StringNode(0,0) for _ in range(n)]
    def update(self, p1, p2):
        n=len(self.nodes)
        self.nodes[0].x,self.nodes[0].y=p1
        self.nodes[-1].x,self.nodes[-1].y=p2
        for i in range(1,n-1):
            nd=self.nodes[i]
            vx=(nd.x-nd.px)*DAMPING; vy=(nd.y-nd.py)*DAMPING
            nd.px,nd.py=nd.x,nd.y
            nd.x+=vx; nd.y+=vy+GRAVITY
        seg=math.dist(p1,p2)/(n-1)
        for _ in range(10):
            self.nodes[0].x,self.nodes[0].y=p1
            self.nodes[-1].x,self.nodes[-1].y=p2
            for i in range(n-1):
                a,b=self.nodes[i],self.nodes[i+1]
                dx=b.x-a.x; dy=b.y-a.y
                d=max(math.sqrt(dx*dx+dy*dy),0.001)
                diff=(seg-d)/d*0.5
                if i>0:   a.x-=dx*diff; a.y-=dy*diff
                if i<n-2: b.x+=dx*diff; b.y+=dy*diff
    def draw(self, frame, col, thick=2):
        pts=[(int(nd.x),int(nd.y)) for nd in self.nodes]
        for i in range(len(pts)-1):
            cv2.line(frame,pts[i],pts[i+1],col,thick+3,cv2.LINE_AA)
        bright=tuple(min(255,int(c*1.5)) for c in col)
        for i in range(len(pts)-1):
            cv2.line(frame,pts[i],pts[i+1],bright,thick,cv2.LINE_AA)
        for pt in pts[1:-1]:
            cv2.circle(frame,pt,4,bright,-1,cv2.LINE_AA)

# ── Gesture detection ─────────────────────────────────────────────
def finger_up(lm, tip, base):
    return lm.landmark[tip].y < lm.landmark[base].y

def detect_gesture(lm, W, H):
    up=[finger_up(lm,FINGER_TIPS[i],FINGER_BASES[i]) for i in range(5)]
    tx=lm.landmark[4].x*W; ty=lm.landmark[4].y*H
    ix=lm.landmark[8].x*W; iy=lm.landmark[8].y*H
    pd=math.sqrt((tx-ix)**2+(ty-iy)**2)
    if pd<40:          return 'PINCH',up,pd
    if sum(up)==0:     return 'FIST',up,pd
    if sum(up)==5:     return 'OPEN',up,pd
    if up[1] and not any([up[2],up[3],up[4]]): return 'POINT',up,pd
    if up[1] and up[2] and not any([up[3],up[4]]): return 'PEACE',up,pd
    return 'CUSTOM',up,pd

# ── Snap particles ───────────────────────────────────────────────
snap_anim=[]
def add_snap(x,y,col):
    for _ in range(14):
        a=np.random.uniform(0,2*math.pi); s=np.random.uniform(4,12)
        snap_anim.append({'x':float(x),'y':float(y),
                          'vx':math.cos(a)*s,'vy':math.sin(a)*s,
                          'life':1.0,'col':col})
def update_snap(frame):
    dead=[]
    for p in snap_anim:
        p['x']+=p['vx']; p['y']+=p['vy']
        p['vy']+=0.3; p['vx']*=0.93; p['life']-=0.04
        if p['life']<=0: dead.append(p); continue
        c=tuple(int(v*p['life']) for v in p['col'])
        cv2.circle(frame,(int(p['x']),int(p['y'])),max(1,int(4*p['life'])),c,-1,cv2.LINE_AA)
    for d in dead: snap_anim.remove(d)

# ── Dashboard panel ───────────────────────────────────────────────
def draw_dashboard(frame, data, W, H):
    pw=310; ph=H
    panel=np.zeros((ph,pw,3),dtype=np.uint8)
    panel[:]=( 8, 8,16)

    def txt(s,y,col=(200,200,200),scale=0.48,thick=1):
        cv2.putText(panel,s,(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale,col,thick,cv2.LINE_AA)

    def bar(label,val,max_val,y,col):
        txt(label,y,(150,150,180),0.4)
        bw=int((val/max(max_val,1))*220)
        bw=min(bw,220)
        cv2.rectangle(panel,(12,y+4),(232,y+16),(30,30,50),-1)
        cv2.rectangle(panel,(12,y+4),(12+bw,y+16),col,-1)
        txt(f'{val:.1f}',y+14,(220,220,220),0.38)

    y=30
    txt('ANTIGRAVITY  VISION v2.0',y,(80,220,120),0.55,1); y+=30
    elapsed=int(time.time()-session_start)
    txt(f'Session: {elapsed//60:02d}:{elapsed%60:02d}   FPS:{data["fps"]:.0f}',y,(100,100,140),0.4); y+=28

    cv2.line(panel,(8,y),(pw-8,y),(40,40,60),1); y+=14

    # Face metrics
    txt('FACE  METRICS',y,(80,180,255),0.48,1); y+=22
    ec=(80,255,130) if data['emotion_col'] else (180,180,180)
    txt(f'Emotion  :  {data["emotion"]}',y,data['emotion_col']); y+=20
    txt(f'Age est  :  {data["age"]}',y,(180,200,180)); y+=20
    txt(f'Gender   :  {data["gender"]}',y,(180,200,180)); y+=20

    cv2.line(panel,(8,y),(pw-8,y),(40,40,60),1); y+=14

    # Eye metrics
    txt('EYE  TRACKING',y,(80,180,255),0.48,1); y+=22
    gc=data['gaze_col']
    txt(f'Gaze     :  {data["gaze_dir"]}',y,gc); y+=20
    txt(f'Gaze X   :  {data["gaze_x"]:+.2f}',y,(160,160,200)); y+=18
    txt(f'Gaze Y   :  {data["gaze_y"]:+.2f}',y,(160,160,200)); y+=18
    txt(f'Blink EAR:  {data["ear"]:.2f}',y,(160,160,200)); y+=18
    txt(f'Blink/min:  {data["blink_rate"]:.1f}',y,(160,160,200)); y+=18
    txt(f'Total blinks: {data["blink_count"]}',y,(160,160,200)); y+=22

    cv2.line(panel,(8,y),(pw-8,y),(40,40,60),1); y+=14

    # Head pose
    txt('HEAD  POSE',y,(80,180,255),0.48,1); y+=22
    txt(f'Pitch (up/dn): {data["pitch"]:+.1f}°',y,(180,180,220)); y+=18
    txt(f'Yaw  (lr)    : {data["yaw"]:+.1f}°',y,(180,180,220)); y+=18
    txt(f'Roll (tilt)  : {data["roll"]:+.1f}°',y,(180,180,220)); y+=22

    cv2.line(panel,(8,y),(pw-8,y),(40,40,60),1); y+=14

    # Mouth
    txt('MOUTH',y,(80,180,255),0.48,1); y+=22
    mcol=(80,255,130) if data['mouth_open'] else (120,120,120)
    txt(f'Mouth    :  {"OPEN" if data["mouth_open"] else "CLOSED"}',y,mcol); y+=22

    cv2.line(panel,(8,y),(pw-8,y),(40,40,60),1); y+=14

    # Attention
    txt('ATTENTION  SCORE',y,(80,180,255),0.48,1); y+=22
    ascore=data['attention']
    acol=(80,255,130) if ascore>70 else (80,200,255) if ascore>40 else (80,80,255)
    bar('Attention',ascore,100,y,acol); y+=30

    cv2.line(panel,(8,y),(pw-8,y),(40,40,60),1); y+=14

    # Hands
    txt('HAND  GESTURES',y,(80,180,255),0.48,1); y+=22
    for hand_label,gname in data['gestures'].items():
        gcols={'OPEN':(80,255,130),'FIST':(80,80,255),'PINCH':(255,180,80),
               'PEACE':(200,130,255),'POINT':(80,200,255),'CUSTOM':(180,180,180)}
        txt(f'{hand_label}: {gname}',y,gcols.get(gname,(180,180,180))); y+=20

    # Gaze direction indicator
    yg=ph-80
    cv2.rectangle(panel,(12,yg),(pw-12,yg+60),(20,20,35),-1)
    cv2.rectangle(panel,(12,yg),(pw-12,yg+60),(40,40,60),1)
    cx_g,cy_g=pw//2, yg+30
    gx=int(data['gaze_x']*60); gy=int(data['gaze_y']*60)
    cv2.circle(panel,(cx_g,cy_g),25,(30,30,50),-1)
    cv2.circle(panel,(cx_g,cy_g),25,(60,60,80),1)
    gx=max(-24,min(24,gx)); gy=max(-24,min(24,gy))
    cv2.circle(panel,(cx_g+gx,cy_g+gy),8,(80,220,255),-1,cv2.LINE_AA)
    cv2.circle(panel,(cx_g+gx,cy_g+gy),4,(255,255,255),-1,cv2.LINE_AA)
    txt('GAZE MAP',yg-6,(100,100,140),0.38)

    # merge panel onto frame
    frame[0:ph, W-pw:W] = cv2.addWeighted(
        frame[0:ph, W-pw:W], 0.15, panel, 0.85, 0)
    cv2.line(frame,(W-pw,0),(W-pw,ph),(60,60,100),1)

# ── Main state ───────────────────────────────────────────────────
strings = [PhysicsString() for _ in range(5)]
prev_gestures = {}
dashboard_data = {
    'fps':0,'emotion':'NEUTRAL','emotion_col':(180,180,180),
    'age':'--','gender':'--',
    'gaze_dir':'CENTER','gaze_col':(180,180,180),
    'gaze_x':0.0,'gaze_y':0.0,
    'ear':0.0,'blink_rate':0.0,'blink_count':0,
    'pitch':0.0,'yaw':0.0,'roll':0.0,
    'mouth_open':False,'attention':100.0,
    'gestures':{}
}

# ── Main loop ─────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    H, W  = frame.shape[:2]
    PW    = 310  # panel width
    VW    = W - PW  # visible camera width

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # slight dark tint
    frame = cv2.addWeighted(frame,0.82,np.zeros_like(frame),0.18,0)

    now=time.time(); fps=1.0/max(now-prev_time,0.001); prev_time=now
    fps_history.append(fps)
    if len(fps_history)>30: fps_history.pop(0)
    dashboard_data['fps']=sum(fps_history)/len(fps_history)

    # ── Face mesh ─────────────────────────────────────────────────
    mesh_res = face_mesh.process(rgb)
    if mesh_res.multi_face_landmarks:
        fm = mesh_res.multi_face_landmarks[0]

        def lmpt(i): return (int(fm.landmark[i].x*W), int(fm.landmark[i].y*H))

        # draw subtle mesh
        mp_draw.draw_landmarks(
            frame, fm,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw_styles.get_default_face_mesh_tesselation_style())

        # Eye landmarks
        l_top=lmpt(L_EYE_TOP); l_bot=lmpt(L_EYE_BOT)
        l_left=lmpt(L_EYE_L);  l_right=lmpt(L_EYE_R)
        r_top=lmpt(R_EYE_TOP); r_bot=lmpt(R_EYE_BOT)
        r_left=lmpt(R_EYE_L);  r_right=lmpt(R_EYE_R)

        ear_l=eye_aspect_ratio(l_top,l_bot,l_left,l_right)
        ear_r=eye_aspect_ratio(r_top,r_bot,r_left,r_right)
        ear  =blink_smooth.update((ear_l+ear_r)/2)
        dashboard_data['ear']=ear

        # blink
        eye_closed = ear < 0.18
        if eye_closed and not prev_eye_closed:
            blink_count+=1
            blink_history.append(time.time())
        prev_eye_closed=eye_closed
        blink_history=[t for t in blink_history if now-t<60]
        blink_rate=len(blink_history)
        dashboard_data['blink_count']=blink_count
        dashboard_data['blink_rate']=blink_rate

        # iris / gaze
        try:
            l_iris=lmpt(L_IRIS_CENTER)
            r_iris=lmpt(R_IRIS_CENTER)
            lgx,lgy=get_gaze(l_iris,l_left,l_right,l_top,l_bot)
            rgx,rgy=get_gaze(r_iris,r_left,r_right,r_top,r_bot)
            gx=gaze_x_smooth.update((lgx+rgx)/2)
            gy=gaze_y_smooth.update((lgy+rgy)/2)
        except:
            gx,gy=0.0,0.0
        dashboard_data['gaze_x']=gx
        dashboard_data['gaze_y']=gy

        if   gx < -0.15: gdir='LEFT'
        elif gx >  0.15: gdir='RIGHT'
        elif gy < -0.15: gdir='UP'
        elif gy >  0.15: gdir='DOWN'
        else:            gdir='CENTER'
        gcmap={'LEFT':(255,130,80),'RIGHT':(255,130,80),
               'UP':(80,200,255),'DOWN':(80,200,255),'CENTER':(80,255,130)}
        dashboard_data['gaze_dir']=gdir
        dashboard_data['gaze_col']=gcmap[gdir]

        # draw iris highlight
        for iris_pt in [l_iris, r_iris]:
            cv2.circle(frame,iris_pt,6,(80,220,255),-1,cv2.LINE_AA)
            cv2.circle(frame,iris_pt,10,(80,220,255),1,cv2.LINE_AA)

        # head pose
        lm_list=[lmpt(i) for i in range(468)]
        try:
            pitch,yaw,roll=get_head_pose(lm_list,W,H)
        except:
            pitch,yaw,roll=0.0,0.0,0.0
        dashboard_data['pitch']=pitch
        dashboard_data['yaw']=yaw
        dashboard_data['roll']=roll

        # emotion
        emo,ecol=estimate_emotion(fm,W,H)
        dashboard_data['emotion']=emo
        dashboard_data['emotion_col']=ecol

        # mouth
        m_h=math.dist(lmpt(MOUTH_TOP),lmpt(MOUTH_BOT))
        m_w=math.dist(lmpt(MOUTH_L),  lmpt(MOUTH_R))
        mouth_open = mouth_smooth.update(m_h/(m_w+1e-6)) > 0.25
        dashboard_data['mouth_open']=mouth_open

        # attention
        att=calc_attention(pitch,yaw,blink_rate,mouth_open)
        att=attention_smooth.update(att)
        dashboard_data['attention']=att

    # ── Face detection (age/gender box) ──────────────────────────
    face_res=face_det.process(rgb)
    if face_res.detections:
        for det in face_res.detections:
            bb=det.location_data.relative_bounding_box
            x1=int(bb.xmin*W); y1=int(bb.ymin*H)
            x2=x1+int(bb.width*W); y2=y1+int(bb.height*H)
            age_b,gender=estimate_age_gender(bb,W,H)
            dashboard_data['age']=age_b
            dashboard_data['gender']=gender
            col=(80,220,120)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,1,cv2.LINE_AA)
            cv2.putText(frame,f'AGE~{age_b}',(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)

    # ── Hands ─────────────────────────────────────────────────────
    hand_res=hands.process(rgb)
    hand_data=[]
    if hand_res.multi_hand_landmarks:
        for idx,lm in enumerate(hand_res.multi_hand_landmarks):
            label='LEFT' if idx==0 else 'RIGHT'
            mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=COLS_BGR[idx*2%5],thickness=1,circle_radius=3),
                mp_draw.DrawingSpec(color=(60,60,100),thickness=1))
            g,up,pd=detect_gesture(lm,W,H)
            tips=[(int(lm.landmark[FINGER_TIPS[f]].x*W),
                   int(lm.landmark[FINGER_TIPS[f]].y*H)) for f in range(5)]
            hand_data.append((label,lm,g,tips,up,pd))
            dashboard_data['gestures'][label]=g
            for fi,tip in enumerate(tips):
                cv2.circle(frame,tip,8,COLS_BGR[fi],-1,cv2.LINE_AA)
                cv2.circle(frame,tip,13,COLS_BGR[fi],1,cv2.LINE_AA)

    # ── Strings between hands ─────────────────────────────────────
    if len(hand_data)==2:
        _,lm0,g0,tips0,up0,pd0=hand_data[0]
        _,lm1,g1,tips1,up1,pd1=hand_data[1]
        for fi in range(5):
            active=up0[fi] and up1[fi] and g0!='FIST' and g1!='FIST'
            if active:
                strings[fi].update(tips0[fi],tips1[fi])
                strings[fi].draw(frame,COLS_BGR[fi])
            else:
                mid=((tips0[fi][0]+tips1[fi][0])//2,
                     (tips0[fi][1]+tips1[fi][1])//2)
                for nd in strings[fi].nodes:
                    nd.x=nd.px=mid[0]; nd.y=nd.py=mid[1]

        # fist release snap
        both_fist=g0=='FIST' and g1=='FIST'
        if prev_gestures.get('both_fist') and not both_fist:
            for fi in range(5):
                add_snap(tips0[fi][0],tips0[fi][1],COLS_BGR[fi])
                add_snap(tips1[fi][0],tips1[fi][1],COLS_BGR[fi])
        prev_gestures['both_fist']=both_fist

        # open hand web
        for _,lm,g,tips,up,pd in hand_data:
            if g=='OPEN':
                for fi in range(4):
                    p1,p2=tips[fi],tips[fi+1]
                    pts=[]
                    for t in range(11):
                        f=t/10
                        mx=int(p1[0]+(p2[0]-p1[0])*f)
                        my=int(p1[1]+(p2[1]-p1[1])*f)
                        pts.append((mx,my-int(math.sin(math.pi*f)*28)))
                    for i in range(len(pts)-1):
                        cv2.line(frame,pts[i],pts[i+1],COLS_BGR[fi],1,cv2.LINE_AA)
                    poly=np.array([p1]+pts+[p2],np.int32)
                    ov=frame.copy()
                    cv2.fillPoly(ov,[poly],COLS_BGR[fi])
                    cv2.addWeighted(ov,0.12,frame,0.88,0,frame)

    update_snap(frame)

    # ── Gaze trail on camera ──────────────────────────────────────
    gx_raw=dashboard_data['gaze_x']; gy_raw=dashboard_data['gaze_y']
    gaze_screen_x=int(VW//2 + gx_raw*VW*0.4)
    gaze_screen_y=int(H//2  + gy_raw*H*0.4)
    gaze_screen_x=max(20,min(VW-20,gaze_screen_x))
    gaze_screen_y=max(20,min(H-20,gaze_screen_y))
    cv2.circle(frame,(gaze_screen_x,gaze_screen_y),18,(80,220,255),1,cv2.LINE_AA)
    cv2.circle(frame,(gaze_screen_x,gaze_screen_y),4,(80,220,255),-1,cv2.LINE_AA)

    # ── Bottom bar ────────────────────────────────────────────────
    emo=dashboard_data['emotion']
    ecol=dashboard_data['emotion_col']
    att=dashboard_data['attention']
    acol=(80,255,130) if att>70 else (80,200,255) if att>40 else (80,80,255)
    cv2.rectangle(frame,(0,H-32),(VW,H),(8,8,16),-1)
    cv2.putText(frame,
        f'Emotion:{emo}  Gaze:{dashboard_data["gaze_dir"]}  '
        f'Attention:{att:.0f}%  Blinks:{blink_count}  '
        f'Head P:{dashboard_data["pitch"]:+.0f} Y:{dashboard_data["yaw"]:+.0f}  '
        f'Q=quit',
        (12,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,acol,1,cv2.LINE_AA)

    # ── Draw dashboard panel ──────────────────────────────────────
    draw_dashboard(frame, dashboard_data, W, H)

    cv2.imshow('ANTIGRAVITY VISION v2.0',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()