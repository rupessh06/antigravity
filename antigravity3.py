import cv2
import mediapipe as mp
import numpy as np
import math, time, random

# ── MediaPipe ────────────────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils

hands     = mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.6)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.6,
                                   min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
W, H = 1280, 720

# ── Landmark IDs ─────────────────────────────────────────────────
L_IRIS = 468
R_IRIS = 473
L_EYE_L, L_EYE_R = 33, 133
L_EYE_T, L_EYE_B = 159, 145
R_EYE_L, R_EYE_R = 362, 263
R_EYE_T, R_EYE_B = 386, 374
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]

# ── Smoother ─────────────────────────────────────────────────────
class Smoother:
    def __init__(self, n=12):
        self.buf=[]; self.n=n
    def update(self, v):
        self.buf.append(v)
        if len(self.buf)>self.n: self.buf.pop(0)
        return sum(self.buf)/len(self.buf)

gx_sm = Smoother(14)
gy_sm = Smoother(14)

# ── Colors ────────────────────────────────────────────────────────
NEON_GREEN  = (0, 255, 120)
NEON_CYAN   = (0, 220, 255)
NEON_RED    = (0, 60, 255)
NEON_ORANGE = (0, 140, 255)
NEON_PURPLE = (220, 80, 255)
NEON_BLUE   = (255, 160, 0)
NEON_YELLOW = (0, 230, 255)
WHITE       = (255, 255, 255)

# ── Particle system ───────────────────────────────────────────────
particles = []

def spawn_particles(x, y, col, n=20, speed=8, life=1.0, size=4):
    for _ in range(n):
        a = random.uniform(0, 2*math.pi)
        s = random.uniform(speed*0.3, speed)
        particles.append({
            'x':float(x), 'y':float(y),
            'vx':math.cos(a)*s, 'vy':math.sin(a)*s,
            'life':life, 'max_life':life,
            'col':col, 'size':size,
            'type':'normal'
        })

def spawn_trail(x, y, col):
    particles.append({
        'x':float(x), 'y':float(y),
        'vx':random.uniform(-1,1), 'vy':random.uniform(-1,1),
        'life':0.3, 'max_life':0.3,
        'col':col, 'size':3, 'type':'trail'
    })

def update_particles(frame):
    dead=[]
    for p in particles:
        p['x']+=p['vx']; p['y']+=p['vy']
        if p['type']=='normal':
            p['vy']+=0.2; p['vx']*=0.95
        p['life']-=0.035
        if p['life']<=0: dead.append(p); continue
        alpha=p['life']/p['max_life']
        c=tuple(min(255,int(v*alpha)) for v in p['col'])
        r=max(1,int(p['size']*alpha))
        px,py=int(p['x']),int(p['y'])
        if 0<=px<W and 0<=py<H:
            cv2.circle(frame,(px,py),r,c,-1,cv2.LINE_AA)
            if r>2:
                cv2.circle(frame,(px,py),r+2,c,1,cv2.LINE_AA)
    for d in dead:
        if d in particles: particles.remove(d)

# ── Bullet ────────────────────────────────────────────────────────
class Bullet:
    def __init__(self, x, y, tx, ty, power=1):
        self.x=float(x); self.y=float(y)
        self.alive=True; self.power=power
        d=max(math.sqrt((tx-x)**2+(ty-y)**2),1)
        spd=22
        self.vx=(tx-x)/d*spd
        self.vy=(ty-y)/d*spd
        self.trail=[]
        self.col=NEON_CYAN if power==1 else NEON_YELLOW

    def update(self, frame):
        self.trail.append((int(self.x),int(self.y)))
        if len(self.trail)>12: self.trail.pop(0)
        self.x+=self.vx; self.y+=self.vy
        if not (0<self.x<W and 0<self.y<H):
            self.alive=False; return
        # draw trail
        for i,pt in enumerate(self.trail):
            alpha=i/len(self.trail)
            c=tuple(int(v*alpha) for v in self.col)
            r=max(1,int(5*alpha))
            cv2.circle(frame,pt,r,c,-1,cv2.LINE_AA)
        # draw bullet
        cv2.circle(frame,(int(self.x),int(self.y)),7,self.col,-1,cv2.LINE_AA)
        cv2.circle(frame,(int(self.x),int(self.y)),11,self.col,1,cv2.LINE_AA)
        cv2.circle(frame,(int(self.x),int(self.y)),16,
                   tuple(v//3 for v in self.col),1,cv2.LINE_AA)
        spawn_trail(self.x,self.y,self.col)

# ── Virus ─────────────────────────────────────────────────────────
VIRUS_TYPES = [
    {'name':'WORM',    'col':NEON_RED,    'hp':1, 'spd':1.8, 'size':18, 'reward':10},
    {'name':'TROJAN',  'col':NEON_ORANGE, 'hp':3, 'spd':1.2, 'size':24, 'reward':30},
    {'name':'ROOTKIT', 'col':NEON_PURPLE, 'hp':6, 'spd':0.8, 'size':32, 'reward':60},
    {'name':'ZERO-DAY','col':NEON_YELLOW, 'hp':2, 'spd':3.0, 'size':14, 'reward':50},
]

class Virus:
    def __init__(self, level=1):
        vt=random.choice(VIRUS_TYPES[:min(level,4)])
        self.name=vt['name']
        self.col=vt['col']
        self.max_hp=vt['hp']+level//3
        self.hp=self.max_hp
        self.spd=vt['spd']+level*0.05
        self.size=vt['size']
        self.reward=vt['reward']+level*5
        self.alive=True
        self.t=0
        self.wobble=random.uniform(0,math.pi*2)
        # spawn on edges
        edge=random.randint(0,3)
        if edge==0: self.x=random.randint(50,W-50); self.y=-30
        elif edge==1: self.x=W+30; self.y=random.randint(50,H-50)
        elif edge==2: self.x=random.randint(50,W-50); self.y=H+30
        else: self.x=-30; self.y=random.randint(50,H-50)
        # core target
        self.tx=W//2; self.ty=H//2
        self.hit_flash=0
        self.angle=random.uniform(0,math.pi*2)
        self.slowmo=False

    def update(self):
        self.t+=1
        self.angle+=0.04
        spd=self.spd*(0.4 if self.slowmo else 1.0)
        dx=self.tx-self.x; dy=self.ty-self.y
        d=max(math.sqrt(dx*dx+dy*dy),1)
        wobble=math.sin(self.t*0.1+self.wobble)*0.3
        self.x+=dx/d*spd + math.cos(self.t*0.05)*wobble*spd
        self.y+=dy/d*spd + math.sin(self.t*0.05)*wobble*spd
        if self.hit_flash>0: self.hit_flash-=1

    def draw(self, frame):
        cx,cy=int(self.x),int(self.y)
        s=self.size
        col=WHITE if self.hit_flash>0 else self.col
        # outer ring
        cv2.circle(frame,(cx,cy),s+4,tuple(v//3 for v in col),1,cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),s+2,tuple(v//2 for v in col),1,cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),s,col,2,cv2.LINE_AA)
        # fill
        ov=frame.copy()
        cv2.circle(ov,(cx,cy),s-2,tuple(v//6 for v in col),-1)
        cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
        # spikes
        for i in range(8):
            a=self.angle+i*math.pi/4
            x1=int(cx+math.cos(a)*(s-2))
            y1=int(cy+math.sin(a)*(s-2))
            x2=int(cx+math.cos(a)*(s+8+math.sin(self.t*0.08+i)*3))
            y2=int(cy+math.sin(a)*(s+8+math.sin(self.t*0.08+i)*3))
            cv2.line(frame,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)
        # hp bar
        bw=s*2
        hp_frac=self.hp/self.max_hp
        cv2.rectangle(frame,(cx-s,cy-s-14),(cx+s,cy-s-6),(30,30,30),-1)
        cv2.rectangle(frame,(cx-s,cy-s-14),(cx-s+int(bw*hp_frac),cy-s-6),col,-1)
        cv2.rectangle(frame,(cx-s,cy-s-14),(cx+s,cy-s-6),col,1)
        # name
        cv2.putText(frame,self.name,(cx-s,cy-s-18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.3,col,1,cv2.LINE_AA)

    def hit(self, dmg=1):
        self.hp-=dmg
        self.hit_flash=6
        spawn_particles(self.x,self.y,self.col,n=10,speed=5,life=0.5,size=3)
        if self.hp<=0:
            self.alive=False
            spawn_particles(self.x,self.y,self.col,n=40,speed=10,life=1.2,size=5)
            return True
        return False

# ── Explosion ─────────────────────────────────────────────────────
class Explosion:
    def __init__(self, x, y, r=120, col=NEON_CYAN):
        self.x=x; self.y=y; self.max_r=r
        self.r=0; self.alive=True; self.col=col
        self.t=0
    def update(self, frame):
        self.t+=4
        self.r=self.t
        alpha=max(0,1-self.t/self.max_r)
        if alpha<=0: self.alive=False; return
        c=tuple(int(v*alpha) for v in self.col)
        cx,cy=int(self.x),int(self.y)
        cv2.circle(frame,(cx,cy),int(self.r),c,2,cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),max(1,int(self.r*0.6)),
                   tuple(int(v*alpha*0.4) for v in self.col),3,cv2.LINE_AA)

# ── Screen effects ────────────────────────────────────────────────
shake_frames=0
shake_intensity=0
glitch_frames=0

def trigger_shake(intensity=8, frames=6):
    global shake_frames, shake_intensity
    shake_frames=frames; shake_intensity=intensity

def trigger_glitch(frames=4):
    global glitch_frames
    glitch_frames=frames

def apply_shake(frame):
    global shake_frames
    if shake_frames<=0: return frame
    dx=random.randint(-shake_intensity,shake_intensity)
    dy=random.randint(-shake_intensity,shake_intensity)
    M=np.float32([[1,0,dx],[0,1,dy]])
    shake_frames-=1
    return cv2.warpAffine(frame,M,(W,H))

def apply_glitch(frame):
    global glitch_frames
    if glitch_frames<=0: return frame
    glitch_frames-=1
    out=frame.copy()
    for _ in range(random.randint(2,5)):
        y1=random.randint(0,H-20)
        h2=random.randint(5,20)
        shift=random.randint(-30,30)
        out[y1:y1+h2]=np.roll(out[y1:y1+h2],shift,axis=1)
    b,g,r=cv2.split(out)
    out=cv2.merge([np.roll(b,2,axis=1),g,np.roll(r,-2,axis=1)])
    return out

# ── Core (system to protect) ──────────────────────────────────────
core_pulse=0

def draw_core(frame, lives, max_lives=3):
    global core_pulse
    core_pulse+=0.08
    cx,cy=W//2,H//2
    frac=lives/max_lives
    col=NEON_GREEN if frac>0.6 else NEON_ORANGE if frac>0.3 else NEON_RED
    # outer rings
    for r,a in [(60,0.15),(50,0.25),(38,0.4)]:
        ov=frame.copy()
        cv2.circle(ov,(cx,cy),r,col,-1)
        cv2.addWeighted(ov,a*0.3,frame,1-a*0.3,0,frame)
        cv2.circle(frame,(cx,cy),r,col,1,cv2.LINE_AA)
    # pulse ring
    pr=int(65+math.sin(core_pulse)*8)
    cv2.circle(frame,(cx,cy),pr,tuple(v//2 for v in col),1,cv2.LINE_AA)
    # hexagon
    for i in range(6):
        a=i*math.pi/3+core_pulse*0.2
        x1=int(cx+math.cos(a)*35); y1=int(cy+math.sin(a)*35)
        a2=(i+1)*math.pi/3+core_pulse*0.2
        x2=int(cx+math.cos(a2)*35); y2=int(cy+math.sin(a2)*35)
        cv2.line(frame,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)
    cv2.putText(frame,'CORE',(cx-14,cy+5),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,WHITE,1,cv2.LINE_AA)
    # lives
    for i in range(max_lives):
        c2=col if i<lives else (40,40,40)
        cv2.circle(frame,(cx-20+i*20,cy+52),6,c2,-1,cv2.LINE_AA)

# ── HUD ───────────────────────────────────────────────────────────
def draw_hud(frame, score, level, combo, wave, shield_active,
             slowmo_active, bomb_cooldown, fire_cooldown, gx, gy):
    # top bar
    cv2.rectangle(frame,(0,0),(W,42),(8,8,16),-1)
    cv2.line(frame,(0,42),(W,42),NEON_CYAN,1)

    # score
    cv2.putText(frame,f'SCORE: {score:07d}',(16,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,NEON_CYAN,1,cv2.LINE_AA)
    # level
    cv2.putText(frame,f'LEVEL {level}',(220,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,NEON_GREEN,1,cv2.LINE_AA)
    # combo
    if combo>1:
        cc=NEON_YELLOW if combo<5 else NEON_ORANGE if combo<10 else NEON_RED
        cv2.putText(frame,f'x{combo} COMBO',(340,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,cc,2,cv2.LINE_AA)
    # wave
    cv2.putText(frame,f'WAVE {wave}',(520,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,NEON_PURPLE,1,cv2.LINE_AA)

    # abilities bar bottom
    y2=H-10
    # shield
    scol=NEON_GREEN if shield_active else ((80,200,100) if bomb_cooldown==0 else (60,60,80))
    cv2.putText(frame,f'[FIST] SHIELD: {"ACTIVE" if shield_active else "READY"}',(16,y2),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,scol,1,cv2.LINE_AA)
    # slowmo
    smcol=NEON_CYAN if slowmo_active else (60,140,160)
    cv2.putText(frame,'[OPEN] SLOW-MO: ACTIVE' if slowmo_active else '[OPEN] SLOW-MO',(260,y2),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,smcol,1,cv2.LINE_AA)
    # bomb
    bcol=NEON_YELLOW if bomb_cooldown==0 else (80,80,40)
    blabel=f'[PEACE] BOMB: {"READY" if bomb_cooldown==0 else f"CD {bomb_cooldown}s"}'
    cv2.putText(frame,blabel,(500,y2),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,bcol,1,cv2.LINE_AA)
    # fire
    fcol=NEON_CYAN if fire_cooldown==0 else (40,80,100)
    cv2.putText(frame,'[PINCH] FIRE: READY' if fire_cooldown==0 else '[PINCH] RELOAD...',
                (740,y2),cv2.FONT_HERSHEY_SIMPLEX,0.4,fcol,1,cv2.LINE_AA)

    # gaze indicator top right
    cv2.putText(frame,f'GAZE ({gx:+.2f},{gy:+.2f})',
                (W-200,28),cv2.FONT_HERSHEY_SIMPLEX,0.4,NEON_PURPLE,1,cv2.LINE_AA)

# ── Reticle ───────────────────────────────────────────────────────
reticle_t=0
def draw_reticle(frame, x, y, firing=False, shield=False):
    global reticle_t
    reticle_t+=0.12
    col=NEON_GREEN if shield else (NEON_RED if firing else NEON_CYAN)
    r1,r2=22,30
    # animated arcs
    for i in range(4):
        a=reticle_t+i*math.pi/2
        x1=int(x+math.cos(a)*r1); y1=int(y+math.sin(a)*r1)
        x2=int(x+math.cos(a+0.8)*r2); y2=int(y+math.sin(a+0.8)*r2)
        cv2.line(frame,(x1,y1),(x2,y2),col,2,cv2.LINE_AA)
    cv2.circle(frame,(int(x),int(y)),5,col,-1,cv2.LINE_AA)
    cv2.circle(frame,(int(x),int(y)),9,col,1,cv2.LINE_AA)
    cv2.circle(frame,(int(x),int(y)),r1,tuple(v//3 for v in col),1,cv2.LINE_AA)
    # crosshairs
    cv2.line(frame,(int(x)-18,int(y)),(int(x)-8,int(y)),col,1,cv2.LINE_AA)
    cv2.line(frame,(int(x)+8, int(y)),(int(x)+18,int(y)),col,1,cv2.LINE_AA)
    cv2.line(frame,(int(x),int(y)-18),(int(x),int(y)-8),col,1,cv2.LINE_AA)
    cv2.line(frame,(int(x),int(y)+8),(int(x),int(y)+18),col,1,cv2.LINE_AA)
    if firing:
        spawn_trail(x,y,col)

# ── Grid background ───────────────────────────────────────────────
grid_t=0
def draw_grid(frame):
    global grid_t
    grid_t+=0.5
    for x in range(0,W,80):
        alpha=0.06+0.02*math.sin(grid_t*0.05+x*0.01)
        col=tuple(int(v*alpha) for v in NEON_CYAN)
        cv2.line(frame,(x,0),(x,H),col,1)
    for y in range(0,H,80):
        alpha=0.06+0.02*math.sin(grid_t*0.05+y*0.01)
        col=tuple(int(v*alpha) for v in NEON_CYAN)
        cv2.line(frame,(0,y),(W,y),col,1)

# ── Gesture helpers ───────────────────────────────────────────────
def finger_up(lm,tip,base):
    return lm.landmark[tip].y < lm.landmark[base].y

def is_pinch(lm):
    tx=lm.landmark[4].x; ty=lm.landmark[4].y
    ix=lm.landmark[8].x; iy=lm.landmark[8].y
    d=math.sqrt((tx-ix)**2+(ty-iy)**2)
    return d<0.07

def is_fist(lm):
    up=[finger_up(lm,FINGER_TIPS[i],FINGER_BASES[i]) for i in range(5)]
    return sum(up)==0

def is_open(lm):
    up=[finger_up(lm,FINGER_TIPS[i],FINGER_BASES[i]) for i in range(5)]
    return sum(up)==5

def is_peace(lm):
    up=[finger_up(lm,FINGER_TIPS[i],FINGER_BASES[i]) for i in range(5)]
    return up[1] and up[2] and not up[3] and not up[4]

# ── Wave system ───────────────────────────────────────────────────
def spawn_wave(level, wave):
    count=3+level*2+wave
    return [Virus(level) for _ in range(min(count,18))]

# ── Game state ────────────────────────────────────────────────────
STATE='PLAYING'   # PLAYING / DEAD / WIN
score=0
lives=3
level=1
wave=1
combo=0
combo_timer=0
viruses=spawn_wave(level,wave)
bullets=[]
explosions=[]
shield_active=False
slowmo_active=False
bomb_cooldown=0
bomb_ts=0
fire_cooldown=0
fire_ts=0
pinch_prev=False
peace_prev=False
gaze_x=W//2; gaze_y=H//2
prev_time=time.time()
score_popups=[]   # [(x,y,text,col,life)]
damage_flash=0

# ── Main loop ─────────────────────────────────────────────────────
while True:
    ret,frame=cap.read()
    if not ret: break
    frame=cv2.flip(frame,1)
    H2,W2=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    now=time.time()
    dt=now-prev_time; prev_time=now

    # ── Gaze tracking ─────────────────────────────────────────────
    mesh_res=face_mesh.process(rgb)
    if mesh_res.multi_face_landmarks:
        fm=mesh_res.multi_face_landmarks[0]
        def lmpt(i): return (fm.landmark[i].x*W, fm.landmark[i].y*H)
        try:
            l_iris=lmpt(L_IRIS)
            l_el=lmpt(L_EYE_L); l_er=lmpt(L_EYE_R)
            l_et=lmpt(L_EYE_T); l_eb=lmpt(L_EYE_B)
            eye_w=max(math.dist(l_el,l_er),1)
            eye_h=max(math.dist(l_et,l_eb),1)
            raw_gx=(l_iris[0]-l_el[0])/eye_w - 0.5
            raw_gy=(l_iris[1]-l_et[1])/eye_h - 0.5
            sgx=gx_sm.update(raw_gx)
            sgy=gy_sm.update(raw_gy)
            gaze_x=int(W//2 + sgx*W*1.4)
            gaze_y=int(H//2 + sgy*H*1.4)
            gaze_x=max(20,min(W-20,gaze_x))
            gaze_y=max(60,min(H-60,gaze_y))
        except: pass

    # ── Hand gestures ─────────────────────────────────────────────
    hand_res=hands.process(rgb)
    pinch_now=False; fist_now=False
    open_now=False;  peace_now=False

    if hand_res.multi_hand_landmarks:
        for lm in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,180,80),thickness=1,circle_radius=2),
                mp_draw.DrawingSpec(color=(0,80,40),thickness=1))
            if is_pinch(lm): pinch_now=True
            if is_fist(lm):  fist_now=True
            if is_open(lm):  open_now=True
            if is_peace(lm): peace_now=True

    # SHIELD
    shield_active=fist_now
    # SLOW-MO
    slowmo_active=open_now
    # BOMB
    if peace_now and not peace_prev:
        if now-bomb_ts>6:
            bomb_ts=now
            explosions.append(Explosion(W//2,H//2,r=W,col=NEON_YELLOW))
            spawn_particles(W//2,H//2,NEON_YELLOW,n=80,speed=15,life=1.5,size=6)
            trigger_shake(12,10); trigger_glitch(8)
            for v in viruses:
                v.hp-=999; v.alive=False
                spawn_particles(v.x,v.y,v.col,n=30,speed=8,life=1.0,size=4)
            viruses=[v for v in viruses if v.alive]
            score+=500; score_popups.append([W//2,H//3,'SYSTEM BOMB! +500',NEON_YELLOW,1.5])

    bomb_cooldown=max(0,int(6-(now-bomb_ts)))
    peace_prev=peace_now

    # FIRE on pinch rising edge
    if pinch_now and not pinch_prev:
        if now-fire_ts>0.25:
            fire_ts=now
            bullets.append(Bullet(gaze_x,gaze_y,
                                   random.randint(gaze_x-5,gaze_x+5),
                                   random.randint(gaze_y-200,gaze_y-50),
                                   power=2 if combo>5 else 1))
            spawn_particles(gaze_x,gaze_y,NEON_CYAN,n=12,speed=6,life=0.4,size=3)

    fire_cooldown=max(0,round((0.25-(now-fire_ts))*10)/10)
    pinch_prev=pinch_now

    # ── Draw background + grid ────────────────────────────────────
    dark=frame.copy()
    cv2.addWeighted(frame,0.55,np.zeros_like(frame),0.45,0,frame)
    draw_grid(frame)

    # ── Core ──────────────────────────────────────────────────────
    draw_core(frame,lives)

    # ── Update + draw viruses ─────────────────────────────────────
    for v in viruses:
        v.slowmo=slowmo_active
        v.update()
        # reach core?
        if math.dist((v.x,v.y),(W//2,H//2))<62 and v.alive:
            if shield_active:
                v.hit(999)
                spawn_particles(v.x,v.y,NEON_GREEN,n=30,speed=8)
                trigger_shake(6,4)
                score_popups.append([v.x,v.y,'SHIELD BLOCK!',NEON_GREEN,1.0])
            else:
                v.alive=False
                lives-=1
                damage_flash=8
                trigger_shake(15,8); trigger_glitch(6)
                spawn_particles(W//2,H//2,NEON_RED,n=50,speed=12,life=1.5,size=5)
                combo=0
                score_popups.append([W//2,H//2-60,'SYSTEM BREACHED!',(0,60,255),1.2])
        v.draw(frame)

    viruses=[v for v in viruses if v.alive]

    # ── Bullet ↔ virus collision ───────────────────────────────────
    for b in bullets:
        b.update(frame)
        for v in viruses:
            if v.alive and math.dist((b.x,b.y),(v.x,v.y))<v.size+8:
                killed=v.hit(b.power)
                b.alive=False
                if killed:
                    pts=v.reward*(combo+1)
                    score+=pts
                    combo+=1; combo_timer=now
                    explosions.append(Explosion(v.x,v.y,r=80,col=v.col))
                    trigger_shake(5,4)
                    score_popups.append([int(v.x),int(v.y)-20,
                                         f'+{pts} x{combo}',v.col,0.9])
                break

    bullets=[b for b in bullets if b.alive]

    # combo decay
    if combo>0 and now-combo_timer>3:
        combo=0

    # ── Explosions ────────────────────────────────────────────────
    for e in explosions:
        e.update(frame)
    explosions=[e for e in explosions if e.alive]

    # ── Particles ─────────────────────────────────────────────────
    update_particles(frame)

    # ── Score popups ──────────────────────────────────────────────
    dead_pop=[]
    for pop in score_popups:
        px,py,txt2,pc,pl=pop
        pop[4]-=dt
        alpha=min(1.0,pop[4])
        c=tuple(int(v*alpha) for v in pc)
        cv2.putText(frame,txt2,(int(px)-30,int(py-20*(1-pop[4]))),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,c,2,cv2.LINE_AA)
        if pop[4]<=0: dead_pop.append(pop)
    for d in dead_pop: score_popups.remove(d)

    # ── Reticle ───────────────────────────────────────────────────
    draw_reticle(frame,gaze_x,gaze_y,firing=pinch_now,shield=shield_active)

    # ── Damage flash ──────────────────────────────────────────────
    if damage_flash>0:
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(W,H),(0,0,180),-1)
        cv2.addWeighted(ov,0.25,frame,0.75,0,frame)
        damage_flash-=1

    # ── Slow-mo overlay ───────────────────────────────────────────
    if slowmo_active:
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(W,H),(80,0,80),-1)
        cv2.addWeighted(ov,0.08,frame,0.92,0,frame)
        cv2.putText(frame,'// SLOW-MO //',(W//2-80,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,NEON_CYAN,1,cv2.LINE_AA)

    # ── Next wave ─────────────────────────────────────────────────
    if len(viruses)==0:
        wave+=1
        if wave>5: level+=1; wave=1
        viruses=spawn_wave(level,wave)
        score_popups.append([W//2-60,H//2-80,
                              f'WAVE {wave} INCOMING!',NEON_RED,2.0])
        spawn_particles(W//2,H//2,NEON_GREEN,n=60,speed=12,life=1.5,size=4)

    # ── Game over ─────────────────────────────────────────────────
    if lives<=0:
        ov=frame.copy()
        cv2.rectangle(ov,(200,220),(W-200,420),(8,8,16),-1)
        cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
        cv2.rectangle(frame,(200,220),(W-200,420),NEON_RED,2)
        cv2.putText(frame,'SYSTEM COMPROMISED',(240,290),
                    cv2.FONT_HERSHEY_SIMPLEX,1.1,NEON_RED,2,cv2.LINE_AA)
        cv2.putText(frame,f'FINAL SCORE: {score:07d}',(300,340),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,NEON_CYAN,1,cv2.LINE_AA)
        cv2.putText(frame,'Press R to restart  |  Q to quit',(280,385),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,1,cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────
    draw_hud(frame,score,level,combo,wave,
             shield_active,slowmo_active,
             bomb_cooldown,fire_cooldown,
             round(gx_sm.buf[-1] if gx_sm.buf else 0,2),
             round(gy_sm.buf[-1] if gy_sm.buf else 0,2))

    # ── Screen FX ────────────────────────────────────────────────
    frame=apply_glitch(frame)
    frame=apply_shake(frame)

    cv2.imshow('ANTIGRAVITY : NEURAL DEFENSE',frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    if key==ord('r'):
        score=0; lives=3; level=1; wave=1; combo=0
        viruses=spawn_wave(1,1); bullets=[]; explosions=[]
        particles.clear(); score_popups.clear()

cap.release()
cv2.destroyAllWindows()