import cv2
import mediapipe as mp
import numpy as np
import math, time, random, json, os

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

# ── High-score persistence ─────────────────────────────────────
SAVE_FILE = os.path.join(os.path.dirname(__file__), 'ag4_save.json')

def load_high_score():
    try:
        with open(SAVE_FILE) as f:
            return json.load(f).get('high_score', 0)
    except:
        return 0

def save_high_score(s):
    try:
        with open(SAVE_FILE, 'w') as f:
            json.dump({'high_score': s}, f)
    except:
        pass

HIGH_SCORE = load_high_score()

# ── Landmark IDs ──────────────────────────────────────────────
L_IRIS = 468;  R_IRIS = 473
L_EYE_L, L_EYE_R = 33, 133
L_EYE_T, L_EYE_B = 159, 145
R_EYE_T, R_EYE_B = 386, 374
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]

# ── Smoother ──────────────────────────────────────────────────
class Smoother:
    def __init__(self, n=12):
        self.buf=[]; self.n=n
    def update(self, v):
        self.buf.append(v)
        if len(self.buf)>self.n: self.buf.pop(0)
        return sum(self.buf)/len(self.buf)

gx_sm = Smoother(14);  gy_sm = Smoother(14)
ear_sm = Smoother(6)   # for blink

# ── Colors ────────────────────────────────────────────────────
NEON_GREEN  = (0, 255, 120)
NEON_CYAN   = (0, 220, 255)
NEON_RED    = (0,  60, 255)
NEON_ORANGE = (0, 140, 255)
NEON_PURPLE = (220, 80, 255)
NEON_BLUE   = (255, 160,   0)
NEON_YELLOW = (0,  230, 255)
NEON_PINK   = (180,  40, 255)
NEON_LIME   = (40,  255,  80)
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)

# ── Stars (parallax layers) ────────────────────────────────────
STAR_LAYERS = []
for layer in range(3):
    stars = []
    for _ in range(40 + layer*30):
        stars.append({
            'x': random.uniform(0, W),
            'y': random.uniform(0, H),
            'r': random.uniform(0.5, 1.5 + layer*0.5),
            'speed': 0.1 + layer*0.15,
            'bright': 80 + layer*60,
        })
    STAR_LAYERS.append(stars)

def draw_stars(frame):
    for layer in STAR_LAYERS:
        for s in layer:
            s['x'] -= s['speed']
            if s['x'] < 0: s['x'] = W; s['y'] = random.uniform(0, H)
            b = int(s['bright'])
            cv2.circle(frame, (int(s['x']), int(s['y'])), int(s['r']), (b, b, b), -1, cv2.LINE_AA)

# ── Particle system ───────────────────────────────────────────
particles = []

def spawn_particles(x, y, col, n=20, speed=8, life=1.0, size=4, gravity=0.2):
    for _ in range(n):
        a = random.uniform(0, 2*math.pi)
        s = random.uniform(speed*0.3, speed)
        particles.append({
            'x': float(x), 'y': float(y),
            'vx': math.cos(a)*s, 'vy': math.sin(a)*s,
            'life': life, 'max_life': life,
            'col': col, 'size': size, 'gravity': gravity,
            'type': 'normal'
        })

def spawn_trail(x, y, col, size=3):
    particles.append({
        'x': float(x), 'y': float(y),
        'vx': random.uniform(-1,1), 'vy': random.uniform(-1,1),
        'life': 0.3, 'max_life': 0.3,
        'col': col, 'size': size, 'gravity': 0, 'type': 'trail'
    })

def update_particles(frame):
    dead=[]
    for p in particles:
        p['x'] += p['vx']; p['y'] += p['vy']
        if p['type'] == 'normal':
            p['vy'] += p['gravity']; p['vx'] *= 0.95
        p['life'] -= 0.035
        if p['life'] <= 0: dead.append(p); continue
        alpha = p['life'] / p['max_life']
        c = tuple(min(255, int(v*alpha)) for v in p['col'])
        r = max(1, int(p['size']*alpha))
        px, py = int(p['x']), int(p['y'])
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(frame, (px,py), r, c, -1, cv2.LINE_AA)
            if r > 2:
                cv2.circle(frame, (px,py), r+2, c, 1, cv2.LINE_AA)
    for d in dead:
        if d in particles: particles.remove(d)

# ── Power-up drops ────────────────────────────────────────────
POWERUP_TYPES = {
    'MULTISHOT': {'col': NEON_PINK,   'symbol': '>>',  'duration': 8.0,  'label': 'MULTI-SHOT'},
    'RAPIDFIRE': {'col': NEON_YELLOW, 'symbol': '>>>', 'duration': 6.0,  'label': 'RAPID FIRE'},
    'SHIELD':    {'col': NEON_GREEN,  'symbol': 'SH',  'duration': 0,    'label': 'SHIELD +1'},
    'ENERGY':    {'col': NEON_CYAN,   'symbol': 'EN',  'duration': 0,    'label': 'ENERGY REFILL'},
    'BOMB':      {'col': NEON_ORANGE, 'symbol': 'BM',  'duration': 0,    'label': 'BOMB READY'},
}

class PowerUp:
    def __init__(self, x, y, kind):
        self.x = float(x); self.y = float(y)
        self.kind = kind
        info = POWERUP_TYPES[kind]
        self.col = info['col']
        self.symbol = info['symbol']
        self.label = info['label']
        self.alive = True
        self.t = 0
        self.size = 18
        self.vy = -0.5  # float upward

    def update(self):
        self.t += 1
        self.y += self.vy + math.sin(self.t*0.05)*0.3

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        if not (0 < cx < W and 0 < cy < H): return
        # glow ring
        pulse = int(5 + math.sin(self.t*0.1)*3)
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), self.size + pulse + 4, self.col, -1)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
        cv2.circle(frame, (cx,cy), self.size, self.col, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), self.size-4, tuple(v//3 for v in self.col), -1)
        cv2.putText(frame, self.symbol, (cx-10, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

# ── Bullet ────────────────────────────────────────────────────
class Bullet:
    def __init__(self, x, y, tx, ty, power=1, angle_offset=0, col=None):
        self.x = float(x); self.y = float(y)
        self.alive = True; self.power = power
        d = max(math.sqrt((tx-x)**2+(ty-y)**2), 1)
        spd = 22
        base_angle = math.atan2(ty-y, tx-x)
        a = base_angle + math.radians(angle_offset)
        self.vx = math.cos(a)*spd
        self.vy = math.sin(a)*spd
        self.trail = []
        self.col = col or (NEON_CYAN if power==1 else NEON_YELLOW)

    def update(self, frame):
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 12: self.trail.pop(0)
        self.x += self.vx; self.y += self.vy
        if not (0 < self.x < W and 0 < self.y < H):
            self.alive = False; return
        for i, pt in enumerate(self.trail):
            alpha = i/len(self.trail)
            c = tuple(int(v*alpha) for v in self.col)
            r = max(1, int(5*alpha))
            cv2.circle(frame, pt, r, c, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(self.x),int(self.y)), 7, self.col, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(self.x),int(self.y)), 11, self.col, 1, cv2.LINE_AA)
        cv2.circle(frame, (int(self.x),int(self.y)), 16, tuple(v//3 for v in self.col), 1, cv2.LINE_AA)
        spawn_trail(self.x, self.y, self.col)

# ── Virus ─────────────────────────────────────────────────────
VIRUS_TYPES = [
    {'name':'WORM',    'col':NEON_RED,    'hp':1, 'spd':1.8, 'size':18, 'reward':10},
    {'name':'TROJAN',  'col':NEON_ORANGE, 'hp':3, 'spd':1.2, 'size':24, 'reward':30},
    {'name':'ROOTKIT', 'col':NEON_PURPLE, 'hp':6, 'spd':0.8, 'size':32, 'reward':60},
    {'name':'ZERO-DAY','col':NEON_YELLOW, 'hp':2, 'spd':3.0, 'size':14, 'reward':50},
    {'name':'PHANTOM', 'col':NEON_PINK,   'hp':2, 'spd':2.5, 'size':16, 'reward':45,
     'blink': True},  # blinks in/out of visibility
]

class Virus:
    def __init__(self, level=1):
        pool = VIRUS_TYPES[:min(level+1, len(VIRUS_TYPES))]
        vt = random.choice(pool)
        self.name = vt['name']
        self.col  = vt['col']
        self.max_hp = vt['hp'] + level//3
        self.hp   = self.max_hp
        self.spd  = vt['spd'] + level*0.05
        self.size = vt['size']
        self.reward = vt['reward'] + level*5
        self.blink_type = vt.get('blink', False)
        self.alive = True
        self.t     = 0
        self.wobble = random.uniform(0, math.pi*2)
        self.drop_powerup = (random.random() < 0.18)
        edge = random.randint(0,3)
        if   edge==0: self.x=random.randint(50,W-50); self.y=-30
        elif edge==1: self.x=W+30; self.y=random.randint(50,H-50)
        elif edge==2: self.x=random.randint(50,W-50); self.y=H+30
        else:         self.x=-30; self.y=random.randint(50,H-50)
        self.tx = W//2; self.ty = H//2
        self.hit_flash = 0
        self.angle = random.uniform(0, math.pi*2)
        self.slowmo = False

    def update(self):
        self.t += 1
        self.angle += 0.04
        spd = self.spd * (0.4 if self.slowmo else 1.0)
        dx = self.tx-self.x; dy = self.ty-self.y
        d = max(math.sqrt(dx*dx+dy*dy), 1)
        wobble = math.sin(self.t*0.1+self.wobble)*0.3
        self.x += dx/d*spd + math.cos(self.t*0.05)*wobble*spd
        self.y += dy/d*spd + math.sin(self.t*0.05)*wobble*spd
        if self.hit_flash > 0: self.hit_flash -= 1

    @property
    def visible(self):
        if self.blink_type:
            return (self.t // 20) % 2 == 0
        return True

    def draw(self, frame):
        if not self.visible: return
        cx, cy = int(self.x), int(self.y)
        s = self.size
        col = WHITE if self.hit_flash > 0 else self.col
        cv2.circle(frame, (cx,cy), s+4, tuple(v//3 for v in col), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), s+2, tuple(v//2 for v in col), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), s, col, 2, cv2.LINE_AA)
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), s-2, tuple(v//6 for v in col), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        for i in range(8):
            a = self.angle + i*math.pi/4
            x1 = int(cx + math.cos(a)*(s-2))
            y1 = int(cy + math.sin(a)*(s-2))
            x2 = int(cx + math.cos(a)*(s+8+math.sin(self.t*0.08+i)*3))
            y2 = int(cy + math.sin(a)*(s+8+math.sin(self.t*0.08+i)*3))
            cv2.line(frame, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
        bw = s*2; hp_frac = self.hp/self.max_hp
        cv2.rectangle(frame, (cx-s,cy-s-14), (cx+s,cy-s-6), (30,30,30), -1)
        cv2.rectangle(frame, (cx-s,cy-s-14), (cx-s+int(bw*hp_frac),cy-s-6), col, -1)
        cv2.rectangle(frame, (cx-s,cy-s-14), (cx+s,cy-s-6), col, 1)
        cv2.putText(frame, self.name, (cx-s,cy-s-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)

    def hit(self, dmg=1):
        if not self.visible and self.blink_type:
            return False  # can't damage phantom when invisible
        self.hp -= dmg
        self.hit_flash = 6
        spawn_particles(self.x, self.y, self.col, n=10, speed=5, life=0.5, size=3)
        if self.hp <= 0:
            self.alive = False
            spawn_particles(self.x, self.y, self.col, n=40, speed=10, life=1.2, size=5)
            return True
        return False

# ── Boss enemy ────────────────────────────────────────────────
class Boss:
    PHASES = [
        {'col': NEON_RED,    'hp_frac': 1.0,  'spd': 0.6, 'fire_rate': 120},
        {'col': NEON_ORANGE, 'hp_frac': 0.6,  'spd': 1.0, 'fire_rate': 80},
        {'col': NEON_PURPLE, 'hp_frac': 0.3,  'spd': 1.5, 'fire_rate': 45},
    ]

    def __init__(self, level=1):
        self.max_hp = 40 + level*20
        self.hp     = self.max_hp
        self.size   = 55
        self.phase  = 0
        self.alive  = True
        self.t      = 0
        self.angle  = 0
        self.hit_flash = 0
        self.fire_t = 0
        self.minions_spawned = [False, False]  # phase 1, phase 2 minion spawns
        self.slowmo = False
        self.reward = 2000 + level*500
        # Spawn from right edge
        self.x = float(W + 80)
        self.y = float(H // 2)
        self.vx = -0.8
        self.vy = 0.0
        self.target_x = W * 0.75
        self.entered = False
        self.warning_shown = False
        self.orbit_angle = 0

    @property
    def col(self):
        frac = self.hp / self.max_hp
        for ph in reversed(self.PHASES):
            if frac <= ph['hp_frac']:
                return ph['col']
        return NEON_RED

    @property
    def fire_rate(self):
        frac = self.hp / self.max_hp
        for ph in reversed(self.PHASES):
            if frac <= ph['hp_frac']:
                return ph['fire_rate']
        return self.PHASES[0]['fire_rate']

    def update(self, viruses, level):
        self.t += 1
        self.angle += 0.025
        if not self.entered:
            self.x += self.vx
            if self.x <= self.target_x:
                self.entered = True
        else:
            spd = 0.5 * (0.4 if self.slowmo else 1.0)
            self.orbit_angle += 0.008
            cx = W * 0.65 + math.cos(self.orbit_angle) * 150
            cy = H/2 + math.sin(self.orbit_angle*0.7) * 160
            dx = cx - self.x; dy = cy - self.y
            d = max(math.sqrt(dx*dx+dy*dy), 1)
            self.x += dx/d * spd * 3
            self.y += dy/d * spd * 3
        if self.hit_flash > 0: self.hit_flash -= 1
        # spawn minions at phase transitions
        frac = self.hp / self.max_hp
        if frac < 0.6 and not self.minions_spawned[0]:
            self.minions_spawned[0] = True
            for _ in range(4): viruses.append(Virus(level))
            return 'MINION_WAVE'
        if frac < 0.3 and not self.minions_spawned[1]:
            self.minions_spawned[1] = True
            for _ in range(6): viruses.append(Virus(level))
            return 'MINION_WAVE'
        return None

    def shoot(self):
        """Returns list of bullets if it's time to fire."""
        if not self.entered: return []
        self.fire_t += 1
        if self.fire_t < self.fire_rate: return []
        self.fire_t = 0
        bullets = []
        num = 8 if self.hp/self.max_hp < 0.3 else (5 if self.hp/self.max_hp < 0.6 else 3)
        for i in range(num):
            a = self.angle + i * (2*math.pi/num)
            tx = self.x + math.cos(a)*300
            ty = self.y + math.sin(a)*300
            bullets.append(Bullet(self.x, self.y, tx, ty, power=-1, col=self.col))
        return bullets

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        s = self.size
        col = WHITE if self.hit_flash > 0 else self.col
        # outer glow
        for r_off, alpha in [(20, 0.04), (12, 0.08), (6, 0.15)]:
            ov = frame.copy()
            cv2.circle(ov, (cx,cy), s+r_off, col, -1)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
        # rings
        cv2.circle(frame, (cx,cy), s+8, tuple(v//4 for v in col), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), s+4, tuple(v//2 for v in col), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), s,   col, 3, cv2.LINE_AA)
        # inner fill
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), s-3, tuple(v//5 for v in col), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        # 12 rotating spikes
        for i in range(12):
            a = self.angle + i*math.pi/6
            a2 = self.angle*0.5 + i*math.pi/6
            x1 = int(cx + math.cos(a)*(s-3))
            y1 = int(cy + math.sin(a)*(s-3))
            x2 = int(cx + math.cos(a)*(s+14+math.sin(self.t*0.06+i)*5))
            y2 = int(cy + math.sin(a)*(s+14+math.sin(self.t*0.06+i)*5))
            cv2.line(frame, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
        # inner rotating hex
        for i in range(6):
            a = self.angle*2 + i*math.pi/3
            x1 = int(cx + math.cos(a)*28); y1 = int(cy + math.sin(a)*28)
            a2 = self.angle*2 + (i+1)*math.pi/3
            x2 = int(cx + math.cos(a2)*28); y2 = int(cy + math.sin(a2)*28)
            cv2.line(frame, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
        # BOSS label + HP
        cv2.putText(frame, 'BOSS', (cx-20, cy-s-24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        bw = s*2; hp_frac = self.hp/self.max_hp
        cv2.rectangle(frame, (cx-s, cy-s-18), (cx+s, cy-s-8), (30,30,30), -1)
        cv2.rectangle(frame, (cx-s, cy-s-18), (cx-s+int(bw*hp_frac), cy-s-8), col, -1)
        cv2.rectangle(frame, (cx-s, cy-s-18), (cx+s, cy-s-8), col, 1)
        # phase indicator
        frac = self.hp / self.max_hp
        phase_txt = 'PHASE III' if frac < 0.3 else ('PHASE II' if frac < 0.6 else 'PHASE I')
        cv2.putText(frame, phase_txt, (cx-30, cy-s-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    def hit(self, dmg=1):
        self.hp -= dmg
        self.hit_flash = 8
        spawn_particles(self.x, self.y, self.col, n=12, speed=6, life=0.6, size=4)
        if self.hp <= 0:
            self.alive = False
            spawn_particles(self.x, self.y, self.col, n=80, speed=14, life=2.0, size=7)
            spawn_particles(self.x, self.y, WHITE, n=40, speed=18, life=1.5, size=5)
            return True
        return False

# ── Enemy bullets (boss projectiles) ─────────────────────────
enemy_bullets = []

# ── Explosion ─────────────────────────────────────────────────
class Explosion:
    def __init__(self, x, y, r=120, col=NEON_CYAN):
        self.x=x; self.y=y; self.max_r=r
        self.r=0; self.alive=True; self.col=col; self.t=0
    def update(self, frame):
        self.t += 4; self.r = self.t
        alpha = max(0, 1 - self.t/self.max_r)
        if alpha <= 0: self.alive=False; return
        c = tuple(int(v*alpha) for v in self.col)
        cv2.circle(frame, (int(self.x),int(self.y)), int(self.r), c, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(self.x),int(self.y)), max(1,int(self.r*0.6)),
                   tuple(int(v*alpha*0.4) for v in self.col), 3, cv2.LINE_AA)

# ── Shockwave ─────────────────────────────────────────────────
class Shockwave:
    def __init__(self, x, y, col=NEON_YELLOW):
        self.x=x; self.y=y; self.r=10; self.alive=True; self.col=col
        self.max_r=min(W,H)*0.9
    def update(self, frame):
        self.r += 35
        if self.r > self.max_r: self.alive=False; return
        alpha = max(0, 1 - self.r/self.max_r)
        c = tuple(int(v*alpha) for v in self.col)
        t = max(1, int(4*alpha))
        cv2.circle(frame, (int(self.x),int(self.y)), int(self.r), c, t, cv2.LINE_AA)
        c2 = tuple(int(v*alpha*0.3) for v in self.col)
        cv2.circle(frame, (int(self.x),int(self.y)), max(1,int(self.r*0.85)), c2, t+1, cv2.LINE_AA)

# ── Screen effects ─────────────────────────────────────────────
shake_frames=0; shake_intensity=0; glitch_frames=0

def trigger_shake(intensity=8, frames=6):
    global shake_frames, shake_intensity
    shake_frames=frames; shake_intensity=intensity

def trigger_glitch(frames=4):
    global glitch_frames
    glitch_frames=frames

def apply_shake(frame):
    global shake_frames
    if shake_frames <= 0: return frame
    dx = random.randint(-shake_intensity, shake_intensity)
    dy = random.randint(-shake_intensity, shake_intensity)
    M = np.float32([[1,0,dx],[0,1,dy]])
    shake_frames -= 1
    return cv2.warpAffine(frame, M, (W,H))

def apply_glitch(frame):
    global glitch_frames
    if glitch_frames <= 0: return frame
    glitch_frames -= 1
    out = frame.copy()
    for _ in range(random.randint(2,5)):
        y1 = random.randint(0, H-20)
        h2 = random.randint(5, 20)
        shift = random.randint(-30, 30)
        out[y1:y1+h2] = np.roll(out[y1:y1+h2], shift, axis=1)
    b, g, r = cv2.split(out)
    out = cv2.merge([np.roll(b,2,axis=1), g, np.roll(r,-2,axis=1)])
    return out

def apply_scanlines(frame):
    """Subtle CRT scanline effect."""
    for y in range(0, H, 4):
        frame[y:y+1] = (frame[y:y+1].astype(np.float32) * 0.75).astype(np.uint8)
    return frame

def apply_vignette(frame):
    """Dark corners vignette."""
    vig = np.zeros((H, W), np.float32)
    cv2.ellipse(vig, (W//2, H//2), (W//2, H//2), 0, 0, 360, 1.0, -1)
    vig = cv2.GaussianBlur(vig, (0, 0), W//4)
    vig = np.clip(vig * 1.3, 0, 1)
    for c in range(3):
        frame[:,:,c] = (frame[:,:,c].astype(np.float32) * vig).astype(np.uint8)
    return frame

# ── Core ──────────────────────────────────────────────────────
core_pulse = 0

def draw_core(frame, lives, max_lives=3, shield_active=False):
    global core_pulse
    core_pulse += 0.08
    cx, cy = W//2, H//2
    frac = lives/max_lives
    col = NEON_GREEN if frac > 0.6 else NEON_ORANGE if frac > 0.3 else NEON_RED
    for r, a in [(60,0.15),(50,0.25),(38,0.4)]:
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), r, col, -1)
        cv2.addWeighted(ov, a*0.3, frame, 1-a*0.3, 0, frame)
        cv2.circle(frame, (cx,cy), r, col, 1, cv2.LINE_AA)
    pr = int(65 + math.sin(core_pulse)*8)
    cv2.circle(frame, (cx,cy), pr, tuple(v//2 for v in col), 1, cv2.LINE_AA)
    for i in range(6):
        a = i*math.pi/3 + core_pulse*0.2
        x1 = int(cx + math.cos(a)*35); y1 = int(cy + math.sin(a)*35)
        a2 = (i+1)*math.pi/3 + core_pulse*0.2
        x2 = int(cx + math.cos(a2)*35); y2 = int(cy + math.sin(a2)*35)
        cv2.line(frame, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
    cv2.putText(frame, 'CORE', (cx-14,cy+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
    for i in range(max_lives):
        c2 = col if i < lives else (40,40,40)
        cv2.circle(frame, (cx-20+i*20, cy+52), 6, c2, -1, cv2.LINE_AA)
    # Shield visual arc around core
    if shield_active:
        for i in range(36):
            a = i*math.pi/18 + core_pulse
            x1 = int(cx + math.cos(a)*78); y1 = int(cy + math.sin(a)*78)
            a2 = a + math.pi/20
            x2 = int(cx + math.cos(a2)*78); y2 = int(cy + math.sin(a2)*78)
            alpha = 0.5 + 0.5*math.sin(core_pulse*3 + i)
            c = tuple(int(v*alpha) for v in NEON_GREEN)
            cv2.line(frame, (x1,y1), (x2,y2), c, 2, cv2.LINE_AA)

# ── Energy bar ────────────────────────────────────────────────
MAX_ENERGY = 100.0
FIRE_COST  = 8.0
ENERGY_REGEN = 12.0  # per second

def draw_energy_bar(frame, energy):
    bx, by, bw, bh = 10, H-60, 200, 14
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (20,20,35), -1)
    frac = energy/MAX_ENERGY
    col = NEON_CYAN if frac > 0.5 else NEON_YELLOW if frac > 0.25 else NEON_RED
    cv2.rectangle(frame, (bx,by), (bx+int(bw*frac),by+bh), col, -1)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), col, 1)
    cv2.putText(frame, f'ENERGY {int(energy)}%', (bx+4,by+11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

# ── XP / Level bar ────────────────────────────────────────────
def draw_xp_bar(frame, xp, xp_next, player_level):
    bx, by, bw, bh = W//2 - 120, H-58, 240, 12
    frac = min(xp/xp_next, 1.0)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (20,20,35), -1)
    for px in range(int(bw*frac)):
        progress = px / bw
        r = int(NEON_PURPLE[0] + (NEON_CYAN[0]-NEON_PURPLE[0])*progress)
        g = int(NEON_PURPLE[1] + (NEON_CYAN[1]-NEON_PURPLE[1])*progress)
        b = int(NEON_PURPLE[2] + (NEON_CYAN[2]-NEON_PURPLE[2])*progress)
        cv2.line(frame, (bx+px,by), (bx+px,by+bh), (r,g,b), 1)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), NEON_PURPLE, 1)
    cv2.putText(frame, f'LVL {player_level}  XP {xp}/{xp_next}',
                (bx+4, by+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

# ── HUD ───────────────────────────────────────────────────────
def draw_hud(frame, score, level, combo, wave, shield_active,
             slowmo_active, bomb_cooldown, fire_cooldown, gx, gy,
             high_score, blink_power, boss_incoming):
    # top bar with gradient
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (W,48), (6,6,14), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0,48), (W,48), NEON_CYAN, 1)

    # score + high score
    cv2.putText(frame, f'SCORE: {score:07d}', (16,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f'BEST: {high_score:07d}', (16,48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, NEON_BLUE, 1, cv2.LINE_AA)
    # level
    cv2.putText(frame, f'LEVEL {level}', (250,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, NEON_GREEN, 1, cv2.LINE_AA)
    # combo
    if combo > 1:
        cc = NEON_YELLOW if combo < 5 else NEON_ORANGE if combo < 10 else NEON_RED
        size = 0.65 + min(combo*0.03, 0.4)
        cv2.putText(frame, f'x{combo} COMBO', (370,30),
                    cv2.FONT_HERSHEY_SIMPLEX, size, cc, 2, cv2.LINE_AA)
    # wave
    cv2.putText(frame, f'WAVE {wave}', (580,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_PURPLE, 1, cv2.LINE_AA)
    # blink power indicator
    if blink_power > 0:
        bcol = NEON_CYAN if blink_power > 0.5 else NEON_YELLOW
        cv2.putText(frame, f'[BLINK] NOVA: {int(blink_power*100)}%',
                    (W-260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bcol, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'GAZE ({gx:+.2f},{gy:+.2f})',
                    (W-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, NEON_PURPLE, 1, cv2.LINE_AA)
    # bottom abilities
    y2 = H-10
    scol = NEON_GREEN if shield_active else (80,200,100)
    cv2.putText(frame, f'[FIST] SHIELD', (16,y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, scol, 1, cv2.LINE_AA)
    smcol = NEON_CYAN if slowmo_active else (60,140,160)
    cv2.putText(frame, '[OPEN] SLOW-MO', (200,y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, smcol, 1, cv2.LINE_AA)
    bcol2 = NEON_YELLOW if bomb_cooldown == 0 else (80,80,40)
    blabel = f'[PEACE] BOMB: {"READY" if bomb_cooldown==0 else f"CD {bomb_cooldown}s"}'
    cv2.putText(frame, blabel, (420,y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, bcol2, 1, cv2.LINE_AA)
    fcol = NEON_CYAN if fire_cooldown == 0 else (40,80,100)
    cv2.putText(frame, '[PINCH] FIRE' if fire_cooldown==0 else '[PINCH] RELOAD...',
                (680,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fcol, 1, cv2.LINE_AA)
    cv2.putText(frame, '[BLINK] NOVA', (870,y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, NEON_CYAN, 1, cv2.LINE_AA)
    # boss warning
    if boss_incoming:
        pulse = int(255 * (0.5 + 0.5*math.sin(time.time()*6)))
        cv2.putText(frame, '!! BOSS INCOMING !!', (W//2-150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, pulse//2, pulse), 2, cv2.LINE_AA)

# ── Kill-streak announcer ─────────────────────────────────────
STREAK_LABELS = {
    2:  ('DOUBLE KILL',   NEON_YELLOW),
    3:  ('TRIPLE KILL',   NEON_ORANGE),
    5:  ('KILLING SPREE', NEON_RED),
    7:  ('RAMPAGE',       NEON_PINK),
    10: ('GODLIKE!!!',    NEON_CYAN),
    15: ('LEGENDARY!!!',  WHITE),
}

def check_streak(combo):
    if combo in STREAK_LABELS:
        return STREAK_LABELS[combo]
    return None

# ── Reticle ───────────────────────────────────────────────────
reticle_t = 0
def draw_reticle(frame, x, y, firing=False, shield=False, blink_charged=False):
    global reticle_t
    reticle_t += 0.12
    col = NEON_LIME if blink_charged else (NEON_GREEN if shield else (NEON_RED if firing else NEON_CYAN))
    r1, r2 = 22, 30
    for i in range(4):
        a = reticle_t + i*math.pi/2
        x1 = int(x + math.cos(a)*r1); y1 = int(y + math.sin(a)*r1)
        x2 = int(x + math.cos(a+0.8)*r2); y2 = int(y + math.sin(a+0.8)*r2)
        cv2.line(frame, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
    cv2.circle(frame, (int(x),int(y)), 5, col, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(x),int(y)), 9, col, 1, cv2.LINE_AA)
    cv2.circle(frame, (int(x),int(y)), r1, tuple(v//3 for v in col), 1, cv2.LINE_AA)
    cv2.line(frame, (int(x)-18,int(y)), (int(x)-8,int(y)), col, 1, cv2.LINE_AA)
    cv2.line(frame, (int(x)+8, int(y)), (int(x)+18,int(y)), col, 1, cv2.LINE_AA)
    cv2.line(frame, (int(x),int(y)-18), (int(x),int(y)-8), col, 1, cv2.LINE_AA)
    cv2.line(frame, (int(x),int(y)+8), (int(x),int(y)+18), col, 1, cv2.LINE_AA)
    if blink_charged:
        cv2.circle(frame, (int(x),int(y)), r1+10, tuple(v//4 for v in col), 1, cv2.LINE_AA)
    if firing: spawn_trail(x, y, col)

# ── Grid background ───────────────────────────────────────────
grid_t = 0
def draw_grid(frame):
    global grid_t
    grid_t += 0.5
    for x in range(0, W, 80):
        alpha = 0.05 + 0.015*math.sin(grid_t*0.05+x*0.01)
        col = tuple(int(v*alpha) for v in NEON_CYAN)
        cv2.line(frame, (x,0), (x,H), col, 1)
    for y in range(0, H, 80):
        alpha = 0.05 + 0.015*math.sin(grid_t*0.05+y*0.01)
        col = tuple(int(v*alpha) for v in NEON_CYAN)
        cv2.line(frame, (0,y), (W,y), col, 1)
    # horizon perspective lines from center
    cx, cy = W//2, H//2
    for i in range(8):
        a = i*math.pi/4 + grid_t*0.003
        ex = int(cx + math.cos(a)*W)
        ey = int(cy + math.sin(a)*W)
        col = tuple(int(v*0.025) for v in NEON_PURPLE)
        cv2.line(frame, (cx,cy), (ex,ey), col, 1)

# ── Gesture helpers ───────────────────────────────────────────
def finger_up(lm, tip, base):
    return lm.landmark[tip].y < lm.landmark[base].y

def is_pinch(lm):
    tx=lm.landmark[4].x; ty=lm.landmark[4].y
    ix=lm.landmark[8].x; iy=lm.landmark[8].y
    return math.sqrt((tx-ix)**2+(ty-iy)**2) < 0.07

def is_fist(lm):
    up = [finger_up(lm,FINGER_TIPS[i],FINGER_BASES[i]) for i in range(5)]
    return sum(up) == 0

def is_open(lm):
    up = [finger_up(lm,FINGER_TIPS[i],FINGER_BASES[i]) for i in range(5)]
    return sum(up) == 5

def is_peace(lm):
    up = [finger_up(lm,FINGER_TIPS[i],FINGER_BASES[i]) for i in range(5)]
    return up[1] and up[2] and not up[3] and not up[4]

# ── Wave system ───────────────────────────────────────────────
def spawn_wave(level, wave):
    count = 3 + level*2 + wave
    return [Virus(level) for _ in range(min(count, 18))]

# ── Start screen ──────────────────────────────────────────────
def draw_start_screen(frame, t, high_score):
    overlay = np.zeros_like(frame)
    cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
    # Title glow
    pulse = 0.6 + 0.4*math.sin(t*3)
    tc = tuple(int(v*pulse) for v in NEON_CYAN)
    cv2.putText(frame, 'ANTIGRAVITY', (W//2-220, H//2-120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, tc, 3, cv2.LINE_AA)
    cv2.putText(frame, 'NEURAL DEFENSE  v4.0', (W//2-220, H//2-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, NEON_PURPLE, 2, cv2.LINE_AA)
    # Instructions
    lines = [
        ('PINCH  → Fire',       NEON_CYAN),
        ('FIST   → Shield',     NEON_GREEN),
        ('OPEN   → Slow-Mo',    NEON_BLUE),
        ('PEACE  → Bomb',       NEON_YELLOW),
        ('BLINK  → Eye Nova',   NEON_PINK),
        ('',                    WHITE),
        ('Look to AIM  |  Gestures to FIGHT', WHITE),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (W//2-190, H//2+20+i*32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    # Blinking start prompt
    if int(t*2) % 2 == 0:
        cv2.putText(frame, 'PINCH TO START', (W//2-130, H//2+280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f'BEST SCORE: {high_score:07d}', (W//2-120, H//2+320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_YELLOW, 1, cv2.LINE_AA)

# ── Pause screen ──────────────────────────────────────────────
def draw_pause_screen(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (W//2-200, H//2-100), (W//2+200, H//2+100), (8,8,16), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (W//2-200, H//2-100), (W//2+200, H//2+100), NEON_CYAN, 2)
    cv2.putText(frame, '// PAUSED //', (W//2-110, H//2-30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, NEON_CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, 'P = Resume  |  R = Restart  |  Q = Quit',
                (W//2-190, H//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)

# ── Game-over screen ──────────────────────────────────────────
def draw_game_over(frame, score, high_score, t):
    overlay = frame.copy()
    cv2.rectangle(overlay, (180,180), (W-180,480), (8,8,16), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (180,180), (W-180,480), NEON_RED, 2)
    pulse = 0.6 + 0.4*math.sin(t*4)
    tc = tuple(int(v*pulse) for v in NEON_RED)
    cv2.putText(frame, 'SYSTEM COMPROMISED', (210,260),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, tc, 2, cv2.LINE_AA)
    cv2.putText(frame, f'FINAL SCORE: {score:07d}', (270,320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, NEON_CYAN, 1, cv2.LINE_AA)
    new_record = score >= high_score and score > 0
    if new_record:
        rc = tuple(int(v*pulse) for v in NEON_YELLOW)
        cv2.putText(frame, '★  NEW HIGH SCORE  ★', (270,365),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'BEST: {high_score:07d}', (340,365),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, 'R = Restart  |  Q = Quit', (310,430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

# ── Wave announce banner ───────────────────────────────────────
wave_banner = None
wave_banner_t = 0

def show_wave_banner(txt, col):
    global wave_banner, wave_banner_t
    wave_banner = (txt, col)
    wave_banner_t = 2.5

def draw_wave_banner(frame, dt):
    global wave_banner_t
    if wave_banner is None: return
    wave_banner_t -= dt
    if wave_banner_t <= 0:
        return
    txt, col = wave_banner
    alpha = min(1.0, wave_banner_t) * min(1.0, 2.5-wave_banner_t)
    c = tuple(int(v*alpha) for v in col)
    y = int(H//2 - 120)
    cv2.putText(frame, txt, (W//2-len(txt)*14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, c, 3, cv2.LINE_AA)

# ── Game state ────────────────────────────────────────────────
STATE = 'START'   # START / PLAYING / PAUSED / DEAD
score = 0
lives = 3
level = 1
wave  = 1
combo = 0
combo_timer = 0
player_level = 1
xp = 0
xp_next = 200
viruses = []
bullets = []
explosions = []
shockwaves = []
powerups   = []
boss       = None
boss_wave  = False
shield_active = False
slowmo_active = False
bomb_cooldown = 0
bomb_ts   = 0
fire_ts   = 0
fire_cooldown = 0
pinch_prev  = False
peace_prev  = False
gaze_x = W//2; gaze_y = H//2
prev_time  = time.time()
start_t    = time.time()
game_time  = 0.0
score_popups = []   # [x,y,text,col,life]
damage_flash = 0
energy     = MAX_ENERGY
multishot  = 0.0   # remaining seconds
rapidfire  = 0.0
blink_power  = 0.0       # charges up between blinks, releases eye nova
blink_nova_used = False
prev_ear_closed = False
blink_charge_rate = 0.15  # per blink
boss_incoming_warned = False

def reset_game():
    global score, lives, level, wave, combo, combo_timer, viruses, bullets
    global explosions, shockwaves, powerups, boss, boss_wave
    global shield_active, slowmo_active, bomb_ts, fire_ts, bomb_cooldown, fire_cooldown
    global pinch_prev, peace_prev, gaze_x, gaze_y, prev_time, game_time
    global score_popups, damage_flash, energy, multishot, rapidfire
    global blink_power, blink_nova_used, prev_ear_closed, player_level, xp, xp_next
    global start_t, boss_incoming_warned, wave_banner, wave_banner_t
    score=0; lives=3; level=1; wave=1; combo=0; combo_timer=0
    player_level=1; xp=0; xp_next=200
    viruses=[]; bullets=[]; explosions=[]; shockwaves=[]; powerups=[]
    boss=None; boss_wave=False
    shield_active=False; slowmo_active=False
    bomb_ts=0; fire_ts=0; bomb_cooldown=0; fire_cooldown=0
    pinch_prev=False; peace_prev=False; gaze_x=W//2; gaze_y=H//2
    prev_time=time.time(); game_time=0.0; start_t=time.time()
    score_popups=[]; damage_flash=0
    energy=MAX_ENERGY; multishot=0.0; rapidfire=0.0
    blink_power=0.0; blink_nova_used=False; prev_ear_closed=False
    boss_incoming_warned=False; wave_banner=None; wave_banner_t=0
    particles.clear(); enemy_bullets.clear()

# ── Main loop ─────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    now = time.time()
    dt  = min(now - prev_time, 0.05)
    prev_time = now

    # ── Background ────────────────────────────────────────────
    cv2.addWeighted(frame, 0.45, np.zeros_like(frame), 0.55, 0, frame)
    draw_stars(frame)
    draw_grid(frame)

    # ── Gaze tracking ─────────────────────────────────────────
    mesh_res = face_mesh.process(rgb)
    ear_val = 0.3  # default open
    if mesh_res.multi_face_landmarks:
        fm = mesh_res.multi_face_landmarks[0]
        def lmpt(i): return (fm.landmark[i].x*W, fm.landmark[i].y*H)
        try:
            l_iris = lmpt(L_IRIS)
            l_el = lmpt(L_EYE_L); l_er = lmpt(L_EYE_R)
            l_et = lmpt(L_EYE_T); l_eb = lmpt(L_EYE_B)
            eye_w = max(math.dist(l_el, l_er), 1)
            eye_h = max(math.dist(l_et, l_eb), 1)
            raw_gx = (l_iris[0]-l_el[0])/eye_w - 0.5
            raw_gy = (l_iris[1]-l_et[1])/eye_h - 0.5
            sgx = gx_sm.update(raw_gx)
            sgy = gy_sm.update(raw_gy)
            gaze_x = int(W//2 + sgx*W*1.4)
            gaze_y = int(H//2 + sgy*H*1.4)
            gaze_x = max(20, min(W-20, gaze_x))
            gaze_y = max(60, min(H-60, gaze_y))
            # EAR (blink)
            l_et2 = lmpt(L_EYE_T); l_eb2 = lmpt(L_EYE_B)
            ear_raw = math.dist(l_et2, l_eb2) / max(eye_w*0.5, 1)
            ear_val = ear_sm.update(ear_raw)
        except:
            pass

    # ── Blink power (Eye Nova) ─────────────────────────────────
    if STATE == 'PLAYING':
        eye_closed_now = ear_val < 0.18
        if eye_closed_now and not prev_ear_closed:
            # blink detected → charge
            blink_power = min(1.0, blink_power + blink_charge_rate)
        prev_ear_closed = eye_closed_now

        # Full charge (double blink sequence) → auto-release nova
        # We release it when eyes are open again and charge is full
        if blink_power >= 1.0 and not eye_closed_now and not blink_nova_used:
            blink_nova_used = True
            # Eye Nova: destroy all nearby enemies in huge radius
            spawn_particles(gaze_x, gaze_y, NEON_CYAN, n=100, speed=18, life=2.0, size=7, gravity=0)
            shockwaves.append(Shockwave(gaze_x, gaze_y, NEON_CYAN))
            trigger_shake(10, 8); trigger_glitch(5)
            killed = 0
            for v in viruses:
                if math.dist((v.x,v.y),(gaze_x,gaze_y)) < 300:
                    v.hp = 0; v.alive = False
                    spawn_particles(v.x, v.y, v.col, n=25, speed=8)
                    killed += 1
            if boss and math.dist((boss.x,boss.y),(gaze_x,gaze_y)) < 350:
                boss.hit(15)
            pts = killed * 80
            score += pts
            score_popups.append([gaze_x, gaze_y-40, f'👁 EYE NOVA! +{pts}', NEON_CYAN, 2.0])
            blink_power = 0.0
            viruses = [v for v in viruses if v.alive]
        if not blink_nova_used or eye_closed_now:
            blink_nova_used = False
        # slow blink-power decay if not used
        if blink_power > 0 and not eye_closed_now:
            blink_power = max(0, blink_power - 0.001)

    # ── Hand gestures ─────────────────────────────────────────
    hand_res = hands.process(rgb)
    pinch_now=False; fist_now=False; open_now=False; peace_now=False

    if hand_res.multi_hand_landmarks:
        for lm in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,180,80), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,80,40), thickness=1))
            if is_pinch(lm): pinch_now=True
            if is_fist(lm):  fist_now=True
            if is_open(lm):  open_now=True
            if is_peace(lm): peace_now=True

    # ────────────────── START SCREEN ──────────────────────────
    if STATE == 'START':
        draw_start_screen(frame, now - start_t, HIGH_SCORE)
        if pinch_now and not pinch_prev:
            STATE = 'PLAYING'
            reset_game()
            viruses = spawn_wave(level, wave)
            show_wave_banner(f'WAVE {wave}  —  ENGAGE!', NEON_GREEN)
        frame = apply_glitch(frame)
        frame = apply_shake(frame)
        cv2.imshow('ANTIGRAVITY : NEURAL DEFENSE v4.0', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        pinch_prev = pinch_now
        continue

    # ────────────────── PAUSED ────────────────────────────────
    if STATE == 'PAUSED':
        draw_core(frame, lives)
        draw_pause_screen(frame)
        cv2.imshow('ANTIGRAVITY : NEURAL DEFENSE v4.0', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'): STATE = 'PLAYING'
        if key == ord('r'): STATE='PLAYING'; reset_game(); viruses=spawn_wave(level,wave); show_wave_banner(f'WAVE {wave}  —  ENGAGE!', NEON_GREEN)
        if key == ord('q'): break
        continue

    # ────────────────── PLAYING ───────────────────────────────
    if STATE == 'PLAYING':
        game_time += dt
        energy = min(MAX_ENERGY, energy + ENERGY_REGEN * dt)
        multishot = max(0, multishot - dt)
        rapidfire = max(0, rapidfire - dt)

        # SHIELD
        shield_active = fist_now
        # SLOW-MO
        slowmo_active = open_now
        # BOMB
        if peace_now and not peace_prev:
            if now - bomb_ts > 6:
                bomb_ts = now
                shockwaves.append(Shockwave(W//2, H//2, NEON_YELLOW))
                explosions.append(Explosion(W//2, H//2, r=W, col=NEON_YELLOW))
                spawn_particles(W//2, H//2, NEON_YELLOW, n=100, speed=18, life=2.0, size=7)
                trigger_shake(14, 12); trigger_glitch(10)
                for v in viruses:
                    v.hp -= 999; v.alive = False
                    spawn_particles(v.x, v.y, v.col, n=30, speed=8)
                if boss: boss.hit(20)
                viruses = [v for v in viruses if v.alive]
                score += 500
                score_popups.append([W//2, H//3, 'SYSTEM BOMB! +500', NEON_YELLOW, 1.5])

        bomb_cooldown = max(0, int(6-(now-bomb_ts)))
        peace_prev = peace_now

        # FIRE
        fire_delay = 0.12 if rapidfire > 0 else 0.25
        if pinch_now and not pinch_prev:
            if energy >= FIRE_COST and now - fire_ts > fire_delay:
                fire_ts = now
                energy -= FIRE_COST
                power = 2 if combo > 5 else 1
                bul_col = NEON_PINK if multishot > 0 else (NEON_YELLOW if rapidfire > 0 else None)
                # Multishot: 3 bullets spread
                angles = [-12, 0, 12] if multishot > 0 else [0]
                for ao in angles:
                    bullets.append(Bullet(gaze_x, gaze_y,
                        gaze_x + random.randint(-5,5),
                        gaze_y - 200,
                        power=power, angle_offset=ao, col=bul_col))
                spawn_particles(gaze_x, gaze_y, NEON_CYAN, n=12, speed=6, life=0.4, size=3)

        fire_cooldown = max(0, round((fire_delay-(now-fire_ts))*10)/10)
        pinch_prev = pinch_now

        # ── Core ──────────────────────────────────────────────
        draw_core(frame, lives, shield_active=shield_active)

        # ── Boss enemy bullets vs core ─────────────────────────
        for eb in enemy_bullets:
            eb.update(frame)
            if math.dist((eb.x,eb.y),(W//2,H//2)) < 65 and eb.alive:
                if shield_active:
                    eb.alive = False
                    spawn_particles(eb.x, eb.y, NEON_GREEN, n=15, speed=6)
                else:
                    eb.alive = False
                    lives -= 1; damage_flash = 10
                    trigger_shake(14, 8); trigger_glitch(6)
                    spawn_particles(W//2, H//2, NEON_RED, n=40, speed=12, life=1.5, size=5)
                    combo = 0
        enemy_bullets[:] = [eb for eb in enemy_bullets if eb.alive]

        # ── Viruses ───────────────────────────────────────────
        for v in viruses:
            v.slowmo = slowmo_active
            v.update()
            if math.dist((v.x,v.y),(W//2,H//2)) < 62 and v.alive:
                if shield_active:
                    v.hit(999)
                    spawn_particles(v.x, v.y, NEON_GREEN, n=30, speed=8)
                    trigger_shake(6,4)
                    score_popups.append([v.x, v.y, 'SHIELD BLOCK!', NEON_GREEN, 1.0])
                else:
                    v.alive = False; lives -= 1; damage_flash = 8
                    trigger_shake(15,8); trigger_glitch(6)
                    spawn_particles(W//2, H//2, NEON_RED, n=50, speed=12, life=1.5, size=5)
                    combo = 0
                    score_popups.append([W//2, H//2-60, 'SYSTEM BREACHED!', NEON_RED, 1.2])
            v.draw(frame)
        viruses = [v for v in viruses if v.alive]

        # ── Boss ──────────────────────────────────────────────
        if boss:
            boss.slowmo = slowmo_active
            event = boss.update(viruses, level)
            if event == 'MINION_WAVE':
                show_wave_banner('BOSS SPAWNS MINIONS!', NEON_RED)
                trigger_shake(10, 6)
            new_boss_bullets = boss.shoot()
            enemy_bullets.extend(new_boss_bullets)
            # boss reaches core?
            if math.dist((boss.x,boss.y),(W//2,H//2)) < 80 and boss.alive and boss.entered:
                if shield_active:
                    boss.hit(2)
                    spawn_particles(boss.x, boss.y, NEON_GREEN, n=20, speed=8)
                else:
                    boss.alive = False; lives -= 1; damage_flash = 12
                    trigger_shake(20, 12); trigger_glitch(10)
                    spawn_particles(W//2, H//2, NEON_RED, n=80, speed=15, life=2.0, size=6)
            if not boss.alive:
                pts = boss.reward * (combo+1)
                score += pts
                combo += 1; combo_timer = now
                xp += boss.reward
                explosions.append(Explosion(boss.x, boss.y, r=200, col=boss.col))
                shockwaves.append(Shockwave(boss.x, boss.y, boss.col))
                trigger_shake(18, 14); trigger_glitch(12)
                score_popups.append([int(boss.x), int(boss.y)-40,
                                     f'BOSS ELIMINATED! +{pts}', NEON_YELLOW, 2.5])
                show_wave_banner('BOSS DESTROYED!', NEON_YELLOW)
                # drop big powerup
                powerups.append(PowerUp(boss.x, boss.y,
                                        random.choice(list(POWERUP_TYPES.keys()))))
                powerups.append(PowerUp(boss.x+40, boss.y,
                                        random.choice(list(POWERUP_TYPES.keys()))))
                boss = None; boss_wave = False
            else:
                boss.draw(frame)

        # ── Bullet ↔ virus/boss collision ─────────────────────
        for b in bullets:
            b.update(frame)
            for v in viruses:
                if v.alive and math.dist((b.x,b.y),(v.x,v.y)) < v.size+8:
                    killed = v.hit(b.power)
                    b.alive = False
                    if killed:
                        pts = v.reward * (combo+1)
                        score += pts; xp += v.reward
                        combo += 1; combo_timer = now
                        explosions.append(Explosion(v.x,v.y,r=80,col=v.col))
                        trigger_shake(5,4)
                        # drop powerup
                        if v.drop_powerup:
                            powerups.append(PowerUp(v.x, v.y,
                                random.choice(list(POWERUP_TYPES.keys()))))
                        # kill streak
                        streak = check_streak(combo)
                        if streak:
                            score_popups.append([W//2, H//3,
                                streak[0], streak[1], 2.0])
                            trigger_shake(8,6)
                        score_popups.append([int(v.x), int(v.y)-20,
                                             f'+{pts} x{combo}', v.col, 0.9])
                    break
            if b.alive and boss:
                if math.dist((b.x,b.y),(boss.x,boss.y)) < boss.size+8:
                    killed = boss.hit(b.power)
                    b.alive = False
        bullets = [b for b in bullets if b.alive]

        # XP & player leveling
        while xp >= xp_next:
            xp -= xp_next; player_level += 1
            xp_next = int(xp_next * 1.4)
            score_popups.append([W//2, H//2-100,
                                  f'LEVEL UP! {player_level}', NEON_LIME, 2.0])
            spawn_particles(W//2, H//2, NEON_LIME, n=60, speed=12, life=1.5, size=5)

        # combo decay
        if combo > 0 and now - combo_timer > 3:
            combo = 0

        # ── Power-ups ─────────────────────────────────────────
        for pu in powerups:
            pu.update(); pu.draw(frame)
            # collect if reticle near
            if math.dist((pu.x,pu.y),(gaze_x,gaze_y)) < 40 and pinch_now:
                pu.alive = False
                info = POWERUP_TYPES[pu.kind]
                score_popups.append([int(pu.x), int(pu.y)-30,
                                     f'{info["label"]} COLLECTED!', pu.col, 1.5])
                spawn_particles(pu.x, pu.y, pu.col, n=30, speed=8)
                if pu.kind == 'MULTISHOT': multishot = info['duration']
                elif pu.kind == 'RAPIDFIRE': rapidfire = info['duration']
                elif pu.kind == 'SHIELD': lives = min(lives+1, 3)
                elif pu.kind == 'ENERGY': energy = MAX_ENERGY
                elif pu.kind == 'BOMB': bomb_ts = 0  # reset cooldown
        powerups = [pu for pu in powerups if pu.alive and 0<pu.y<H]

        # ── Explosions + Shockwaves ───────────────────────────
        for e in explosions: e.update(frame)
        explosions = [e for e in explosions if e.alive]
        for sw in shockwaves: sw.update(frame)
        shockwaves = [sw for sw in shockwaves if sw.alive]

        # ── Particles ─────────────────────────────────────────
        update_particles(frame)

        # ── Score popups ──────────────────────────────────────
        dead_pop=[]
        for pop in score_popups:
            px, py, txt2, pc, pl = pop
            pop[4] -= dt
            alpha = min(1.0, pop[4])
            c = tuple(int(v*alpha) for v in pc)
            cv2.putText(frame, txt2, (int(px)-30, int(py-20*(1-pop[4]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2, cv2.LINE_AA)
            if pop[4] <= 0: dead_pop.append(pop)
        for d in dead_pop: score_popups.remove(d)

        # ── Wave banner ───────────────────────────────────────
        draw_wave_banner(frame, dt)

        # ── Reticle ───────────────────────────────────────────
        draw_reticle(frame, gaze_x, gaze_y,
                     firing=pinch_now, shield=shield_active,
                     blink_charged=(blink_power >= 1.0))

        # ── Damage flash ──────────────────────────────────────
        if damage_flash > 0:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (0,0,180), -1)
            cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)
            damage_flash -= 1

        # ── Slow-mo overlay ───────────────────────────────────
        if slowmo_active:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (80,0,80), -1)
            cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)
            cv2.putText(frame, '// SLOW-MO //', (W//2-80,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_CYAN, 1, cv2.LINE_AA)

        # ── Energy + XP bars ──────────────────────────────────
        draw_energy_bar(frame, energy)
        draw_xp_bar(frame, xp, xp_next, player_level)

        # ── HUD ───────────────────────────────────────────────
        draw_hud(frame, score, level, combo, wave,
                 shield_active, slowmo_active,
                 bomb_cooldown, fire_cooldown,
                 round(gx_sm.buf[-1] if gx_sm.buf else 0, 2),
                 round(gy_sm.buf[-1] if gy_sm.buf else 0, 2),
                 HIGH_SCORE, blink_power,
                 boss_incoming=(boss is not None and not boss.entered))

        # ── Active powerup indicators ─────────────────────────
        pi_y = H - 90
        if multishot > 0:
            cv2.putText(frame, f'MULTI-SHOT {multishot:.1f}s', (W-220, pi_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_PINK, 1, cv2.LINE_AA)
            pi_y -= 20
        if rapidfire > 0:
            cv2.putText(frame, f'RAPID FIRE {rapidfire:.1f}s', (W-220, pi_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_YELLOW, 1, cv2.LINE_AA)

        # ── Wave advancement ──────────────────────────────────
        if len(viruses) == 0 and boss is None:
            wave += 1
            # every 5 waves → boss wave
            if wave % 5 == 0:
                boss = Boss(level)
                boss_wave = True
                boss_incoming_warned = False
                score_popups.append([W//2-100, H//2-100, '!! BOSS WAVE !!', NEON_RED, 3.0])
                show_wave_banner('!! BOSS INCOMING !!', NEON_RED)
                trigger_shake(8, 6)
            else:
                if wave > 5: level += 1; wave = 1
                viruses = spawn_wave(level, wave)
                show_wave_banner(f'WAVE {wave}  —  ENGAGE!', NEON_GREEN)
                spawn_particles(W//2, H//2, NEON_GREEN, n=60, speed=12, life=1.5, size=4)

        # ── Game over ─────────────────────────────────────────
        if lives <= 0:
            STATE = 'DEAD'
            if score > HIGH_SCORE:
                HIGH_SCORE = score
                save_high_score(HIGH_SCORE)

    # ────────────────── DEAD SCREEN ───────────────────────────
    if STATE == 'DEAD':
        update_particles(frame)
        draw_game_over(frame, score, HIGH_SCORE, now - start_t)

    # ── Screen FX (always) ─────────────────────────────────────
    frame = apply_glitch(frame)
    frame = apply_shake(frame)
    frame = apply_scanlines(frame)
    frame = apply_vignette(frame)

    cv2.imshow('ANTIGRAVITY : NEURAL DEFENSE v4.0', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('p') and STATE == 'PLAYING': STATE = 'PAUSED'
    if key == ord('r') and STATE in ('DEAD', 'PLAYING'):
        STATE = 'PLAYING'; reset_game()
        viruses = spawn_wave(level, wave)
        show_wave_banner(f'WAVE {wave}  —  ENGAGE!', NEON_GREEN)

cap.release()
cv2.destroyAllWindows()
