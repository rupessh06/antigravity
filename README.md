# 🧠 ANTIGRAVITY — Vision & Neural Interface Project

> **A real-time computer vision system that tracks your eyes, hands, face and emotions — built from scratch, version by version.**

Built by **Rupesh Thakur** | Started April 2026

---

## 🚀 What Is Antigravity?

Antigravity is a personal computer vision project that turns your webcam into a full biometric + gesture interface. It started as a simple hand tracker and is evolving into an AI-powered sign language decoder, eye-tracking game engine, and neural interface system.

No fancy hardware. Just a webcam, Python, and MediaPipe.

---

## 📁 Project Versions

### `antigravity.py` — v0.1 · Hand Tracking Demo
> The beginning. Proof of concept.

**What it does:**
- Live webcam feed with face detection (green bounding box)
- Detects both hands using MediaPipe (21 landmark points per hand)
- Connects index fingertips with animated strings
- Two modes: **Strings** (S key) and **Elastic Band** (E key)
- Mode display HUD on screen

**Tech used:** OpenCV, MediaPipe Hands, MediaPipe Face Detection, NumPy

**How to run:**
```bash
python antigravity.py
```
**Controls:** `S` = strings | `E` = elastic | `Q` = quit

---

### `antigravity1.py` — v1.0 · Full Physics Hand System
> Strings got real physics. Gestures became meaningful.

**What it does:**
- All 5 fingers now connect with individual physics strings (not just index)
- Each finger pair has its own color: amber, pink, cyan, green, purple
- **Verlet physics simulation** — strings have real gravity, bounce, tension
- Full gesture detection system:
  - 🖐️ `OPEN` — spider web membrane appears between fingers
  - ✊ `FIST` then release — snap explosion with flying particles
  - 🤏 `PINCH` — pulls all strings toward fingertip like a magnet
  - ✌️ `PEACE` — only index + middle strings active
  - ☝️ `POINT` — single string mode
- Live gesture HUD showing finger state per hand
- Snap particle burst system on fist release

**New concepts introduced:**
- Verlet integration physics
- Per-finger gesture state tracking
- Particle explosion system

**How to run:**
```bash
python antigravity1.py
```
**Controls:** Gestures control everything | `Q` = quit

---

### `antigravity2.py` — v2.0 · Full Biometric Monitor
> The system now reads YOU — not just your hands.

**What it does:**
- Full **face mesh** (468 landmark points) rendered live
- 👁️ **Eye tracking** — iris position detected, gaze direction calculated (LEFT / RIGHT / UP / DOWN / CENTER)
- 😊 **Emotion detection** — HAPPY, SAD, ANGRY, SURPRISED, SLEEPY, NEUTRAL (geometry-based)
- 🎂 **Age estimation** — face size proxy bands
- 😮 **Mouth open/closed** detection
- 💤 **Blink detection** — counts blinks, measures blink rate per minute
- 📐 **Head pose** — pitch, yaw, roll in degrees (PnP solver)
- 🧠 **Attention score** — 0–100 live score based on gaze, blink rate, head angle
- Right side **dashboard panel** showing all metrics live
- Gaze crosshair dot that follows your eye direction on screen
- All v1.0 hand physics + gesture system included

**New concepts introduced:**
- MediaPipe FaceMesh with iris refinement
- Eye Aspect Ratio (EAR) for blink detection
- solvePnP for 3D head pose estimation
- Smoother class for signal stabilization
- Gaze direction mapping

**How to run:**
```bash
python antigravity2.py
```

---

### `antigravity3.py` — v3.0 · NEURAL DEFENSE (Eye Tracking Game)
> Your eyes aim. Your hands fire. Viruses attack your system core.

**What it does:**
- Full **eye-tracking game** — gaze controls the targeting reticle
- Viruses spawn from screen edges and crawl toward your CORE
- 3 lives system — viruses that reach the core drain your health
- **4 virus types:**
  - 🔴 `WORM` — fast, 1HP
  - 🟠 `TROJAN` — tanky, 3HP
  - 🟣 `ROOTKIT` — boss tier, 6HP
  - 🟡 `ZERO-DAY` — ultra fast, 2HP
- **Combo multiplier** — chain kills for score multipliers
- **Wave + Level system** — gets harder every 5 waves
- Score popups, damage flash, screen shake, glitch distortion
- Animated core with hex pattern and HP lives display

**Gesture controls in-game:**
| Gesture | Action |
|---------|--------|
| 🤏 PINCH | Fire plasma shot at gaze target |
| ✊ FIST | Activate shield — blocks next breach |
| 🖐️ OPEN HAND | Slow-motion field (viruses at 40% speed) |
| ✌️ PEACE SIGN | System bomb — destroys ALL viruses (6s cooldown) |

**Visual effects:**
- Neon grid background
- Bullet plasma trails
- Particle explosions on kills
- Screen shake on damage
- RGB glitch distortion
- Slow-mo purple tint overlay
- Animated reticle arcs

**How to run:**
```bash
python antigravity3.py
```
**Controls:** Eyes aim | Gestures fire | `R` = restart | `Q` = quit

---

## 🛠️ Installation

### Requirements
- Python **3.11.x** (MediaPipe does NOT support Python 3.12+)
- Webcam

### Install dependencies
```bash
pip install opencv-python mediapipe numpy
```

### Clone and run
```bash
git clone https://github.com/YOUR_USERNAME/antigravity.git
cd antigravity
python antigravity3.py
```

---

## 🗺️ Roadmap

| Version | Status | Description |
|---------|--------|-------------|
| v0.1 | ✅ Done | Hand tracking + string demo |
| v1.0 | ✅ Done | Full physics gesture system |
| v2.0 | ✅ Done | Full biometric monitor |
| v3.0 | ✅ Done | Eye tracking game — Neural Defense |
| v4.0 | 🔜 Next | Sign language gesture recorder |
| v5.0 | 🔜 Soon | AI voice decoder — speak with hands |
| v6.0 | 🔜 Future | Full Antigravity neural interface |

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| OpenCV | Camera feed, drawing, image processing |
| MediaPipe | Hand tracking, face mesh, iris detection |
| NumPy | Physics math, matrix operations |
| solvePnP | 3D head pose from 2D landmarks |

---

## 💡 Phase 2 — Sign Language AI (Coming)

The next phase turns Antigravity into a personal sign language system:

1. **Record your own gestures** — map hand poses to words/letters
2. **Train a classifier** — scikit-learn on MediaPipe landmark coordinates
3. **Real-time decoder** — detects your signs as you make them
4. **AI voice response** — Claude/OpenAI API understands intent, speaks back via speaker
5. **Agentic listener** — always-on system that responds to your custom sign vocabulary

---

## 📸 Screenshots

> *(Add your screenshots here as you go)*

---

## 👤 Author

**Rupesh Thakur**
Building Antigravity — one version at a time.

---

## 📄 License

MIT License — use it, build on it, make it yours.
