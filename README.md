# 🧠 ANTIGRAVITY — Vision & Neural Interface Project

> **A real-time computer vision system that tracks your eyes, hands, face and emotions — built version by version using AI-assisted development.**

Built by **Rupesh Thakur** | Started April 2026

---

## 🤝 Honest About How This Was Built

I'll be straight with you — I don't know how to code. Not a single line.

This entire project was built through a conversation between me and **Claude (by Anthropic)**. I had the idea, I described what I wanted, I tested every version on my own machine, I debugged errors by sharing them back, and I kept pushing for the next level.

Claude wrote the code. I directed it.

Think of it like this — a film director doesn't hold the camera. But the film is still theirs.

This project is proof that in 2026, **you don't need to know how to code to build real things**. You need:
- A clear vision
- The ability to describe what you want
- The persistence to keep going when things break
- And the curiosity to push further

Every feature in Antigravity came from my head. The hand strings, the physics, the biometric dashboard, the eye-tracking game, the virus defense system — I asked for all of it. I tested all of it. I broke it and fixed it.

**That's a new kind of skill. And I'm owning it.**

---

## 🚀 What Is Antigravity?

Antigravity is a computer vision project that turns your webcam into a full biometric + gesture interface. It started as a simple hand tracker and is evolving into an AI-powered sign language decoder, eye-tracking game engine, and neural interface system.

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
- All 5 fingers connect with individual physics strings
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

**How to run:**
```bash
python antigravity1.py
```

---

### `antigravity2.py` — v2.0 · Full Biometric Monitor
> The system now reads YOU — not just your hands.

**What it does:**
- Full **face mesh** (468 landmark points) rendered live
- 👁️ **Eye tracking** — iris position, gaze direction (LEFT / RIGHT / UP / DOWN / CENTER)
- 😊 **Emotion detection** — HAPPY, SAD, ANGRY, SURPRISED, SLEEPY, NEUTRAL
- 🎂 **Age estimation** — face proportion analysis
- 😮 **Mouth open/closed** detection
- 💤 **Blink detection** — counts blinks, blink rate per minute
- 📐 **Head pose** — pitch, yaw, roll in degrees
- 🧠 **Attention score** — 0 to 100 live score
- Right side dashboard panel showing all metrics live
- Gaze crosshair dot that follows your eye direction

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
- 3 lives system — lose a life when viruses breach the core
- **4 virus types:** WORM, TROJAN, ROOTKIT, ZERO-DAY
- Combo multiplier, wave system, score popups
- Screen shake, glitch distortion, neon particle explosions

**Gesture controls:**
| Gesture | Action |
|---------|--------|
| 🤏 PINCH | Fire plasma shot at gaze target |
| ✊ FIST | Activate shield |
| 🖐️ OPEN HAND | Slow-motion field |
| ✌️ PEACE SIGN | System bomb — destroys all viruses |

**How to run:**
```bash
python antigravity3.py
```
**Controls:** Eyes aim | Gestures fire | `R` = restart | `Q` = quit

---

## 🛠️ Installation

### Requirements
- Python **3.11.x** — MediaPipe does NOT support Python 3.12+
- A webcam

### Install dependencies
```bash
pip install opencv-python mediapipe numpy
```

### Clone and run
```bash
git clone https://github.com/rupessh06/antigravity.git
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
| Claude (Anthropic) | AI pair programmer — wrote all the code |

---

## 💡 Phase 2 — Sign Language AI (Coming Soon)

The next phase turns Antigravity into a personal sign language system:

1. **Record your own gestures** — map hand poses to words
2. **Train a classifier** — on MediaPipe landmark coordinates
3. **Real-time decoder** — detects signs as you make them
4. **AI voice response** — understands intent, speaks back via speaker
5. **Agentic listener** — always-on, responds to your custom sign vocabulary

---

## 🧠 What I Learned Building This

Even without writing a single line of code myself, building Antigravity taught me:

- How computer vision works — cameras, pixels, landmark detection
- What MediaPipe actually does under the hood
- How physics simulations work (Verlet integration)
- How eye tracking works using iris position
- How GitHub works — version control, commits, pushing
- How to debug real errors and understand what they mean
- How to think like a developer — breaking big ideas into small steps

**AI didn't replace the thinking. It replaced the typing.**

---

## 👤 Author

**Rupesh Thakur** — vision, direction, testing, debugging
**Claude by Anthropic** — code generation, problem solving

> *"I didn't write the code. I wrote the idea."*

---

## 📄 License

MIT License — use it, build on it, make it yours.
