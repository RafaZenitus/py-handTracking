# Hand Tracking System – Full Project

This project is the **Final Undergraduate Project (TCC)**, consisting of a **Python backend** for real-time hand tracking and a **Godot frontend** for interactive visualization and user feedback. Both parts are required for the system to function correctly.

---

## Backend (Python)

### About

This repository contains the **backend module** of the project. It performs **real-time hand tracking** using computer vision techniques with the **MediaPipe** library and sends the processed hand landmark data to the Godot frontend via **TCP socket communication**.

The system captures hand landmarks from the webcam, applies **smoothing filters** to reduce noise, and transmits structured **JSON data** to the frontend for interaction and visualization.

> Both repositories are required for the system to function correctly.

### Frontend (Godot) Repository:
https://github.com/RafaZenitus/gd-handTrackig

---

### Functionality

- Captures **webcam input** for both hands
- Detects and tracks **hand landmarks**
- Applies **moving average filters** for smooth movement
- Stores the **last known hand positions** to handle temporary tracking loss
- Sends **JSON-formatted data** over TCP to the Godot frontend
- Optionally generates **graphs of hand paths** after the session
- Handles **session metadata** (patient name, date, time)

### Requirements

- Python 3.8 or higher

- Libraries: mediapipe, opencv-python, numpy, matplotlib

- Connected webcam

- Godot frontend must be running first
