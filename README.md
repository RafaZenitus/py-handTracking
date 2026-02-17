About This Project

This repository contains the backend module of my Final Undergraduate Project. It is responsible for real-time hand tracking using computer vision techniques and sending the processed data through TCP socket communication.

The system captures hand landmarks from the webcam, applies smoothing filters to reduce noise, and transmits structured JSON data to an external application.

For the system to function correctly, it is necessary to use the frontend application developed in Godot, which acts as the TCP server and is responsible for receiving the hand tracking data and handling the interactive/visual components of the project.
