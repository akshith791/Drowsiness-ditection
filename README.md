# Drowsiness Detection with OpenCV

A Python-based project to detect and alert when a driver starts to doze off. This project uses **OpenCV**, **dlib**, and other Python libraries to implement a simple and effective blink detection system.

## Overview

This system monitors the driver's eye aspect ratio (EAR) in real-time using a webcam. If the EAR falls below a predefined threshold for a certain number of frames, it triggers an alarm to alert the driver.

The project is inspired by Adrian Rosebrock's tutorial on drowsiness detection from PyImageSearch.

---

## Key Features

- Real-time face and eye detection using dlib's pre-trained models.
- Calculates Eye Aspect Ratio (EAR) to determine if eyes are closed.
- Plays an alarm sound when drowsiness is detected.
- Lightweight and works on standard hardware.

---

## How It Works

1. **Face Localization**: Detects the face in the video frame.
2. **Facial Landmark Detection**: Identifies key facial structures (e.g., eyes).
3. **Eye Landmark Extraction**: Extracts coordinates for the eyes.
4. **Eye Aspect Ratio Calculation**:
   - EAR is computed using the formula:
     ![EAR FORMULA](assets/blink_detection_equation.png)
   - EAR decreases significantly when eyes are closed.
5. **Thresholding**:
   - If EAR is below a threshold (e.g., 0.26) for a certain number of frames, it indicates drowsiness.
6. **Alert Mechanism**:
   - Plays an alarm sound using the `playsound` library in a separate thread to avoid blocking the main script.
7. 68 coordinates are detected for the given face by the face detector as shown below:
![FACIAL LANDMARKS MARKUP](assets/facial_landmarks_68markup-768x619.jpg)
---

## Requirements

Below are the dependencies used in this project:

| Dependency | Version  |
|------------|----------|
| Python     | 3.7.3    |
| OpenCV     | 4.1.0    |
| NumPy      | 1.16.4   |
| imutils    | 0.5.2    |
| dlib       | 19.17.0  |

 ## **Commands to run the detection:**
 ```
 python detect_drowsiness.py \
	--shape-predictor shape_predictor_68_face_landmarks.dat \
	--alarm alarm.wav
```

## **Results:**
![Example output](assets/output.gif)

