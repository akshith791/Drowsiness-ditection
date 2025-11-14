import cv2
import time
import argparse
from threading import Thread
import playsound
import os
import sys
import subprocess


def sound_alarm(path):
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        print(f"[ERROR] Alarm file not found: {resolved}")
        return
    print(f"[INFO] Playing alarm: {resolved}")
    try:
        if sys.platform == "darwin":
            # Use macOS built-in audio player for reliable WAV playback
            subprocess.run(["afplay", resolved], check=False)
        else:
            playsound.playsound(resolved)
    except Exception as e:
        print(f"[ERROR] Alarm playback failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alarm", type=str, default="alarm.wav", help="path to alarm .wav file")
    parser.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
    parser.add_argument("-f", "--face_cascade", type=str, default=None, help="optional path to face cascade xml")
    parser.add_argument("-e", "--eye_cascade", type=str, default=None, help="optional path to eye cascade xml")
    parser.add_argument("--eye_scale", type=float, default=1.1, help="Haar eye scaleFactor")
    parser.add_argument("--eye_neighbors", type=int, default=7, help="Haar eye minNeighbors")
    parser.add_argument("--eye_min", type=int, default=22, help="Minimum eye box size in pixels")
    parser.add_argument("-t", "--threshold_frames", type=int, default=15, help="frames without eyes before alarm")
    args = parser.parse_args()

    # Prefer the more robust eyeglasses eye cascade
    face_xml = args.face_cascade or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_xml = args.eye_cascade or (cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    face_cascade = cv2.CascadeClassifier(face_xml)
    eye_cascade = cv2.CascadeClassifier(eye_xml)

    if face_cascade.empty() or eye_cascade.empty():
        print("[ERROR] Could not load Haar cascades.")
        return

    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(args.webcam)
    time.sleep(1.0)

    counter = 0
    alarm_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        eyes_detected = False

        for (x, y, w, h) in faces:
            # Constrain eye search to upper half of the face ROI to avoid mouth/cheek false positives
            roi_gray = gray[y:y + h, x:x + w]
            upper_gray = roi_gray[0:h // 2, :]
            roi_color = frame[y:y + h, x:x + w]
            upper_color = roi_color[0:h // 2, :]

            # Light normalization improves robustness in variable lighting
            upper_eq = cv2.equalizeHist(upper_gray)

            eyes = eye_cascade.detectMultiScale(
                upper_eq,
                scaleFactor=args.eye_scale,
                minNeighbors=args.eye_neighbors,
                minSize=(args.eye_min, args.eye_min)
            )

            # draw face and eyes for feedback
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(upper_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            if len(eyes) > 0:
                eyes_detected = True

        if not eyes_detected:
            counter += 1
            cv2.putText(frame, f"No eyes: {counter}/{args.threshold_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if counter >= args.threshold_frames:
                if not alarm_on:
                    alarm_on = True
                    if args.alarm:
                        t = Thread(target=sound_alarm, args=(args.alarm,))
                        t.daemon = True
                        t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            counter = 0
            alarm_on = False

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()