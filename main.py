import cv2
import numpy as np
import os

# Path to dataset folder
video_folder = "dataset"

# Loop through all videos
for video_file in os.listdir(video_folder):

    print(f"Processing: {video_file}")

    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        print("Skipping empty video")
        continue

    prev = cv2.resize(prev, (640, 480))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # 🔹 Define crosswalk zone (adjust if needed)
    
    zone = [(150,200),(500,200),(500,350),(150,350)]
    

    # 🔹 Video writer (Step 5)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"output_{video_file}.avi", fourcc, 20.0, (640,480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 🔹 Preprocessing (Denoise)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # 🔹 Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, blur, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 🔹 Get velocity magnitude
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        # 🔹 Create zone mask
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [np.array(zone)], 255)

        # 🔹 Conflict detection (tune threshold if needed)
        conflict = (mag > 2.5) & (mask == 255)

        # 🔹 Highlight conflict in RED
        frame[conflict] = [0, 0, 255]

        # 🔹 Draw crosswalk zone in GREEN
        cv2.polylines(frame, [np.array(zone)], True, (0,255,0), 2)

        # 🔹 Display video name
        cv2.putText(frame, video_file, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 🔹 Save frame (Step 5)
        out.write(frame)

        # 🔹 Show output
        cv2.imshow("Pedestrian Conflict Detection", frame)

        # Update previous frame
        prev_gray = blur.copy()

        # Press ESC to move to next video
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()