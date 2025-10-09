from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 small model
model = YOLO("yolov8n.pt")  # Pretrained model with "sports ball" class

# ---- Physics simulator ----
def simulate_flight(v0, angle_deg):
    g = 9.81
    t = np.linspace(0, 5, 200)
    x = v0 * np.cos(np.radians(angle_deg)) * t
    y = v0 * np.sin(np.radians(angle_deg)) * t - 0.5 * g * t**2
    y = np.maximum(y, 0)
    return list(zip(x.tolist(), y.tolist()))

# ---- Analyze Existing File ----
@app.get("/analyze-existing")
def analyze_existing(filename: str):
    video_path = os.path.join(os.path.dirname(__file__), "..", "process-video", filename)
    if not os.path.exists(video_path):
        return JSONResponse({"error": "File not found."}, status_code=404)

    out_path = video_path.replace(".mp4", "_processed.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    positions = []
    diameters = []
    frame_idx = 0
    print(f"Analyzing {filename} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        detected = False
        results = model.predict(frame, conf=0.25, verbose=False)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                if model.names[cls] == "sports ball":
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    diameter = max(x2 - x1, y2 - y1)
                    diameters.append(diameter)
                    move_dist = np.linalg.norm(np.array([cx, cy]) - np.array(positions[-1])) if positions else None
                    print(f"YOLO: Frame {frame_idx}, Detected ball at ({cx:.2f}, {cy:.2f}), move_dist={move_dist}")
                    if not positions or move_dist > 0.5:
                        positions.append((float(cx), float(cy)))
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    detected = True
        if not detected:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0,0,180), (180,60,255))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, thresh)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"OpenCV fallback: Frame {frame_idx}, {len(contours)} contours found")
            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)
                move_dist = np.linalg.norm(np.array([x, y]) - np.array(positions[-1])) if positions else None
                print(f"OpenCV: Frame {frame_idx}, Ball at ({x:.2f}, {y:.2f}), radius={radius:.2f}, move_dist={move_dist}")
                if radius > 1:
                    if not positions or move_dist > 0.5:
                        positions.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
        if not detected:
            print(f"Frame {frame_idx}: No ball detected")
        for i in range(1, len(positions)):
            cv2.line(frame,
                     (int(positions[i - 1][0]), int(positions[i - 1][1])),
                     (int(positions[i][0]), int(positions[i][1])),
                     (255, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    print(f"Processing complete. Frames: {frame_idx}, Ball detections: {len(positions)}")
    if diameters:
        print(f"First detected ball diameter (pixels): {diameters[0]:.2f}")
    else:
        print("No ball diameter detected. Calibration needed.")

    # --- Physics estimation ---
    if len(positions) >= 5:
        positions_arr = np.array(positions[:5])
        fps = fps or 30
        t_arr = np.arange(len(positions_arr)) / fps
        fit_x = np.polyfit(t_arr, positions_arr[:, 0], 1)
        fit_y = np.polyfit(t_arr, positions_arr[:, 1], 2)
        vx_pix = fit_x[0]
        vy_pix = 2 * fit_y[0] * t_arr[0] + fit_y[1]
        BALL_DIAMETER_PIXELS = diameters[0] if diameters else 20
        BALL_DIAMETER_METERS = 0.04267
        PIXEL_TO_METER = BALL_DIAMETER_METERS / BALL_DIAMETER_PIXELS
        vx = vx_pix * fps * PIXEL_TO_METER
        vy = vy_pix * fps * PIXEL_TO_METER
        speed_mps = np.sqrt(vx**2 + vy**2)
        angle_deg = np.degrees(np.arctan2(vy, vx))
        g = 9.81
        hang_time = (2 * vy) / g if vy > 0 else 0
        carry_distance = vx * hang_time
        apex_height = (vy**2) / (2 * g)
    else:
        speed_mps = angle_deg = carry_distance = apex_height = hang_time = 0
        print("Not enough ball positions detected for physics calculation.")
    simulated_trajectory = simulate_flight(speed_mps, angle_deg)
    metrics = {
        "speed": round(speed_mps * 2.23694, 1),
        "angle": round(angle_deg, 1),
        "carry_distance": round(carry_distance * 1.09361, 1),
        "apex_height": round(apex_height * 3.28084, 1),
        "hang_time": round(hang_time, 1),
        "trajectory": [(float(x), float(y)) for x, y in positions],
        "simulated_trajectory": simulated_trajectory,
        "processed_video_url": f"/download-video?path={out_path}",
    }
    return JSONResponse(metrics)

# ---- Upload & Process New Video ----
@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    process_video_dir = os.path.join(os.path.dirname(__file__), "..", "process-video")
    os.makedirs(process_video_dir, exist_ok=True)
    video_path = os.path.join(process_video_dir, file.filename)

    # Save upload
    with open(video_path, "wb") as f:
        f.write(await file.read())

    out_path = video_path.replace(".mp4", "_processed.mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    positions = []
    diameters = []
    frame_idx = 0
    print(f"Processing uploaded video: {file.filename}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        detected = False
        results = model.predict(frame, conf=0.25, verbose=False)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                if model.names[cls] == "sports ball":
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    diameter = max(x2 - x1, y2 - y1)
                    diameters.append(diameter)
                    move_dist = np.linalg.norm(np.array([cx, cy]) - np.array(positions[-1])) if positions else None
                    print(f"YOLO: Frame {frame_idx}, Detected ball at ({cx:.2f}, {cy:.2f}), move_dist={move_dist}")
                    if not positions or move_dist > 0.5:
                        positions.append((float(cx), float(cy)))
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    detected = True
        if not detected:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0,0,180), (180,60,255))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, thresh)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"OpenCV fallback: Frame {frame_idx}, {len(contours)} contours found")
            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)
                move_dist = np.linalg.norm(np.array([x, y]) - np.array(positions[-1])) if positions else None
                print(f"OpenCV: Frame {frame_idx}, Ball at ({x:.2f}, {y:.2f}), radius={radius:.2f}, move_dist={move_dist}")
                if radius > 1:
                    if not positions or move_dist > 0.5:
                        positions.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
        if not detected:
            print(f"Frame {frame_idx}: No ball detected")
        for i in range(1, len(positions)):
            cv2.line(frame,
                     (int(positions[i - 1][0]), int(positions[i - 1][1])),
                     (int(positions[i][0]), int(positions[i][1])),
                     (255, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    print(f"Frames: {frame_idx}, Ball detections: {len(positions)}")
    if diameters:
        print(f"First detected ball diameter (pixels): {diameters[0]:.2f}")
    else:
        print("No ball diameter detected. Calibration needed.")

    # --- Physics metrics ---
    if len(positions) >= 5:
        positions_arr = np.array(positions[:5])
        fps = fps or 30
        t_arr = np.arange(len(positions_arr)) / fps
        fit_x = np.polyfit(t_arr, positions_arr[:, 0], 1)
        fit_y = np.polyfit(t_arr, positions_arr[:, 1], 2)
        vx_pix = fit_x[0]
        vy_pix = 2 * fit_y[0] * t_arr[0] + fit_y[1]
        BALL_DIAMETER_PIXELS = diameters[0] if diameters else 20
        BALL_DIAMETER_METERS = 0.04267
        PIXEL_TO_METER = BALL_DIAMETER_METERS / BALL_DIAMETER_PIXELS
        vx = vx_pix * fps * PIXEL_TO_METER
        vy = vy_pix * fps * PIXEL_TO_METER
        speed_mps = np.sqrt(vx**2 + vy**2)
        angle_deg = np.degrees(np.arctan2(vy, vx))
        g = 9.81
        hang_time = (2 * vy) / g if vy > 0 else 0
        carry_distance = vx * hang_time
        apex_height = (vy**2) / (2 * g)
    else:
        speed_mps = angle_deg = carry_distance = apex_height = hang_time = 0
        print("Not enough ball positions detected for physics calculation.")
    simulated_trajectory = simulate_flight(speed_mps, angle_deg)
    metrics = {
        "speed": round(speed_mps * 2.23694, 1),
        "angle": round(angle_deg, 1),
        "carry_distance": round(carry_distance * 1.09361, 1),
        "apex_height": round(apex_height * 3.28084, 1),
        "hang_time": round(hang_time, 1),
        "trajectory": [(float(x), float(y)) for x, y in positions],
        "simulated_trajectory": simulated_trajectory,
        "processed_video_url": f"/download-video?path={out_path}",
    }

    return JSONResponse(metrics)

# ---- Download processed video ----
@app.get("/download-video")
def download_video(path: str):
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))
