from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import socket
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Configuration
class Config:
    OPENGOLFSIM_HOST = "127.0.0.1"
    OPENGOLFSIM_PORT = 3111
    ENABLE_OPENGOLFSIM = False  # Toggle for OpenGolfSim integration
    BALL_DIAMETER_METERS = 0.04267  # Standard golf ball
    MIN_DETECTIONS = 5  # Increased from 3
    MOVEMENT_THRESHOLD = 10.0  # Increased from 3.0 to ignore static objects
    YOLO_CONFIDENCE = 0.4  # Increased from 0.3
    MAX_STATIC_DETECTIONS = 3  # Skip if ball hasn't moved in N frames

config = Config()

# Pydantic models for API
class ShotData(BaseModel):
    ballSpeed: float  # m/s or mph depending on unit
    verticalLaunchAngle: float
    horizontalLaunchAngle: float = 0.0
    spinSpeed: int = 0
    spinAxis: float = 0.0

class AnalysisResult(BaseModel):
    speed: float  # mph
    angle: float  # degrees
    carry_distance: float  # yards
    apex_height: float  # feet
    hang_time: float  # seconds
    trajectory: List[tuple]
    simulated_trajectory: List[tuple]
    processed_video_url: str
    detections_count: int
    calibration_info: Dict

# OpenGolfSim Integration
class OpenGolfSimClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
    
    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to OpenGolfSim at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OpenGolfSim: {e}")
            return False
    
    def send_device_status(self, status: str) -> bool:
        """Send device status (ready/busy)"""
        try:
            payload = {"type": "device", "status": status}
            self.socket.sendall(json.dumps(payload).encode() + b'\n')
            logger.info(f"Sent device status: {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to send device status: {e}")
            return False
    
    def send_shot(self, shot_data: ShotData, unit: str = "imperial") -> bool:
        """Send shot data to OpenGolfSim"""
        try:
            payload = {
                "type": "shot",
                "unit": unit,
                "shot": {
                    "ballSpeed": shot_data.ballSpeed,
                    "verticalLaunchAngle": shot_data.verticalLaunchAngle,
                    "horizontalLaunchAngle": shot_data.horizontalLaunchAngle,
                    "spinSpeed": shot_data.spinSpeed,
                    "spinAxis": shot_data.spinAxis
                }
            }
            self.socket.sendall(json.dumps(payload).encode() + b'\n')
            logger.info(f"Sent shot data: {payload}")
            return True
        except Exception as e:
            logger.error(f"Failed to send shot data: {e}")
            return False
    
    def disconnect(self):
        if self.socket:
            self.socket.close()
            logger.info("Disconnected from OpenGolfSim")

opengolfsim_client = OpenGolfSimClient(config.OPENGOLFSIM_HOST, config.OPENGOLFSIM_PORT)

# ---- Helper Functions ----
def simulate_flight(v0, angle_deg):
    """Simulate ball flight trajectory"""
    g = 9.81
    t = np.linspace(0, 5, 200)
    x = v0 * np.cos(np.radians(angle_deg)) * t
    y = v0 * np.sin(np.radians(angle_deg)) * t - 0.5 * g * t**2
    y = np.maximum(y, 0)
    return list(zip(x.tolist(), y.tolist()))

def detect_ball_in_frame(frame, frame_idx, positions, last_position=None):
    """Detect ball in a single frame using YOLO + OpenCV fallback"""
    detected = False
    best_detection = None
    best_confidence = 0
    
    # Try YOLO detection first
    results = model.predict(frame, conf=config.YOLO_CONFIDENCE, verbose=False)
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if model.names[cls] == "sports ball" and conf > best_confidence:
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                diameter = max(x2 - x1, y2 - y1)
                best_detection = {
                    'x': float(cx), 
                    'y': float(cy), 
                    'diameter': diameter, 
                    'confidence': conf,
                    'method': 'YOLO'
                }
                best_confidence = conf
    
    if best_detection:
        return best_detection, True
    
    # Fallback to OpenCV detection (more restrictive)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Tighter HSV range for white golf balls
    mask = cv2.inRange(hsv, (0, 0, 220), (180, 25, 255))
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by circularity, size, and movement
    valid_detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50 or area > 3000:  # More restrictive size range
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:  # More strict circularity
            (x, y), radius = cv2.minEnclosingCircle(c)
            
            # If we have a last position, prioritize detections that moved
            if last_position:
                dist = np.linalg.norm(np.array([x, y]) - np.array(last_position))
                # Skip if too close to last position (likely static object)
                if dist < 5:
                    continue
                valid_detections.append((c, x, y, radius, dist))
            else:
                valid_detections.append((c, x, y, radius, 0))
    
    if valid_detections:
        # Prefer detections that moved more
        if last_position and len(valid_detections) > 1:
            valid_detections.sort(key=lambda d: d[4], reverse=True)
        
        c, x, y, radius, _ = valid_detections[0]
        
        if radius > 5 and radius < 80:
            return {
                'x': float(x),
                'y': float(y),
                'diameter': float(radius * 2),
                'confidence': 0.5,
                'method': 'OpenCV'
            }, True
    
    return None, False

def calculate_physics(positions, diameters, fps):
    """Calculate ball physics from detected positions"""
    if len(positions) < config.MIN_DETECTIONS:
        logger.warning(f"Not enough positions: {len(positions)} (need {config.MIN_DETECTIONS})")
        return None
    
    # Use up to 10 positions for better accuracy
    num_positions = min(len(positions), 10)
    positions_arr = np.array(positions[:num_positions])
    t_arr = np.arange(num_positions) / fps
    
    # Fit polynomial to trajectory
    fit_x = np.polyfit(t_arr, positions_arr[:, 0], 1)
    fit_y = np.polyfit(t_arr, positions_arr[:, 1], 2)
    
    # Calculate initial velocities in pixels/second
    vx_pix = fit_x[0]
    vy_pix = fit_y[1]  # Initial velocity from quadratic fit
    
    # Calibration: convert pixels to meters
    ball_diameter_pixels = np.median(diameters) if diameters else 20
    pixel_to_meter = config.BALL_DIAMETER_METERS / ball_diameter_pixels
    
    logger.info(f"Calibration: {ball_diameter_pixels:.2f} pixels = {config.BALL_DIAMETER_METERS} meters")
    logger.info(f"Pixel to meter ratio: {pixel_to_meter:.6f}")
    
    # Convert to real-world velocities
    vx = vx_pix * pixel_to_meter
    vy = -vy_pix * pixel_to_meter  # Negative because Y increases downward
    
    speed_mps = np.sqrt(vx**2 + vy**2)
    angle_deg = np.degrees(np.arctan2(vy, vx))
    
    # Physics calculations
    g = 9.81
    if vy > 0:
        hang_time = (2 * vy) / g
        carry_distance = vx * hang_time
        apex_height = (vy**2) / (2 * g)
    else:
        hang_time = carry_distance = apex_height = 0
    
    logger.info(f"Physics: speed={speed_mps:.2f} m/s, angle={angle_deg:.1f}°, carry={carry_distance:.2f}m")
    
    return {
        'speed_mps': speed_mps,
        'speed_mph': speed_mps * 2.23694,
        'angle_deg': angle_deg,
        'carry_distance_m': carry_distance,
        'carry_distance_yards': carry_distance * 1.09361,
        'apex_height_m': apex_height,
        'apex_height_ft': apex_height * 3.28084,
        'hang_time': hang_time,
        'calibration': {
            'ball_diameter_pixels': float(ball_diameter_pixels),
            'pixel_to_meter': float(pixel_to_meter),
            'fps': fps
        }
    }

def process_video_file(video_path: str, output_path: str):
    """Core video processing logic"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    positions = []
    diameters = []
    frame_idx = 0
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Video specs: {width}x{height} @ {fps} fps")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        last_pos = positions[-1] if positions else None
        detection, detected = detect_ball_in_frame(frame, frame_idx, positions, last_pos)
        
        if detection:
            x, y = detection['x'], detection['y']
            diameter = detection['diameter']
            
            # Check if ball moved significantly
            move_dist = np.linalg.norm(np.array([x, y]) - np.array(positions[-1])) if positions else None
            
            if not positions or move_dist > config.MOVEMENT_THRESHOLD:
                positions.append((x, y))
                diameters.append(diameter)
                move_str = f"{move_dist:.1f}" if move_dist else "0.0"
                logger.info(f"Frame {frame_idx}: {detection['method']} detected ball at ({x:.2f}, {y:.2f}), conf={detection['confidence']:.2f}, moved={move_str}px")
            
            # Draw detection
            if detection['method'] == 'YOLO':
                cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"{detection['confidence']:.2f}", (int(x)+10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                radius = int(diameter / 2)
                cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 0), 2)
        
        # Draw trajectory
        for i in range(1, len(positions)):
            cv2.line(frame,
                    (int(positions[i - 1][0]), int(positions[i - 1][1])),
                    (int(positions[i][0]), int(positions[i][1])),
                    (0, 255, 255), 3)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx} | Detections: {len(positions)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    
    cap.release()
    out.release()
    
    logger.info(f"Processing complete. Frames: {frame_idx}, Detections: {len(positions)}")
    
    # Calculate physics
    physics = calculate_physics(positions, diameters, fps)
    
    return {
        'positions': positions,
        'diameters': diameters,
        'physics': physics,
        'frame_count': frame_idx,
        'fps': fps
    }

# ---- API Endpoints ----

@app.get("/")
def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Golf Simulator API",
        "version": "2.0",
        "opengolfsim_enabled": config.ENABLE_OPENGOLFSIM
    }

@app.get("/config")
def get_config():
    """Get current configuration"""
    return {
        "opengolfsim_enabled": config.ENABLE_OPENGOLFSIM,
        "opengolfsim_host": config.OPENGOLFSIM_HOST,
        "opengolfsim_port": config.OPENGOLFSIM_PORT,
        "yolo_confidence": config.YOLO_CONFIDENCE,
        "movement_threshold": config.MOVEMENT_THRESHOLD,
        "min_detections": config.MIN_DETECTIONS
    }

@app.post("/config/opengolfsim")
def toggle_opengolfsim(enable: bool):
    """Enable or disable OpenGolfSim integration"""
    config.ENABLE_OPENGOLFSIM = enable
    if enable:
        success = opengolfsim_client.connect()
        if success:
            opengolfsim_client.send_device_status("ready")
            return {"status": "enabled", "connected": True}
        else:
            config.ENABLE_OPENGOLFSIM = False
            return {"status": "failed", "error": "Could not connect to OpenGolfSim"}
    else:
        opengolfsim_client.disconnect()
        return {"status": "disabled"}

@app.get("/analyze-existing")
def analyze_existing(filename: str):
    """Analyze an existing video file"""
    try:
        video_path = os.path.join(os.path.dirname(__file__), "..", "process-video", filename)
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="File not found")

        out_path = video_path.replace(".mp4", "_processed.mp4").replace(".mov", "_processed.mp4")
        
        # Process video using refactored function
        result = process_video_file(video_path, out_path)
        
        if not result['physics']:
            return JSONResponse({
                "error": "Not enough ball detections",
                "detections_count": len(result['positions']),
                "min_required": config.MIN_DETECTIONS
            }, status_code=400)
        
        physics = result['physics']
        simulated_trajectory = simulate_flight(physics['speed_mps'], physics['angle_deg'])
        
        # Send to OpenGolfSim if enabled
        if config.ENABLE_OPENGOLFSIM:
            shot_data = ShotData(
                ballSpeed=physics['speed_mph'],
                verticalLaunchAngle=physics['angle_deg'],
                horizontalLaunchAngle=0.0,
                spinSpeed=0,
                spinAxis=0.0
            )
            opengolfsim_client.send_shot(shot_data, unit="imperial")
        
        return JSONResponse({
            "speed": round(physics['speed_mph'], 1),
            "angle": round(physics['angle_deg'], 1),
            "carry_distance": round(physics['carry_distance_yards'], 1),
            "apex_height": round(physics['apex_height_ft'], 1),
            "hang_time": round(physics['hang_time'], 1),
            "trajectory": [(float(x), float(y)) for x, y in result['positions']],
            "simulated_trajectory": simulated_trajectory,
            "processed_video_url": f"/download-video?path={out_path}",
            "detections_count": len(result['positions']),
            "calibration_info": physics['calibration'],
            "opengolfsim_sent": config.ENABLE_OPENGOLFSIM
        })
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---- Upload & Process New Video ----
@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """Upload and process a new video file"""
    try:
        process_video_dir = os.path.join(os.path.dirname(__file__), "..", "process-video")
        os.makedirs(process_video_dir, exist_ok=True)
        video_path = os.path.join(process_video_dir, file.filename)

        # Save upload
        with open(video_path, "wb") as f:
            f.write(await file.read())

        out_path = video_path.replace(".mp4", "_processed.mp4").replace(".mov", "_processed.mp4")
        
        # Process video using refactored function
        result = process_video_file(video_path, out_path)
        
        if not result['physics']:
            return JSONResponse({
                "error": "Not enough ball detections",
                "detections_count": len(result['positions']),
                "min_required": config.MIN_DETECTIONS
            }, status_code=400)
        
        physics = result['physics']
        simulated_trajectory = simulate_flight(physics['speed_mps'], physics['angle_deg'])
        
        # Send to OpenGolfSim if enabled
        if config.ENABLE_OPENGOLFSIM:
            shot_data = ShotData(
                ballSpeed=physics['speed_mph'],
                verticalLaunchAngle=physics['angle_deg'],
                horizontalLaunchAngle=0.0,
                spinSpeed=0,
                spinAxis=0.0
            )
            opengolfsim_client.send_shot(shot_data, unit="imperial")
        
        return JSONResponse({
            "speed": round(physics['speed_mph'], 1),
            "angle": round(physics['angle_deg'], 1),
            "carry_distance": round(physics['carry_distance_yards'], 1),
            "apex_height": round(physics['apex_height_ft'], 1),
            "hang_time": round(physics['hang_time'], 1),
            "trajectory": [(float(x), float(y)) for x, y in result['positions']],
            "simulated_trajectory": simulated_trajectory,
            "processed_video_url": f"/download-video?path={out_path}",
            "detections_count": len(result['positions']),
            "calibration_info": physics['calibration'],
            "opengolfsim_sent": config.ENABLE_OPENGOLFSIM
        })
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---- Download processed video ----
@app.get("/download-video")
def download_video(path: str):
    """Download a processed video file"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))

# ---- Test endpoint for OpenGolfSim ----
@app.post("/test-opengolfsim")
def test_opengolfsim_connection():
    """Test connection to OpenGolfSim"""
    try:
        success = opengolfsim_client.connect()
        if success:
            opengolfsim_client.send_device_status("ready")
            # Send test shot
            test_shot = ShotData(
                ballSpeed=100.0,
                verticalLaunchAngle=15.0,
                horizontalLaunchAngle=0.0,
                spinSpeed=3000,
                spinAxis=0.0
            )
            shot_sent = opengolfsim_client.send_shot(test_shot, unit="imperial")
            opengolfsim_client.disconnect()
            return {
                "status": "success",
                "connected": True,
                "shot_sent": shot_sent
            }
        else:
            return {
                "status": "failed",
                "connected": False,
                "error": "Could not connect to OpenGolfSim"
            }
    except Exception as e:
        logger.error(f"Error testing OpenGolfSim: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Golf Simulator API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
