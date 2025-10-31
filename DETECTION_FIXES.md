# Detection Fixes - Golf Simulator

## Problem Analysis

Your video analysis showed **107 detections** but most were at the same static positions:
- Position 1: `(690, 1243)` - detected ~60 times
- Position 2: `(705, 1242)` - detected ~40 times  
- Actual ball movement: Only ~7 frames with real ball positions

This caused:
- ❌ **Carry distance**: 0.24 yards (should be 150-250 yards)
- ❌ **Ball speed**: 2.09 m/s (should be 40-70 m/s)
- ❌ **Launch angle**: 73.9° (unrealistic)

## Root Causes

### 1. **Detecting Static Background Objects**
The OpenCV fallback was detecting white objects in the background:
- White tees/markers
- Bright spots on the ground
- Static white objects in the scene

### 2. **Too Permissive HSV Range**
```python
# OLD (too broad)
mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
```
This detected ANY white object with brightness > 200.

### 3. **Low Movement Threshold**
```python
# OLD
MOVEMENT_THRESHOLD = 3.0  # pixels
```
Only 3 pixels meant static objects kept being re-detected.

## Fixes Applied

### 1. **Stricter HSV Range**
```python
# NEW (more selective)
mask = cv2.inRange(hsv, (0, 0, 220), (180, 25, 255))
```
- Minimum brightness: 220 (was 200)
- Maximum saturation: 25 (was 30)
- Targets very white, low-saturation objects (golf balls)

### 2. **Increased Movement Threshold**
```python
MOVEMENT_THRESHOLD = 10.0  # pixels (was 3.0)
```
- Ignores detections that moved < 10 pixels
- Filters out static objects and jitter

### 3. **Stricter Size Constraints**
```python
# OLD
if area < 10 or area > 5000:

# NEW  
if area < 50 or area > 3000:
```
- Minimum area: 50 pixels (was 10)
- Maximum area: 3000 pixels (was 5000)
- Reduces false positives from small/large objects

### 4. **Higher Circularity Threshold**
```python
# OLD
if circularity > 0.6:

# NEW
if circularity > 0.7:
```
- More strict shape requirement
- Golf balls are very circular

### 5. **Movement-Based Filtering**
```python
# NEW: Prioritize moving objects
if last_position:
    dist = np.linalg.norm(np.array([x, y]) - np.array(last_position))
    if dist < 5:  # Skip if too close to last position
        continue
    valid_detections.append((c, x, y, radius, dist))

# Sort by movement distance (prefer objects that moved more)
valid_detections.sort(key=lambda d: d[4], reverse=True)
```

### 6. **Larger Morphological Kernel**
```python
# OLD
kernel = np.ones((3, 3), np.uint8)

# NEW
kernel = np.ones((5, 5), np.uint8)
```
- Better noise reduction
- Removes small artifacts

### 7. **Higher YOLO Confidence**
```python
YOLO_CONFIDENCE = 0.4  # (was 0.3)
```
- Reduces false positives from YOLO

### 8. **Minimum Detections Increased**
```python
MIN_DETECTIONS = 5  # (was 3)
```
- Requires more data points for physics calculation
- More reliable results

## Expected Improvements

After these fixes, you should see:

✅ **Fewer detections** (~10-20 instead of 107)  
✅ **Only moving ball positions** (not static objects)  
✅ **Realistic carry distance** (150-250 yards)  
✅ **Realistic ball speed** (40-70 m/s or 90-155 mph)  
✅ **Realistic launch angle** (10-25°)  

## Testing

Restart your backend and try again:

```bash
cd backend
python main.py
```

Then upload your video through the frontend at `http://localhost:5173`

## Debugging Tips

Watch the console logs for:
```
INFO:__main__:Frame X: YOLO/OpenCV detected ball at (x, y), conf=0.X, moved=XXpx
```

Good signs:
- ✅ Ball positions change significantly between frames
- ✅ Movement > 10 pixels
- ✅ Positions form a trajectory (not random jumps)

Bad signs:
- ❌ Same position detected multiple times
- ❌ Movement < 5 pixels
- ❌ Random positions all over the frame

## Further Tuning

If still having issues, adjust in `main.py`:

```python
class Config:
    MOVEMENT_THRESHOLD = 15.0  # Increase if still detecting static objects
    YOLO_CONFIDENCE = 0.5      # Increase if too many false positives
    MIN_DETECTIONS = 7         # Increase for more reliable physics
```

## Video Quality Tips

For best results:
- 📹 High frame rate (60+ fps)
- 🎥 Good lighting
- 🏌️ Clear view of ball flight
- 📏 Consistent camera distance
- 🎯 Minimal background clutter
