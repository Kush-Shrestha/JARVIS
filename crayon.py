import cv2
import numpy as np

# ==================== SETUP ====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera: {WIDTH}x{HEIGHT}")

# Colors (BGR)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)

colors_list = [BLACK, RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE]
color_names = ['Black', 'Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange']
color_index = 0
current_color = colors_list[color_index]

brush_size = 15
eraser_size = 60
prev_x, prev_y = None, None

# Default: track BLUE color (you can change this)
# HSV ranges for common colors
color_ranges = {
    'red': [(np.array([0, 100, 100], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
            np.array([160, 100, 100], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)],
    'blue': (np.array([100, 150, 0], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8)),
    'green': (np.array([40, 50, 50], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8)),
    'yellow': (np.array([20, 100, 100], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
}

# Current tracking color (blue by default)
track_color = 'blue'
current_hsv_lower = color_ranges['blue'][0]
current_hsv_upper = color_ranges['blue'][1]

# ==================== FUNCTIONS ====================

def detect_color_object(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        if area > 1000:
            M = cv2.moments(max_contour)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return x, y, area
    return None, None, 0

# ==================== MAIN ====================
print("=" * 50)
print("🎨 COLOR TRACKING DRAWING APP")
print("=" * 50)
print("USE A BLUE OBJECT TO DRAW")
print("(blue pen, blue glove, blue tape, etc)")
print("")
print("1 finger/object = DRAW")
print("No object = STOP")
print("C = Clear, Q = Quit")
print("=" * 50)

running = True

while running:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    original = frame.copy()
    
    # Detect colored object
    x, y, area = detect_color_object(frame, current_hsv_lower, current_hsv_upper)
    
    if x is None:
        prev_x, prev_y = None, None
    
    # Drawing
    if x:
        draw_color = current_color
        brush = brush_size
        
        if prev_x is not None:
            cv2.line(frame, (prev_x, prev_y), (x, y), draw_color, brush)
        
        prev_x, prev_y = x, y
        
        # Draw circle at position
        cv2.circle(frame, (x, y), 20, (0, 255, 0), 3)
    
    # UI - Show normal video, no red masks
    cv2.putText(frame, "🎨 DRAWING MODE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Color: {color_names[color_index]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)
    cv2.putText(frame, "Use BLUE object to draw", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    cv2.putText(frame, "C=Clear  Q=Quit", (20, HEIGHT-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.imshow("Drawing", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        cv2.imwrite("my_drawing.png", frame)
        print("💾 Saved!")
        break
    elif key == ord('c') or key == ord('C'):
        print("🗑️ Cleared!")
        prev_x, prev_y = None, None

cap.release()
cv2.destroyAllWindows()