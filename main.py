from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import time
import threading

try:
    if hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    elif hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

app = Flask(__name__)

camera_lock = threading.Lock()
matrix_lock = threading.Lock()
camera_id = 0
capture = None

offset_x = 0
offset_y = 0
scale_x = 50
scale_y = 50
rotation_x = 0
rotation_y = 0
rotation_z = 0
step_mode = False
current_color = 1
step_x, step_y = 0, 0

matrix = [
    [0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,1,0],
    [0,2,0,0,0,0,2,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,3,0,0,0,0,3,0],
    [0,0,4,5,5,4,0,0],
    [0,0,0,0,0,0,0,0]
]
colors_rgb = {
    1: (231, 25, 74),
    2: (60, 179, 75),
    3: (68, 99, 216),
    4: (254, 224, 28),
    5: (65, 212, 245),
    6: (145, 31, 181)
}

colors = {k: (v[2], v[1], v[0]) for k, v in colors_rgb.items()}

def generate_frames():
    global capture, camera_id
    default_h, default_w = 480, 640

    while True:
        success, frame = False, None
        
        with camera_lock:
            if capture is None or not capture.isOpened():
                if capture:
                    capture.release()
                print(f"Attempting to open camera {camera_id}...")
                capture = cv2.VideoCapture(camera_id)
                if capture.isOpened():
                    time.sleep(0.1) 
            
            if capture and capture.isOpened():
                success, frame = capture.read()

        if not success or frame is None:
            h_frame, w_frame = default_h, default_w
            frame = np.zeros((h_frame, w_frame, 3), np.uint8)
            cv2.putText(frame, f"Camera {camera_id} not available", (50,240),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(frame, "Select a different camera or restart", (50,280),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            time.sleep(0.5)
        else:
            h_frame, w_frame = frame.shape[:2]

        with matrix_lock:
            current_matrix = [row[:] for row in matrix]

        rows, cols = len(current_matrix), len(current_matrix[0])
        ow_f = cols * scale_x
        oh_f = rows * scale_y
        
        if ow_f <= 0 or oh_f <= 0:
            ret, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            continue

        ow, oh = int(np.ceil(ow_f)), int(np.ceil(oh_f))
        overlay = np.zeros((oh, ow, 3), np.uint8)
        
        for i in range(rows):
            for j in range(cols):
                v = current_matrix[i][j]
                if v and (not step_mode or (i == step_y and j == step_x)):
                    x1 = int(j * scale_x)
                    y1 = int(i * scale_y)
                    x2 = int((j + 1) * scale_x) - 1
                    y2 = int((i + 1) * scale_y) - 1
                    bgr_color = (colors_rgb[v][2], colors_rgb[v][1], colors_rgb[v][0])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, -1)

        final_offset_x = float(offset_x)
        final_offset_y = float(offset_y)

        if rotation_z != 0:
            center_f = (ow_f / 2, oh_f / 2)
            M = cv2.getRotationMatrix2D(center_f, rotation_z, 1.0)
            rads = np.radians(rotation_z)
            cos_val = abs(np.cos(rads))
            sin_val = abs(np.sin(rads))
            new_ow_f = ow_f * cos_val + oh_f * sin_val
            new_oh_f = ow_f * sin_val + oh_f * cos_val
            new_ow, new_oh = int(np.ceil(new_ow_f)), int(np.ceil(new_oh_f))
            M[0, 2] += (new_ow / 2) - center_f[0]
            M[1, 2] += (new_oh / 2) - center_f[1]
            rotated_overlay = cv2.warpAffine(overlay, M, (new_ow, new_oh))
            final_offset_x -= (new_ow_f - ow_f) / 2
            final_offset_y -= (new_oh_f - oh_f) / 2
            overlay = rotated_overlay
            ow, oh = new_ow, new_oh

        x1, y1 = int(round(final_offset_x)), int(round(final_offset_y))
        x2, y2 = x1 + ow, y1 + oh
        
        fx1, fy1 = max(x1, 0), max(y1, 0)
        fx2, fy2 = min(x2, w_frame), min(y2, h_frame)

        ox1, oy1 = fx1 - x1, fy1 - y1
        ox2, oy2 = ox1 + (fx2 - fx1), oy1 + (fy2 - fy1)

        if fx1 < fx2 and fy1 < fy2 and ox1 < ox2 and oy1 < oy2:
            roi = frame[fy1:fy2, fx1:fx2]
            ov = overlay[oy1:oy2, ox1:ox2]
            gray = cv2.cvtColor(ov, cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            inv = cv2.bitwise_not(m)
            bg = cv2.bitwise_and(roi, roi, mask=inv)
            fg = cv2.bitwise_and(ov, ov, mask=m)
            frame[fy1:fy2, fx1:fx2] = cv2.add(bg, fg)

        ret, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
          
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global offset_x, offset_y, scale_x, scale_y
    global rotation_x, rotation_y, rotation_z
    global step_mode, current_color
    global camera_id, capture
    global step_x, step_y

    data = request.get_json() or {}
    offset_x    = data.get('offset_x', offset_x)
    offset_y    = data.get('offset_y', offset_y)
    scale_x     = data.get('scale_x', scale_x)
    scale_y     = data.get('scale_y', scale_y)
    rotation_x  = data.get('rotation_x', rotation_x)
    rotation_y  = data.get('rotation_y', rotation_y)
    rotation_z  = data.get('rotation_z', rotation_z)
    step_mode   = data.get('step_mode', step_mode)
    current_color = data.get('current_color', current_color)
    step_x      = data.get('step_x', step_x)
    step_y      = data.get('step_y', step_y)

    if 'camera_id' in data:
        try:
            new_id = int(data['camera_id'])
            if new_id != camera_id:
                with camera_lock:
                    camera_id = new_id
                    if capture:
                        capture.release()
                    capture = None
        except Exception as e:
            print(f"Error handling camera switch: {e}")
            pass

    return ('',200)

@app.route('/get_config')
def get_config():
    with matrix_lock:
        return jsonify({
            'matrix': matrix,
            'colors': colors_rgb,
        })

@app.route('/update_matrix', methods=['POST'])
def update_matrix():
    global matrix
    data = request.get_json()
    if not data or 'matrix' not in data:
        return jsonify({'error': 'Missing matrix data'}), 400
    
    new_matrix = data['matrix']
    if not isinstance(new_matrix, list) or len(new_matrix) == 0 or not isinstance(new_matrix[0], list):
        return jsonify({'error': 'Invalid matrix format'}), 400
    
    rows, cols = len(new_matrix), len(new_matrix[0])
    if not all(len(row) == cols for row in new_matrix):
         return jsonify({'error': 'Matrix must be rectangular'}), 400

    with matrix_lock:
        matrix = new_matrix
    
    return jsonify({'success': True})

@app.route('/get_cameras')
def get_cameras():
    cams = []
    for i in range(5):
        try:
            tmp = cv2.VideoCapture(i)
            if tmp.isOpened():
                cams.append(i)
            tmp.release()
        except: pass
    
    if camera_id not in cams:
        cams.append(camera_id)
        cams.sort()
        
    return jsonify({'cameras':cams,'current':camera_id})

@app.route('/add_camera', methods=['POST'])
def add_camera():
    data = request.get_json() or {}
    try:
        cid = int(data['camera_id'])
        return jsonify({'success':True,'camera_id':cid})
    except:
        return jsonify({'success':False,'error':'Invalid camera ID'}),400

if __name__ == '__main__':
    app.run(debug=True, port=6001, use_reloader=False)