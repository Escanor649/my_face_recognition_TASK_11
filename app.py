from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('scanner.html')

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image data'})

    header, encoded = data['image'].split(",", 1)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(100, 100))

    if len(faces) > 0:
        return jsonify({'status': 'success', 'message': 'Face Verified'})
    
    return jsonify({'status': 'denied', 'message': 'No Face Detected'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)