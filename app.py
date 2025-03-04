from flask import Flask, request, jsonify
import cv2
import numpy as np
from detect_face import detect_face
from predict_age import predict_age

app = Flask(__name__)

@app.route('/predict_age', methods=['POST'])
def predict():
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    face, _ = detect_face(img)
    if face is None:
        return jsonify({"error": "No face detected"})
    
    age = predict_age(face)
    return jsonify({"age": age})

if __name__ == '__main__':
    app.run(debug=True)