import base64
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify, render_template, session
from PIL import Image
from io import BytesIO
import os
import random
import cv2

model = tf.keras.models.load_model("model/best_model.h5")
label_classes = np.load("model/label_classes.npy")

mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

app = Flask(__name__)
app.secret_key = 'gesturelink-secret-key'

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/learn')
def learn():
    completed = session.get('completed', [])
    return render_template('learn.html', all_chars=getChars(), completed=completed)

@app.route('/learn/<label>')
def learn_letter(label):
    label = label.upper()
    if label not in getChars():
        return "Label tidak valid", 404
    return render_template('learn_single.html', label=label)

@app.route('/mark_complete/<label>', methods=['POST'])
def mark_complete(label):
    label = label.upper()
    valid_labels = getChars()
    if label not in valid_labels:
        return "Label tidak valid", 400

    completed = session.get('completed', [])
    if label not in completed:
        completed.append(label)
        session['completed'] = completed
    return "OK", 200

@app.route('/get_completed')
def get_completed():
    return jsonify(session.get('completed', []))

@app.route('/quiz')
def quiz():
    quiz_labels = random.sample(getChars(), 10)
    session['quiz_labels'] = quiz_labels
    session['quiz_index'] = 0
    session['quiz_score'] = 0
    return render_template('quiz.html')

@app.route('/quiz/next', methods=['POST'])
def quiz_next():
    prediction = request.json.get('prediction')
    index = session.get('quiz_index', 0)
    labels = session.get('quiz_labels', [])
    score = session.get('quiz_score', 0)

    if not labels or index >= len(labels):
        return jsonify({'done': True, 'score': score})

    if prediction is None:
        return jsonify({'done': False, 'label': labels[index], 'index': index + 1})

    prediction = prediction.upper()

    if prediction:
        if prediction == labels[index]:
            score += 1
        session['quiz_score'] = score

    session['quiz_index'] = index + 1
    index += 1

    if index >= len(labels):
        return jsonify({'done': True, 'score': score})

    return jsonify({'done': False, 'label': labels[index], 'index': index + 1})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    session['current_text'] = ""
    return jsonify({'status': 'cleared'})

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': session.get('current_text', '')})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("[INFO] Received image for prediction")
        data = request.json
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img = np.array(img)

        print("[INFO] Image shape:", img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        landmarks = result.multi_hand_landmarks
        handedness = result.multi_handedness

        if not landmarks:
            print("[INFO] Tidak ada tangan terdeteksi")
            return jsonify({'prediction': '-'})

        data = []
        handed_landmarks = sorted(zip(handedness, landmarks), key=lambda x: x[0].classification[0].label)

        for _, lm in handed_landmarks:
            for point in lm.landmark:
                data.extend([point.x, point.y, point.z])

        while len(data) < 126:
            data.extend([0.0] * 3)

        input_data = np.array(data).reshape(1, 126)
        prediction = model.predict(input_data, verbose=0)
        pred_index = np.argmax(prediction)
        pred_label = label_classes[pred_index]

        print("[INFO] Prediksi:", pred_label)

        return jsonify({'prediction': pred_label})
    except Exception as e:
        print("[ERROR] Exception in predict():", str(e))
        return jsonify({'prediction': '-', 'error': str(e)})

def getChars():
    labels_path = 'dataset/labels.txt'
    all_chars = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 1:
                    label = parts[0].strip().upper()
                    all_chars.append(label)
    return all_chars

if __name__ == '__main__':
    app.run(debug=True)

# Tambahkan ini di paling akhir file index.py
def handler(environ, start_response):
    return app(environ, start_response)
