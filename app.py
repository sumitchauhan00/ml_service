from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import pickle
from pathlib import Path
import mediapipe as mp

from landmark_features import extract_features
from sentence_builder import SentenceBuilder

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "landmark_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    artefact = pickle.load(f)

pipeline = artefact["pipeline"]
label_encoder = artefact["label_encoder"]
CONFIDENCE_THRESHOLD = float(artefact.get("confidence_threshold", 0.65))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# sentence_builder.py old signature supports only timeout, max_words
sb = SentenceBuilder(
    timeout=2.2,
    max_words=8
)

def get_hand_by_label(results, target: str):
    if not results.multi_hand_landmarks:
        return None
    for i, h in enumerate(results.multi_handedness):
        if h.classification[0].label == target:
            return results.multi_hand_landmarks[i]
    return None

def hand_to_flat(hand_lm):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float64).flatten()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    if "frame" not in request.files:
        return jsonify({"error": "No frame file"}), 400

    file = request.files["frame"]
    img_bytes = file.read()
    if not img_bytes or len(img_bytes) < 1000:
        return jsonify({"error": "Empty or very small image upload"}), 400

    data = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        return jsonify({"error": "Malformed/corrupt image"}), 400

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({"error": "RGB conversion failed", "details": str(e)}), 400

    try:
        with mp_hands.Hands(    # create a new instance for EVERY request!
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as hands:
            results = hands.process(rgb)
    except Exception as e:
        return jsonify({"error": "MediaPipe crash", "details": str(e)}), 500

    sign = ""
    confidence = 0.0
    sentence = ""

    if results.multi_hand_landmarks:
        right_hand = get_hand_by_label(results, "Right")
        left_hand = get_hand_by_label(results, "Left")

        raw = None
        if right_hand is not None and left_hand is not None:
            raw = np.concatenate([hand_to_flat(right_hand), hand_to_flat(left_hand)])
        elif right_hand is not None:
            raw = hand_to_flat(right_hand)
        elif left_hand is not None:
            raw = hand_to_flat(left_hand)

        if raw is not None:
            features = extract_features(raw).reshape(1, -1)
            proba = pipeline.predict_proba(features)[0]
            top_idx = int(np.argmax(proba))
            confidence = float(proba[top_idx])

            if confidence >= CONFIDENCE_THRESHOLD:
                sign = label_encoder.inverse_transform([top_idx])[0]
                sb.add_sign(sign)

    if sb.update():
        sentence = sb.get_sentence()
        sb.reset()

    return jsonify({
        "sign": sign,
        "confidence": confidence,
        "sentence": sentence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)