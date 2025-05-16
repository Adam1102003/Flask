from flask import Flask, request, jsonify
import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import io
# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

# === Model & Labels Loading ===
LABELS_PATH = 'all_classes.txt'
MODEL_PATH = 'inception_classifier.keras'

if not os.path.exists(LABELS_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Make sure 'all_classes.txt' and 'inception_classifier.keras' exist.")

with open(LABELS_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# === Helper Functions ===
def load_data():
    try:
        with open("children_data.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_data(data):
    with open("children_data.json", "w") as file:
        json.dump(data, file, indent=4)

def calculate_correct_pieces(user_order, image_name, puzzle_type, puzzle_size):
    base_path = f"Puzzle_data/{puzzle_type}_Shuffled_Pieces"
    if puzzle_size == "2x2":
        image_folder_path = os.path.join(base_path + "(2x2)", image_name.split("_")[0], image_name)
    elif puzzle_size == "3x3":
        image_folder_path = os.path.join(base_path + "(3x3)", image_name.split("_")[0], image_name)
    else:
        return "Invalid puzzle size."

    file_path = os.path.join(image_folder_path, "shuffle_order.txt")

    try:
        with open(file_path, "r") as file:
            shuffle_order = file.read().strip().split(",")
        print("Shuffle order:", shuffle_order)
        print("User order:", user_order)

        # تحويل الترتيب إلى أعداد صحيحة للمقارنة
        shuffle_order = [int(i) for i in shuffle_order]  # التأكد أن الترتيب المبعثر أعداد صحيحة
        user_order = [int(i) for i in user_order]  # التأكد أن ترتيب المستخدم أعداد صحيحة

        # بناء الترتيب العكسي (reconstructed_order) من shuffle_order
        reconstructed_order = [0] * len(shuffle_order)
        for shuffled_index, original_index in enumerate(shuffle_order):
            reconstructed_order[original_index] = shuffled_index
        
        print("Reconstructed order:", reconstructed_order)

        # مقارنة ترتيب المستخدم مع الترتيب العكسي (reconstructed_order)
        correct_pieces = sum(1 for i, piece in enumerate(user_order) if piece == reconstructed_order[i])
        return correct_pieces
    except FileNotFoundError:
        return 0


def calculate_overall_average(child_data):
    total_correct = child_data["total_correct"]
    total_attempts = child_data["total_attempts"]
    return (total_correct / total_attempts) * 100 if total_attempts > 0 else 0

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((255, 255))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

def predict_label(image_path):
    try:
        preprocessed = preprocess_image(image_path)
        predictions = model.predict(preprocessed)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        return {
            "predicted_label": class_names[predicted_index],
            "confidence": confidence
        }
    except Exception as e:
        return {
            "predicted_label": "Unknown",
            "confidence": 0.0,
            "error": str(e)
        }

# === Validate Puzzle Endpoint ===
@app.route('/validate_puzzle', methods=['POST'])
def validate_puzzle():
    try:
        data = request.get_json()

        required_keys = ["user_order", "image_name", "puzzle_type", "puzzle_size", "username"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required field: {key}"}), 400

        user_order = data["user_order"]
        image_name = data["image_name"]
        puzzle_type = data["puzzle_type"]
        puzzle_size = data["puzzle_size"]
        username = data["username"]

        children_data = load_data()

        if username not in children_data:
            children_data[username] = {
                "total_correct": 0,
                "total_attempts": 0,
                "average": 0,
                "puzzle_stats": {}
            }

        puzzle_key = f"{image_name}_{puzzle_size}"
        if puzzle_key not in children_data[username]["puzzle_stats"]:
            children_data[username]["puzzle_stats"][puzzle_key] = {
                "correct": 0,
                "attempts": 0
            }

        correct_pieces = calculate_correct_pieces(user_order, image_name, puzzle_type, puzzle_size)
        total_pieces = 9 if puzzle_size == "3x3" else 4
        incorrect_pieces = total_pieces - correct_pieces

        # Update attempts
        children_data[username]["total_attempts"] += 1
        children_data[username]["puzzle_stats"][puzzle_key]["attempts"] += 1

        prediction_result = {}

        # Check if solved perfectly
        if correct_pieces == total_pieces:
            children_data[username]["total_correct"] += 1
            children_data[username]["puzzle_stats"][puzzle_key]["correct"] += 1
            result_message = "Perfect! Puzzle solved correctly."
            encouragement_message = "Great job! You completed the puzzle perfectly!"

            # Predict label
            original_image_path = os.path.join(f"Puzzle_data/{puzzle_type}_Original_Images_Modified", image_name.split("_")[0], image_name + ".jpg")
            if os.path.exists(original_image_path):
                prediction_result = predict_label(original_image_path)
            else:
                prediction_result = {"predicted_label": "Not Found", "confidence": 0.0}
        elif correct_pieces == 0:
            result_message = "Incorrect. No pieces are in the right place."
            encouragement_message = "Keep going! You can do it with more practice!"
        else:
            result_message = f"Partially correct. {correct_pieces} out of {total_pieces} pieces are correct."
            encouragement_message = f"Good progress! You've solved {correct_pieces}/{total_pieces} pieces. {incorrect_pieces} pieces remaining."

        # Update average
        overall_average = calculate_overall_average(children_data[username])
        children_data[username]["average"] = round(overall_average, 2)

        save_data(children_data)

        return jsonify({
            "result": result_message,
            "child_data": {
                "username": username,
                "total_correct": children_data[username]["total_correct"],
                "total_attempts": children_data[username]["total_attempts"],
                "average": round(overall_average, 2),
                "encouragement_message": encouragement_message,
                "puzzle_details": {
                    "image_name": image_name,
                    "correct_pieces": correct_pieces,
                    "incorrect_pieces": incorrect_pieces,
                    "total_attempts": children_data[username]["puzzle_stats"][puzzle_key]["attempts"],
                    **prediction_result
                }
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# === Run Server ===
if __name__ == '__main__':
    app.run()
