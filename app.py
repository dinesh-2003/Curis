from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from PIL import Image

from gradcam import grad_cam_densenet, detect_orientation
from utils import preprocess_image

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(
    "dfu_densenet_ce_model (1).h5",
    compile=False
)


with open("class_map.json") as f:
    class_map = json.load(f)

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file:
            image = Image.open(file).convert("RGB")
            img_arr = preprocess_image(image, IMG_SIZE)

            # Prediction
            preds = model.predict(img_arr, verbose=0)[0]
            grade_idx = int(np.argmax(preds))
            grade = grade_idx + 1
            confidence = preds[grade_idx] * 100

            # Grad-CAM
            heatmap = grad_cam_densenet(model, img_arr)
            orientation = detect_orientation(heatmap)

            heatmap = cv2.resize(heatmap, IMG_SIZE)
            heatmap_col = cv2.applyColorMap(
                np.uint8(255 * heatmap),
                cv2.COLORMAP_JET
            )

            overlay = heatmap_col * 0.4 + np.array(image.resize(IMG_SIZE))

            output_path = os.path.join(
                UPLOAD_FOLDER, "gradcam_result.png"
            )
            cv2.imwrite(
                output_path,
                cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_RGB2BGR)
            )

            return render_template(
                "index.html",
                grade=grade,
                confidence=f"{confidence:.2f}",
                orientation=orientation,
                image_path=output_path
            )

    return render_template("index.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


