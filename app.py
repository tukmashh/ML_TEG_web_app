from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("TEG_ML_model.h5", compile=False)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form.get("eta_F_n")),
                float(request.form.get("eta_F_p")),
                float(request.form.get("h")),
                float(request.form.get("R")),
                float(request.form.get("T_h")),
            ]
            input_data = np.array([features])
            prediction = model.predict(input_data)[0]
            prediction = [round(float(p), 3) for p in prediction]
        except Exception as e:
            prediction = f"Ошибка: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
