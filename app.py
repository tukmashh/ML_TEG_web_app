from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Загружаем модель и скейлеры
model = tf.keras.models.load_model("TEG_ML_model.h5", compile=False)
scalers = joblib.load("scalers.pkl")

# Все названия признаков
feature_names = [
    "mc_n", "tau_eff_n", "eta_F_n", "E_g_n", "kappa_L_300_n",
    "mc_p", "tau_eff_p", "eta_F_p", "E_g_p", "kappa_L_300_p",
    "h", "R", "T_h"
]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_values = {name: "" for name in feature_names}

    if request.method == "POST":
        try:
            # Считываем и преобразуем входные значения
            features = []
            for name in feature_names:
                value = float(request.form.get(name))
                input_values[name] = value
                features.append(value)

            # Создаём DataFrame
            input_df = pd.DataFrame([features], columns=feature_names)

            # Применяем скейлер к каждой колонке
            for col in input_df.columns:
                input_df[col] = scalers[col].transform(input_df[[col]])

            # Предсказание
            input_data = input_df.to_numpy()
            prediction = model.predict(input_data)[0]
            prediction = [round(float(p), 3) for p in prediction]

        except Exception as e:
            prediction = f"Ошибка: {e}"

    return render_template("index.html", prediction=prediction, input_values=input_values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
