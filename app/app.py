"""
Diabetes Risk Prediction — Flask web app.

Loads the tuned Random Forest model (and its scaler / feature columns / decision
threshold) produced by the capstone notebooks and serves a small form-based UI:
enter routine diagnostic measurements, get a risk prediction back.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "diabetes_best_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_scaler.joblib"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "diabetes_feature_columns.joblib"))
threshold = joblib.load(os.path.join(MODEL_DIR, "diabetes_decision_threshold.joblib"))

app = Flask(__name__)
# Used only to sign the session cookie that remembers a visitor's last-entered
# values (see "prefill" below) — no sensitive data involved. Fine as a fixed
# local value for a capstone demo; a real deployment would set this via an
# environment variable instead.
app.secret_key = "diabetes-risk-prediction-capstone-dev-key"

# Population medians from the training data, used as silent defaults for the two
# lab values (Insulin, SkinThickness) that almost nobody outside a clinic would
# know off-hand. A public screening tool shouldn't block on values this obscure.
INSULIN_DEFAULT = 102.5
SKIN_THICKNESS_DEFAULT = 28.0

# A public value can legitimately exceed anything seen in training (e.g. a glucose
# reading of 300). Tree-based models cannot extrapolate past the range they were
# trained on — every value above the training max routes to the same terminal leaf,
# so probability silently plateaus instead of climbing further. TRAINING_RANGES
# records the observed min/max per raw feature so the app can detect this and warn,
# instead of silently returning a flat, misleadingly low number.
TRAINING_RANGES = {
    "Pregnancies": (0, 17),
    "Glucose": (44, 199),
    "BloodPressure": (24, 122),
    "SkinThickness": (7, 99),
    "Insulin": (14, 846),
    "BMI": (18.2, 67.1),
    "DiabetesPedigreeFunction": (0.078, 2.42),
    "Age": (21, 81),
}

# Clinical safety-net: WHO/ADA diagnostic criteria for a 2-hour OGTT reading say
# >=200 mg/dL is diabetes range on its own, independent of any other factor. The
# trained model can't reliably say this itself once inputs exceed its training
# range, so this threshold is applied as an explicit, transparent override rather
# than trusting a plateaued model score for a value that's diagnostic by itself.
CLINICAL_GLUCOSE_DIABETIC_THRESHOLD = 200

# The override above decides the Higher/Lower Risk *label*, but the displayed
# *percentage* still came straight from the model, which plateaus (~33%) for any
# glucose past ~199 — so a 300 and a 400 showed the identical, misleadingly low
# number. This blends the model's score toward a high-confidence ceiling as
# glucose climbs from 200 (just diagnostic) to 300 (severe), so the percentage
# itself keeps responding instead of freezing. It caps at 97%, not 100%: this is
# a screening estimate, and claiming absolute certainty would overstate what any
# model — or clinical rule — can honestly promise. Beyond 300, clinical severity
# doesn't keep scaling in any well-established way, so the cap holds rather than
# inventing further precision the data can't support.
GLUCOSE_SEVERITY_CEILING = 0.97
GLUCOSE_SEVERITY_SPAN = 100  # 200 -> 300 mg/dL spans the full blend


def apply_glucose_severity(prob, glucose):
    if glucose < CLINICAL_GLUCOSE_DIABETIC_THRESHOLD:
        return prob
    severity = min((glucose - CLINICAL_GLUCOSE_DIABETIC_THRESHOLD) / GLUCOSE_SEVERITY_SPAN, 1.0)
    return prob + (GLUCOSE_SEVERITY_CEILING - prob) * severity


def glucose_screening_category(glucose):
    """Plain-language screening band for the app's 2-hour OGTT-style glucose input.

    This is a glucose-range screen, not a diabetes-type classifier.
    The underlying Pima dataset has only a binary Outcome label, so it cannot
    distinguish type 1, type 2, or gestational diabetes.
    """
    if glucose >= 200:
        return "Diabetes range", "A 2-hour glucose value of 200 mg/dL or higher is in the diabetes range."
    if glucose >= 140:
        return "Prediabetes range", "A 2-hour glucose value from 140–199 mg/dL is in the prediabetes range."
    return "Below prediabetes range", "This 2-hour glucose value is below the prediabetes range."

# A simplified, plain-language stand-in for DiabetesPedigreeFunction (a computed
# genetic-risk score nobody would know as a raw number). Maps a question anyone
# can answer to an approximate value spanning the dataset's typical range.
FAMILY_HISTORY_TO_DPF = {
    "none": 0.20,
    "one": 0.40,
    "two_or_more": 0.80,
}


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def age_group(age):
    if age < 30:
        return "Young Adult"
    elif age < 45:
        return "Adult"
    elif age < 60:
        return "Middle-Aged"
    return "Senior"


def gauge_point(value_pct, radius, cx=100, cy=100):
    """Point on the 180-degree gauge arc for a 0-100 value (0=left, 100=right)."""
    theta = math.pi - (value_pct / 100) * math.pi
    return cx + radius * math.cos(theta), cy - radius * math.sin(theta)


def predict_diabetes_risk(patient_dict):
    """Re-derives the Sprint 1 engineered features and returns a prediction dict."""
    row = pd.DataFrame([patient_dict])

    row["BMI_Category"] = row["BMI"].apply(bmi_category)
    row["Age_Group"] = row["Age"].apply(age_group)
    row["Glucose_BMI_Interaction"] = row["Glucose"] * row["BMI"]
    row["Insulin_Glucose_Ratio"] = row["Insulin"] / row["Glucose"].replace(0, 1)

    row = pd.get_dummies(row, columns=["BMI_Category", "Age_Group"])
    row = row.reindex(columns=feature_columns, fill_value=0)
    row_scaled = scaler.transform(row)

    prob = float(model.predict_proba(row_scaled)[0, 1])
    model_says_high_risk = prob >= threshold
    # Which raw inputs fall outside what the model ever saw in training —
    # the model's score for these is not reliable, regardless of direction.
    FIELD_LABELS = {
        "Pregnancies": "Pregnancies", "Glucose": "Glucose", "BloodPressure": "Blood Pressure",
        "SkinThickness": "Skin Thickness", "Insulin": "Insulin", "BMI": "BMI",
        "DiabetesPedigreeFunction": "Family History score", "Age": "Age",
    }
    out_of_range = []
    for key, (lo, hi) in TRAINING_RANGES.items():
        if key in patient_dict and not (lo <= patient_dict[key] <= hi):
            out_of_range.append(FIELD_LABELS[key])

    # Clinical override: a 2-hour glucose reading of 200+ is diagnostic of diabetes
    # on its own under WHO/ADA criteria, independent of the model's score. Applied
    # explicitly and flagged, rather than silently trusting a plateaued model output.
    clinical_override = patient_dict["Glucose"] >= CLINICAL_GLUCOSE_DIABETIC_THRESHOLD
    is_high_risk = model_says_high_risk or clinical_override

    # The displayed percentage gets the same severity treatment as the label, so
    # a 300 mg/dL reading doesn't show the same flat number as a 350 or a 400.
    prob = apply_glucose_severity(prob, patient_dict["Glucose"])

    # Top contributing factors, from the model's global feature importances,
    # limited to the raw inputs this patient actually supplied.
    importances = dict(zip(feature_columns, model.feature_importances_))
    raw_importance = {
        "Glucose": importances.get("Glucose", 0) + importances.get("Glucose_BMI_Interaction", 0) / 2,
        "BMI": importances.get("BMI", 0) + importances.get("Glucose_BMI_Interaction", 0) / 2,
        "Insulin": importances.get("Insulin", 0) + importances.get("Insulin_Glucose_Ratio", 0) / 2,
        "Age": importances.get("Age", 0),
        "SkinThickness": importances.get("SkinThickness", 0),
        "BloodPressure": importances.get("BloodPressure", 0),
        "Pregnancies": importances.get("Pregnancies", 0),
        "DiabetesPedigreeFunction": importances.get("DiabetesPedigreeFunction", 0),
    }
    top_factors = sorted(raw_importance.items(), key=lambda x: x[1], reverse=True)[:3]

    # Gauge geometry: needle points at the displayed probability; a short tick
    # marks where the decision threshold sits on the same 0-100 arc.
    display_pct = round(prob * 100, 1)
    glucose_category, glucose_category_detail = glucose_screening_category(patient_dict["Glucose"])
    needle_x, needle_y = gauge_point(display_pct, radius=65)
    thresh_pct = round(threshold * 100, 1)
    tick_x1, tick_y1 = gauge_point(thresh_pct, radius=72)
    tick_x2, tick_y2 = gauge_point(thresh_pct, radius=90)

    return {
        "prediction": "Higher Risk" if is_high_risk else "Lower Risk",
        "is_high_risk": is_high_risk,
        "probability": display_pct,
        "threshold": thresh_pct,
        "glucose_category": glucose_category,
        "glucose_category_detail": glucose_category_detail,
        "top_factors": [f[0] for f in top_factors],
        "clinical_override": clinical_override,
        "out_of_range": out_of_range,
        "needle_x": round(needle_x, 1),
        "needle_y": round(needle_y, 1),
        "tick_x1": round(tick_x1, 1),
        "tick_y1": round(tick_y1, 1),
        "tick_x2": round(tick_x2, 1),
        "tick_y2": round(tick_y2, 1),
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", saved=session.get("last_input", {}))


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    session["last_input"] = form.to_dict()

    height_m = float(form["height_cm"]) / 100
    weight_kg = float(form["weight_kg"])
    bmi = round(weight_kg / (height_m ** 2), 1)

    insulin = float(form["insulin"]) if form.get("insulin") else INSULIN_DEFAULT
    skin_thickness = float(form["skin_thickness"]) if form.get("skin_thickness") else SKIN_THICKNESS_DEFAULT
    dpf = FAMILY_HISTORY_TO_DPF[form["family_history"]]

    patient_dict = {
        "Pregnancies": float(form["pregnancies"]),
        "Glucose": float(form["glucose"]),
        "BloodPressure": float(form["blood_pressure"]),
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": float(form["age"]),
    }

    result = predict_diabetes_risk(patient_dict)
    result["current_pregnancy"] = form.get("current_pregnancy", "no") == "yes"
    result["type_classification_note"] = (
        "This dataset supports binary diabetes-risk screening only. It does not contain "
        "labels for type 1, type 2, or gestational diabetes, so those types are not "
        "predicted by the machine-learning model."
    )

    return render_template(
        "result.html",
        result=result,
        patient=patient_dict,
        bmi=bmi,
        used_default_insulin=not form.get("insulin"),
        used_default_skin=not form.get("skin_thickness"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
