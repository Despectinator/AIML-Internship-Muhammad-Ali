# Diabetes Risk Prediction

A machine learning capstone project that predicts a patient's risk of diabetes from routine diagnostic measurements, built as part of a 6-week AI/ML Internship Program.

## Problem Statement

Build a supervised machine learning model that predicts whether a patient is likely to have diabetes based on 8 diagnostic health measurements, and surface the key risk factors driving each prediction. Early, low-cost risk flagging like this is genuinely useful in primary-care / screening contexts where full diagnostic testing isn't always immediately available.

## Dataset

**Pima Indians Diabetes Dataset** ([source](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)) — 768 patient records, 8 diagnostic features, binary target.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| **Outcome** | **Target: 1 = diabetic, 0 = non-diabetic** |

## Results

| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Baseline (majority class) | 0.000 | — | 0.500 |
| Logistic Regression (tuned) | 0.648 | 0.636 | 0.835 |
| Keras Neural Network | 0.852 | 0.648 | 0.880 |
| **Random Forest (tuned, selected)** | **0.796** | **0.811** | **0.941** |
| **Random Forest @ tuned threshold (0.412)** | **0.852** | **0.793** | 0.941 |

**Final model: tuned Random Forest**, selected over a Keras neural network because it had a meaningfully better ROC-AUC (0.941 vs. 0.880) and precision (0.811 vs. 0.648), for only a small recall trade-off — one that threshold tuning fully recovers. At a decision threshold of 0.412 (instead of the default 0.5), the Random Forest matches the neural network's recall (0.852) while keeping far fewer false alarms (0.793 vs. 0.648 precision).

Test-set ROC-AUC of 0.941 is stable under bootstrap resampling (500 iterations): 95% CI **[0.901, 0.972]**.

Full metrics, plots, and reasoning are in [`docs/Project_Report.pdf`](docs/Project_Report.pdf) and the notebooks below.

## Try it

The trained model is served through a small Flask app — enter routine diagnostic measurements and get a risk prediction back.

```bash
cd app
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Repository Structure

```
├── app/
│   ├── app.py                              # Flask application
│   ├── requirements.txt
│   ├── templates/                          # index.html (form), result.html
│   ├── static/style.css
│   └── models/                             # trained model, scaler, feature columns, threshold
├── notebooks/
│   ├── 01_EDA_Feature_Engineering.ipynb    # Sprint 1 — EDA, cleaning, feature engineering
│   ├── 02_Modeling.ipynb                   # Sprint 2 — model training, tuning, comparison
│   └── 03_Testing_Refinement.ipynb         # Sprint 3 — error analysis, threshold tuning, robustness
├── data/
│   └── diabetes_cleaned_features.csv       # Cleaned, feature-engineered dataset
├── docs/
│   ├── Project_Report.pdf                  # Full project report (problem, EDA, app, results, future work)
│   └── presentation.pptx                   # Final presentation slides
└── README.md
```

## Methodology

1. **EDA** — examined class balance (~65/35 split), feature distributions, and correlations with the target.
2. **Data cleaning** — `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contained biologically-impossible zeros (used as an implicit "not recorded" marker). Fixed via median imputation **grouped by `Outcome`**, preserving the distributional difference between diabetic and non-diabetic patients rather than pulling both groups toward one blended average.
3. **Feature engineering** — clinically-grounded bins (`BMI_Category`, `Age_Group`) and interaction/ratio terms (`Glucose_BMI_Interaction`, `Insulin_Glucose_Ratio`) reflecting known compounding diabetes risk factors.
4. **Modeling** — compared Logistic Regression, KNN, Decision Tree, Random Forest (all tuned via `GridSearchCV` with 5-fold stratified cross-validation), and a Dropout/EarlyStopping-tuned Keras neural network.
5. **Selection** — prioritized recall on the diabetic class and ROC-AUC over raw accuracy, since a missed diagnosis is costlier than a false alarm in a screening context.
6. **Testing & refinement** — individual error analysis on false negatives/positives, precision-recall threshold tuning, bootstrap stability checks, and a subgroup performance check across age bands.

## Usage

```python
import joblib

model = joblib.load("app/models/diabetes_best_model.joblib")
scaler = joblib.load("app/models/diabetes_scaler.joblib")
feature_columns = joblib.load("app/models/diabetes_feature_columns.joblib")
threshold = joblib.load("app/models/diabetes_decision_threshold.joblib")

# See app/app.py for the full predict_diabetes_risk() function, which re-derives
# engineered features from raw patient inputs automatically. Or just run the app
# (see "Try it" above) for a form-based UI instead of calling this directly.
```

## Live Demo

- **Live app:** [add your deployed URL here, if deployed]
- **Demo video:** [add your Google Drive link here — shows the app running locally if not deployed]

## Limitations

- The underlying Pima dataset has a **binary `Outcome` target only (0/1)**. The model therefore estimates diabetes risk; it does **not** classify type 1 diabetes, type 2 diabetes, or gestational diabetes. The UI now makes this limitation explicit and shows a separate 2-hour OGTT glucose screening band.
- The optional current-pregnancy question is contextual only and does not feed the trained model. A pregnancy-related result should not be interpreted as a diagnosis of gestational diabetes.
- Trained on a single-population historical dataset (768 records); performance on other populations is not validated.
- `Insulin` and `SkinThickness` required imputing ~49% and ~30% of values respectively due to missing-data-as-zero encoding in the source data.
- This is a screening aid, not a diagnostic tool, and should never be presented to a real patient as one.
- Subgroup performance across age bands was checked on a small test set (154 rows) — group-level numbers are directional, not conclusive.

## Tech Stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Flask, matplotlib, seaborn, joblib

## Author

Syed Muhammad Ali Bokhari — AI/ML Internship Program, Capstone Project (Weeks 5–6)
