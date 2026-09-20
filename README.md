# AI/ML Internship — Muhammad Ali

> A 6-week hands-on journey from Python fundamentals to Deep Learning, Generative AI, and a diabetes-risk prediction capstone project.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## About Me

I'm Muhammad Ali, a Computer Science student in my 7th semester with a main interest in AI and Machine Learning. My projects apply these concepts to practical problems, including InsuGenie, an AI-powered diabetes monitoring platform involving machine learning and computer vision, a spam email classifier, and a customer intelligence platform using K-Means clustering.

## Program Progress

| Week | Focus | Status |
|------|-------|--------|
| Week 1 | Git, Python, Pandas, and EDA | Complete |
| Week 2 | Classic Machine Learning | Complete |
| Week 3 | Deep Learning | Complete |
| Week 4 | Generative AI | Complete |
| Weeks 5–6 | Capstone Project | Complete |

## Module Work

### Week 1 — Git, Python, and EDA

| File | Description |
|------|-------------|
| `Lab2_MuhammadAli_Python_NumPy_Pandas.ipynb` | Python, NumPy, and Pandas fundamentals |
| `Lab_03_MuhammadAli_Colab_File.ipynb` | Data cleaning and preprocessing |
| `Lab_04_Muhammad_Ali_Colab_File.ipynb` | Data visualization using the Titanic dataset |
| `Assignment1_EDA_MuhammadAli.ipynb` | Weekend Assignment 1: full EDA on NYC Airbnb Open Data |

### Week 2 — Classic Machine Learning

| File | Description |
|------|-------------|
| `Lab_06_MuhammadAli.ipynb` | Introduction to ML using the Iris dataset |
| `Lab_07_Muhammad_Ali.ipynb` | Regression models using California Housing |
| `Lab_08_MuhammadAli.ipynb` | Classification models |
| `Lab_09_Muhammad_Ali.ipynb` | Model evaluation metrics |
| `Lab_10_Muhammad_Ali.ipynb` | Feature engineering |
| `Assignment2_Muhammad_Ali.ipynb` | End-to-end insurance regression pipeline |

### Week 3 — Deep Learning

| File | Description |
|------|-------------|
| `Lab_11_MuhammadAli.ipynb` | Neural-network fundamentals from scratch in NumPy |
| `Lab_12_Muhammad_Ali.ipynb` | TensorFlow and Keras Sequential models |
| `Lab_13_MuhammadAli.ipynb` | CNNs on Fashion-MNIST |
| `Lab_14_Muhammad_Ali_.ipynb` | RNN, LSTM, and GRU sentiment models |
| `Lab_15_Muhammad_Ali.ipynb` | Model tuning and optimization |
| `Assignment3_MuhammadAli.ipynb` | Customer churn prediction with a tuned Keras model |

### Week 4 — Generative AI

| File | Description |
|------|-------------|
| `Lab_16_Muhammad_Ali.ipynb` | Large Language Models, tokenization, and sampling |
| `Lab_17_Muhammad_Ali.ipynb` | Prompt engineering |
| `Lab_18_Muhammad_Ali.ipynb` | Retrieval-Augmented Generation (RAG) |

Labs 19 and 20 covered AI agents, tool use, APIs, model integration, multi-turn messages, streaming, and error handling. Their corresponding files can be added to this repository whenever available.

## Capstone: Diabetes Risk Prediction

A machine-learning application that predicts a patient's diabetes risk from routine diagnostic measurements. The project uses the Pima Indians Diabetes Dataset: 768 patient records, eight diagnostic features, and a binary target.

### Problem Statement

Build a supervised machine-learning model that predicts whether a patient is likely to have diabetes from eight diagnostic health measurements, while surfacing the key risk factors behind each prediction. The goal is early, low-cost risk flagging for screening contexts, not clinical diagnosis.

### Dataset

**Pima Indians Diabetes Dataset** ([source](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)) — 768 patient records, eight diagnostic features, and a binary target.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| **Outcome** | **Target: 1 = diabetic, 0 = non-diabetic** |

### Results

| Model | Recall | Precision | ROC-AUC |
|---|---:|---:|---:|
| Baseline (majority class) | 0.000 | — | 0.500 |
| Logistic Regression (tuned) | 0.648 | 0.636 | 0.835 |
| Keras Neural Network | 0.852 | 0.648 | 0.880 |
| **Random Forest (tuned, selected)** | **0.796** | **0.811** | **0.941** |
| **Random Forest at tuned threshold (0.412)** | **0.852** | **0.793** | **0.941** |

The final model is a tuned Random Forest. Threshold tuning recovers the neural network's recall while maintaining better precision. Bootstrap resampling produced a 95% ROC-AUC confidence interval of **[0.901, 0.972]**.

### Try the Application

```bash
cd app
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

### Python Usage

```python
import joblib

model = joblib.load("app/models/diabetes_best_model.joblib")
scaler = joblib.load("app/models/diabetes_scaler.joblib")
feature_columns = joblib.load("app/models/diabetes_feature_columns.joblib")
threshold = joblib.load("app/models/diabetes_decision_threshold.joblib")
```

See `app/app.py` for the complete prediction function and use the Flask interface for form-based predictions.

### Methodology

1. **EDA:** examined class balance, feature distributions, and correlations.
2. **Data cleaning:** replaced biologically impossible zero values with outcome-grouped median imputation.
3. **Feature engineering:** created BMI and age categories plus glucose/BMI and insulin/glucose interaction terms.
4. **Modeling:** compared Logistic Regression, KNN, Decision Tree, Random Forest, and a Keras neural network.
5. **Selection:** prioritized diabetic-class recall and ROC-AUC over raw accuracy.
6. **Testing:** performed error analysis, threshold tuning, bootstrap stability checks, and age-band subgroup evaluation.

### Repository Structure

```text
├── app/
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/
│   ├── static/style.css
│   └── models/
├── data/diabetes_cleaned_features.csv
├── docs/
│   ├── Project_Report.pdf
│   └── presentation.pptx
├── notebooks/
│   ├── 01_EDA_Feature_Engineering.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Testing_Refinement.ipynb
└── README.md
```

### Limitations

- The binary target estimates diabetes risk; it does not classify diabetes type.
- The model was trained on a single historical population of 768 records.
- This is a screening aid, not a diagnostic tool.
- Several source features required imputation because missing values were encoded as zero.

### Demo

- **Live app:** Add a deployed URL here if the app is deployed.
- **Demo video:** Add a Google Drive or other demo link here if available.

## Tech Stack

- **Languages and core:** Python, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Classic ML:** Scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **Web app:** Flask
- **Model persistence:** joblib
- **Environment:** Google Colab and local Python

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Muhammad Ali — AI/ML Internship Program, Capstone Project
