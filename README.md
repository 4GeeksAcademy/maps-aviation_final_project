# ✈️ Flight Incident Risk Predictor

A Streamlit-based machine learning web application that predicts the risk of flight incidents in U.S. using historical aviation data from the past five years.

![App Screenshot](/workspaces/Madesh10-aviation_final_project/src/static/photo.jpg)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Live Demo](#-live-demo)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This project uses a machine learning model to estimate the likelihood of flight incidents based on origin and destination airports and departure time. It provides visual insights into model performance and the historical dataset.

The app is built with **Streamlit** and integrates interactive visualizations, model evaluation metrics, and animated flight mapping.

---

## ✅ Features

- 🔮 **Incident Predictor**  
  Estimate the probability of an incident for selected airports and departure times.

- 📈 **Model Performance**  
  Visualize confusion matrix, ROC curve, feature importance, and accuracy metrics.

- 📊 **Dataset Explorer**  
  Explore historical incident data through charts and plots.

- 🌍 **Flight Animation**  
  View animated flight paths between selected airports on a map.

---

## 🧠 Model Details

- **Algorithm**: `HistGradientBoostingClassifier` (Scikit-learn)
- **Features Used**:
  - Origin Airport
  - Destination Airport
  - Departure Time (encoded)
- **Preprocessing**: Label encoding for categorical variables
- **Performance**: Accuracy ~92%, AUC ~0.91

---

## 📁 Dataset

- **Source**: FAA, NTSB, and U.S. aviation data repositories
- **Time Frame**: Last 5 years
- **Size**: ~100,000 records
- **Columns**: Airport codes, timestamps, incident labels

> Dataset is stored in the `data/` directory and preprocessed using `pandas`.

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/flight-incident-risk-predictor.git
cd flight-incident-risk-predictor
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## ▶️ Usage

- Navigate to the Incident Predictor tab and select origin, destination, and departure time.

- View the predicted incident risk % .

- Explore model performance and dataset plots in other tabs.

- Select airports to see a flight animation 


# ✈️ Flight Incident Risk Predictor App 

## 🌐 Live Demo
👉 Try the App ([Live](https://madesh10-aviation-final-project.onrender.com/))

# ✈️ Flight Incident Risk Predictor

A Streamlit web application that predicts the risk of USA flight incidents based on historical aviation data from the past 5 years. The app offers interactive visualizations and predictions powered by a machine learning model trained on flight incident data.


## 🚀 Features

This application provides the following functionalities across four interactive tabs:

1. **Incident Predictor Based on Airports Selected**  
   Predict the likelihood of a flight incident based on selected origin and destination airports and departure time.

2. **Model Performance**  
   Visual insights into model metrics such as accuracy, confusion matrix, feature importance, and more.

3. **Data Set Plots**  
   Explore trends and distributions in the historical flight incident dataset through various visualizations.

4. **Flight Animation**  
   Animated map visualizing the flight path between selected airports, enhancing the interpretability of predictions.

---

## 🧠 Machine Learning Model

- **Model Used**: `HistGradientBoostingClassifier` from Scikit-learn
- **Features**: Origin Airport, Destination Airport, Departure Time
- **Target**: Flight Incident Occurrence (Yes/No)
- **Encoding**: Categorical variables are encoded using pre-trained encoders

---

## 📊 Dataset

- **Source**: FAA for open aviation incident database and NTSB for 
- **Scope**: U.S. flight incident and all flight records from the past 5 years
- **Processed**: Cleaned, encoded, and pre-processed for model training and visualization

---

## 🛠️ Installation

### Requirements

- Python 3.8+
- pip

### Clone and Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/flight-incident-risk-predictor.git
cd flight-incident-risk-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
