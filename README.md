# Flight Incident Risk Predictor

A Streamlit-based machine learning web application that predicts the risk of flight incidents in U.S. using historical aviation data from the past five years.

<img src="https://github.com/4GeeksAcademy/Madesh10-aviation_final_project/blob/main/src/static/photo.jpg" alt="App Screenshot" width="400"/>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Motivation](#-motivation)
- [Features](#-features)
- [Machine Learning Model Details](#-machine-learning-model-details)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Live Demo](#-live-demo)
- [Authors](#-authors)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This project uses a machine learning model to estimate the likelihood of flight incidents based on origin and destination airports and departure time. It provides visual insights into model performance and the historical dataset.

The app is built with **Streamlit** and integrates interactive visualizations, model evaluation metrics, and animated flight mapping.

---

## 🔍 Problem Statement

Air travel is one of the safest modes of transportation, yet certain flight incidents—such as severe turbulence, emergency landings, and mechanical failures—can pose risks to passenger safety and operational efficiency. While aviation authorities and airlines utilize historical data and maintenance reports to enhance safety protocols, individual travelers often lack access to predictive insights about their specific flights.

Existing flight risk assessments generally focus on macro-level factors such as weather conditions, aircraft type, and pilot experience, but do not provide personalized risk estimates based on selected airports and travel time. However, data-driven models have shown potential in uncovering patterns in flight incidents by analyzing key parameters like departure time, flight routes, and historical incident data.

This web application addresses the gap in personalized flight risk predictions by leveraging machine learning models trained on flight data. The system allows travelers to input their departure airport, destination, and flight time, providing a probability estimate of an incident occurrence. By integrating historical flight trends and cyclical encoding techniques for departure time, the model enhances accuracy in forecasting potential disruptions.

---

## 🔍 Motivation

while flight safety in the U.S. was relatively stable for several years, reported [data](https://trends.google.com/trends/explore?date=today%205-y&geo=US&q=flight%20crashes&hl=en) shows an uptick in flight crashes around late 2024 to early 2025. Below graph presents a time series of monthly flight crash counts over the past five years. 

<img src="https://github.com/4GeeksAcademy/Madesh10-aviation_final_project/blob/main/src/static/5year_trend.png" alt="App Screenshot" width="600"/>

The flight crash counts were stable between 2020 and mid 2024 with crash count only ranging from 4 to 10. Following that, the flight crash count spiked between late 2024 and early 2025. A significant and sudden surge in flight crashes is visible with count exceeding 100 crashes per week around early 2025. This suggests a major event or systemic issue occurred during this time (e.g., regulatory lapses, mechanical failure patterns, or political administration shift causing significant changes in policy, regulations, and agency priorities).

American tourists are increasingly apprehensive about air travel safety in 2025, a sentiment driven by various factors. According to an AP-NORC [poll](https://apnorc.org/projects/most-continue-to-view-air-travel-as-a-safe-mode-of-transportation/), only 64% of Americans perceive air travel as “very” or “somewhat” safe—a significant decline from previous years.

While commercial aviation remains highly safe, regional variations in accident rates highlight the importance of data-driven risk assessment tools.By analyzing historical flight incident data, this web app helps travelers understand potential risks associated with their flight routes and departure times. The model’s ability to detect patterns in flight incidents aligns with industry efforts to improve aviation safety awareness and passenger decision-making.

Potential Benefits to Aviation Authorities:
- Proactive Risk Mitigation: Helps aviation safety regulators identify high-risk routes and peak-time vulnerabilities for better resource allocation and risk prevention strategies.
- Enhanced Incident Investigation: Provides data-driven insights into patterns of flight incidents, supporting aviation compliance teams in post-incident analysis and regulatory decision-making.
- Real-Time Monitoring Integration: Can serve as a supplementary risk assessment tool when integrated with existing aviation monitoring systems, improving predictive analytics for air traffic control and airport operations.

Potential Benefits to Insurance Companies:
- Better Risk Assessment Models: Supports aviation underwriters in determining insurance premiums based on data-backed flight risks, leading to more accurate pricing models.
- Fraud Detection & Claim Verification: Provides an additional layer of validation when assessing claims related to flight incidents, improving the efficiency of insurance investigations.
- Personalized Travel Insurance Options: Could help insurers develop dynamic pricing for travelers, offering customized insurance plans based on flight route risk predictions.

By offering real-time, data-backed risk assessment, the Flight Incident Risk Predictor App empowers travelers with insightful predictions, helping them make informed decisions and potentially improving aviation safety awareness. Airlines, regulators, and passengers can benefit from enhanced risk visualization, contributing to proactive safety measures and operational planning.

---

## 🚀 Features

- 🔮 **Incident Predictor**  
  Predict the probability of a flight incident based on selected airports and departure times.

- 📈 **Model Performance**  
  Visual insights into model metrics such as accuracy, confusion matrix, feature importance, and more.

- 📊 **Dataset Explorer**  
  Explore trends and distributions in the historical flight incident dataset through various visualizations.

- 🌍 **Flight Animation**  
  View animated flight path between selected airports on a map.

---

## 🧠 Machine Learning Model Details

- **Algorithm**: `HistGradientBoostingClassifier` (Scikit-learn)
- **Features Used**:
  - Origin Airport
  - Destination Airport
  - Departure Time (encoded)
- **Target**: Flight Incident Occurrence (Yes/No)
- **Encoding**: Categorical variables are encoded using pre-trained encoders
- **Performance**: Accuracy ~99.67%, AUC ~0.99

---

## 📁 Dataset

- **Source**: [NTSB](https://www.ntsb.gov/Pages/home.aspx) and [BTS](https://www.bts.gov/browse-statistical-products-and-data/bts-publications/airline-service-quality-performance-234-time) U.S. aviation data repositories
- **Scope**: U.S. flight incident and all flight records from the past 5 years
- **Size**: ~600,000 records
- **Columns**: Airport IATA codes, timestamps, and incident labels
- **Processed**: Cleaned, encoded, and pre-processed for model training and visualization

> Dataset is stored in the `data/` directory and preprocessed using `pandas`.

---

## 🛠️ Installation

### Requirements

- Python 3.11+
- pip

1.Clone the repository:

```bash
git clone https://github.com/4GeeksAcademy/aviation_final_project.git
cd flight-incident-risk-predictor
```

2.Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3.Run the app:
```bash
streamlit run app.py
```

---

## ▶️ Usage

- Navigate to the Incident Predictor tab and select origin, destination, and departure time.

- View the predicted incident risk % .

- Explore model performance and dataset plots in other tabs.

- Select airports to see a flight animation on a U.S. map. 

---

## 📁 Project Structure

```

flight-incident-risk-predictor/
├── app.py
├── models/
│   ├── model.pkl
├── data/
│   ├── interim
│       └── incidents.csv
│       └── ontime.csv
│   ├── processed
│       └── combined_data.csv
│       └── all_encoded.csv
│       └── test_encoded.csv
│       └── train_encoded.csv
│   ├── raw
│       └── incidents.csv
│       └── ontime.csv
├── notebooks/
│   ├── data_acquisition.ipynb
│   ├── data_preparation.ipynb
│   ├── model_building.ipynb
│   ├── app_data_preparation.ipynb
├── requirements.txt
├── README.md
└── src/
    ├── static
        └── plots.png

```

---

## 🌐 Live Demo

👉 Try the App ([Live](https://madesh10-aviation-final-project.onrender.com/))

---

## 🧑‍💻 Authors

Madeshwaran Selvaraj 
[GitHub](https://github.com/Madesh-Selvaraj) | [LinkedIn](https://www.linkedin.com/in/madeshwaran-selvaraj/)

Dyimah Ansah 
[GitHub](https://github.com/Dansah2) | [LinkedIn](https://www.linkedin.com/in/dyimah-ansah/)

Adam Val 
[GitHub](https://github.com/adam6268) | [LinkedIn]()

George Perdrizet 
[GitHub](https://github.com/gperdrizet) | [LinkedIn](https://www.linkedin.com/in/gperdrizet/)

---

## 🚀 Future Work

- Integration of flight animation into the current model prediction.

- real-time weather and aircraft metadata

- Flight route optimization based on risk scores

- Include more features and develop a more user-friendly map on flight travel.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

