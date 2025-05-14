# <img src="https://github.com/4GeeksAcademy/Madesh10-aviation_final_project/blob/main/src/static/Logo.png" alt="Logo" width="100" /> Flight Incident Risk Predictor

A Streamlit-based machine learning web application that predicts the risk of flight incidents in U.S. using historical aviation data from the past five years.

<img src="https://github.com/4GeeksAcademy/Madesh10-aviation_final_project/blob/main/src/static/photo.jpg" alt="App Screenshot" width="400" />

---

## 📌 Table of Contents

- [Overview](#-overview)
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

