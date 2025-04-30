# AVIATION INCIDENT PREDICTION

**Insight:**
This project utilizes the most up-to-date data from the Department of Transportation, incorporating crashed flights and on-time flights data spanning from 1990 to 2024.

**Purpose:**
Our predictive model aims to achieve the following:
* **Enhanced Safety:** Reduce aviation accidents and incidents, ultimately saving lives and preventing injuries.
* **Proactive Risk Management:** Enable the anticipation of potential hazards for effective mitigation and prevention, moving beyond a reactive approach.
* **Resource Allocation:** Optimize the distribution of resources for inspections, maintenance, and training by accurately predicting high-risk periods, areas, and circumstances.
* **Cost Reduction:** Minimize costs associated with delays, damages, legal fees, reputational damage, and most importantly, prevent the tragic loss of life.
* **Improved Operational Efficiency:** Foster a seamless and efficient travel experience for everyone involved by minimizing disruptions and negative incidents.

**Installation:**
To get started with this project:

1.  **Fork the Repository:** Click the "Fork" button at the top right of this page to create a copy of this repository in your GitHub account.
2.  **Run in Codespace:** Once you've forked the repository, navigate to your forked version and click the "Code" button. From the dropdown menu, select "Create codespace on main". This will open a pre-configured development environment in your browser.
3.  **Run the Notebook:** Within the Codespace environment, you will find the project notebooks (likely with a `.ipynb` extension). Open the desired notebook and follow the instructions within to run the code.

**Requirements**

numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
scipy==1.15.2
matplotlib==3.10.1
seaborn==0.13.2
ipykernel==6.29.5
pyarrow==19.0.1
requests==2.32.3
access-parser==0.0.6
beautifulsoup4==4.13.4
streamlit==1.44.1

**Model Performance Summary**
The Aviation Incidence Prediction model achieved high performance, with around 99.7% accuracy on unseen data. It excels at correctly identifying non-incident cases (very low false positive rate). However, it has a slightly higher error rate in missing actual incident cases (around 8% false negatives).
Attempts to optimize the model's settings didn't significantly change its overall accuracy. Further examination of the model's predictions for incident cases is important to ensure a good balance between correctly identifying true incidents and minimizing missed cases, especially given the potential impact of false negatives. 


**Notebook Links**
1.	Data Acquisition – 
https://github.com/4GeeksAcademy/aviation_final_project/blob/main/src/01-data_acquisition.ipynb
2.	Data Preparation – 
https://github.com/4GeeksAcademy/aviation_final_project/blob/main/src/02-data_prepration.ipynb
3.	Model Building – 
https://github.com/4GeeksAcademy/aviation_final_project/blob/main/src/03-model_building_Dyimah.ipynb


**Data Sources:**
Airline Service Quality Performance 234 (On-Time performance data): [https://www.bts.gov/browse-statistical-products-and-data/bts-publications/airline-service-quality-performance-234-time

NTSB incident data: [https://data.ntsb.gov/avdata](https://data.ntsb.gov/avdata)

**Contact:**
[https://github.com/4GeeksAcademy/aviation\_final\_project](https://github.com/4GeeksAcademy/aviation_final_project)





