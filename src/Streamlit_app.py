import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Page config
st.set_page_config(page_title="Flight Incident Predictor", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/combined_data.csv")
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df['departure_time'] = pd.to_numeric(df['departure_time'], errors='coerce')
    return df

# Load model and results
@st.cache_resource
def load_model_and_results():
    model = joblib.load("models/model.pkl")
    feature_maps = joblib.load("models/feature_maps.pkl")
    results = joblib.load("models/results.pkl")
    return model, feature_maps, results

# Feature preparation
def cyclical_encode(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def prepare_features(df, origin_freq_map, dest_freq_map, tail_freq_map):
    X = df.copy()
    X = cyclical_encode(X, 'departure_time', 2400)
    X['origin_freq'] = X['origin'].map(origin_freq_map).fillna(min(origin_freq_map.values()))
    X['destination_freq'] = X['destination'].map(dest_freq_map).fillna(min(dest_freq_map.values()))
    X['tail_number_freq'] = X['tail_number'].map(tail_freq_map).fillna(min(tail_freq_map.values()))
    return X[['departure_time_sin', 'departure_time_cos', 'origin_freq', 'destination_freq', 'tail_number_freq']]

# Main App
def main():
    st.title("✈️ Flight Incident Prediction")

    df = load_data()
    model, feature_maps, results = load_model_and_results()

    tab1, tab2, tab3 = st.tabs(["Incident Predictor", "Model Performance", "Data Exploration"])

    # Tab 1: Prediction
    with tab1:
        st.header("Predict Flight Incident Risk")
        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Origin Airport", sorted(df['origin'].unique()))
            destination = st.selectbox("Destination Airport", sorted(df['destination'].unique()))
        with col2:
            departure_time = st.number_input("Departure Time (e.g., 1430)", min_value=0, max_value=2359, step=5)
            tail_number = st.selectbox("Tail Number", sorted(df['tail_number'].unique()))

        input_df = pd.DataFrame([{
            'origin': origin,
            'destination': destination,
            'departure_time': departure_time,
            'tail_number': tail_number
        }])

        X_pred = prepare_features(
            input_df,
            origin_freq_map=feature_maps['origin'],
            dest_freq_map=feature_maps['destination'],
            tail_freq_map=feature_maps['tail_number']
        )

        if st.button("Predict Incident Probability"):
            prob = model.predict_proba(X_pred)[0, 1]
            st.subheader("Prediction Results")

            col1, col2 = st.columns([1, 2])
            with col1:
                if prob < 0.25:
                    st.markdown("### 🟢 Low Risk")
                elif prob < 0.5:
                    st.markdown("### 🟡 Moderate Risk")
                elif prob < 0.75:
                    st.markdown("### 🟠 High Risk")
                else:
                    st.markdown("### 🔴 Very High Risk")

            with col2:
                st.progress(float(prob))
                st.markdown(f"**Incident Probability: {prob:.1%}**")

            # Feature contribution table
            st.subheader("Feature Contributions")
            contrib_df = pd.DataFrame({
                'Feature': ['Origin', 'Destination', 'Dep Time (Sin)', 'Dep Time (Cos)', 'Tail #'],
                'Value': [
                    f"{origin} ({feature_maps['origin'].get(origin, 0):.3f})",
                    f"{destination} ({feature_maps['destination'].get(destination, 0):.3f})",
                    f"{X_pred['departure_time_sin'].values[0]:.3f}",
                    f"{X_pred['departure_time_cos'].values[0]:.3f}",
                    f"{tail_number} ({feature_maps['tail_number'].get(tail_number, 0):.3f})"
                ],
                'Importance': [
                    results['feature_importances']['origin_freq'],
                    results['feature_importances']['destination_freq'],
                    results['feature_importances']['departure_time_sin'],
                    results['feature_importances']['departure_time_cos'],
                    results['feature_importances']['tail_number_freq']
                ]
            })
            contrib_df['Weighted'] = contrib_df['Importance'] / contrib_df['Importance'].sum()
            st.dataframe(contrib_df.sort_values('Importance', ascending=False))

    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy")
            st.write(f"Training: {results['train_score']:.4f}")
            st.write(f"Test: {results['test_score']:.4f}")
            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(results['classification_report']).transpose())
        with col2:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            ax.plot(results['fpr'], results['tpr'], label=f"AUC = {results['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        st.subheader("Feature Importance")
        fig, ax = plt.subplots()
        keys, vals = zip(*sorted(results['feature_importances'].items(), key=lambda x: x[1], reverse=True))
        ax.bar(keys, vals)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Tab 3: Data Exploration
    with tab3:
        st.header("Exploration")
        st.subheader("Raw Data")
        st.dataframe(df)
        st.subheader("Incident Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='incident', data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("Departure Time vs Incident")
        fig, ax = plt.subplots()
        sns.boxplot(x='incident', y='departure_time', data=df, ax=ax)
        st.pyplot(fig)

# Run app
if __name__ == "__main__":
    main()
