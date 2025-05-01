import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.inspection import permutation_importance

# Set page config
st.set_page_config(page_title="Flight Incident Predictor", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("/workspaces/Madesh_datascience_project/data.csv")
    
    # Clean the data
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Convert departure_time to numeric if it's not already
    df['departure_time'] = pd.to_numeric(df['departure_time'])
    
    return df

# Cyclical encoding for time
def cyclical_encode(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

# Frequency encoding for categorical features
def frequency_encode(df, col):
    freq_map = df[col].value_counts(normalize=True).to_dict()
    df[f'{col}_freq'] = df[col].map(freq_map)
    return df, freq_map

# Function to prepare features
def prepare_features(df, origin_freq_map=None, dest_freq_map=None, tail_freq_map=None, training=False):
    # Make a copy of the dataframe to avoid modifying the original
    X = df.copy()
    
    # Cyclical encoding for departure time (assuming 24-hour format or max value 2400)
    max_time = 2400
    X = cyclical_encode(X, 'departure_time', max_time)
    
    # Frequency encoding for origin, destination, and tail_number
    if training:
        X, origin_freq_map = frequency_encode(X, 'origin')
        X, dest_freq_map = frequency_encode(X, 'destination')
        X, tail_freq_map = frequency_encode(X, 'tail_number')
        maps = {'origin': origin_freq_map, 'destination': dest_freq_map, 'tail_number': tail_freq_map}
    else:
        # Use the provided frequency maps
        X['origin_freq'] = X['origin'].map(origin_freq_map)
        X['destination_freq'] = X['destination'].map(dest_freq_map)
        X['tail_number_freq'] = X['tail_number'].map(tail_freq_map)
        maps = None
        
        # Handle any new categories not seen in training
        X['origin_freq'].fillna(min(origin_freq_map.values()), inplace=True)
        X['destination_freq'].fillna(min(dest_freq_map.values()), inplace=True)
        X['tail_number_freq'].fillna(min(tail_freq_map.values()), inplace=True)
    
    # Select features for model
    feature_cols = ['departure_time_sin', 'departure_time_cos', 
                   'origin_freq', 'destination_freq', 'tail_number_freq']
    
    return X[feature_cols], maps

# Function to train model
@st.cache_data
def train_model(df):
    y = df['incident']
    
    # Prepare features
    X, feature_maps = prepare_features(df, training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Initialize and train the model with specified hyperparameters
    model = HistGradientBoostingClassifier(
        l2_regularization=0.1,
        learning_rate=0.01,
        max_iter=1000,
        max_leaf_nodes=15,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Model evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Use permutation importance instead of feature_importances_
    # This works with any model, including HistGradientBoostingClassifier
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = {name: importance for name, importance in 
                          zip(X.columns, perm_importance.importances_mean)}
    
    results = {
        'model': model,
        'feature_maps': feature_maps,
        'train_score': train_score,
        'test_score': test_score,
        'classification_report': report,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'feature_importances': feature_importances
    }
    
    return results

# Main app
def main():
    st.title("✈️ Flight Incident Prediction")
    st.markdown("""
    This app predicts the likelihood of a flight incident based on origin, destination, 
    departure time, and tail number information.
    """)
    
    # Load and process data
    df = load_and_preprocess_data()
    
    # Train model and get results
    with st.spinner("Training model... (this may take a moment)"):
        model_results = train_model(df)
    
    # Create tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["Incident Predictor", "Model Performance", "Data Exploration"])
    
    # Tab 1: Prediction Interface
    with tab1:
        st.header("Predict Flight Incident Risk")
        
        # Create columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique values for dropdowns
            origins = sorted(df['origin'].unique().tolist())
            destinations = sorted(df['destination'].unique().tolist())
            
            # Origin and destination dropdowns
            origin = st.selectbox("Origin Airport", origins)
            destination = st.selectbox("Destination Airport", destinations)
        
        with col2:
            # Departure time (use a slider for better UX)
            departure_time = st.number_input("Departure Time (24hr format e.g. 1430 for 2:30 PM)", 
                                            min_value=0, max_value=2359, value=1200, step=5)
            
            # Tail number
            tail_numbers = sorted(df['tail_number'].unique().tolist())
            tail_number = st.selectbox("Aircraft Tail Number", tail_numbers)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'origin': [origin],
            'destination': [destination],
            'departure_time': [departure_time],
            'tail_number': [tail_number]
        })
        
        # Prepare features using the frequency maps from training
        X_pred, _ = prepare_features(
            pred_df,
            origin_freq_map=model_results['feature_maps']['origin'],
            dest_freq_map=model_results['feature_maps']['destination'],
            tail_freq_map=model_results['feature_maps']['tail_number'],
            training=False
        )
        
        # Make prediction
        if st.button("Predict Incident Probability"):
            with st.spinner("Calculating..."):
                # Get probability of incident
                incident_prob = model_results['model'].predict_proba(X_pred)[0, 1]
                
                # Display results
                st.subheader("Prediction Results")
                
                # Create a nicer visual display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if incident_prob < 0.25:
                        st.markdown(f"### 🟢 Low Risk")
                    elif incident_prob < 0.5:
                        st.markdown(f"### 🟡 Moderate Risk")
                    elif incident_prob < 0.75:
                        st.markdown(f"### 🟠 High Risk")
                    else:
                        st.markdown(f"### 🔴 Very High Risk")
                
                with col2:
                    # Create progress bar for risk visualization
                    st.progress(float(incident_prob))
                    st.markdown(f"**Incident Probability: {incident_prob:.1%}**")
                
                # Show flight details
                st.subheader("Flight Details")
                st.markdown(f"""
                - **Route:** {origin} → {destination}
                - **Departure Time:** {departure_time}
                - **Aircraft:** {tail_number}
                """)
                
                # Feature contribution
                st.subheader("Feature Contributions")
                
                # Here we could implement SHAP values for better explanations
                # But for simplicity, we'll just show the feature values relative to the frequency maps
                contribution_data = {
                    'Feature': ['Origin Airport', 'Destination Airport', 'Departure Time (Sin)', 
                               'Departure Time (Cos)', 'Tail Number'],
                    'Value': [
                        f"{origin} (freq: {model_results['feature_maps']['origin'].get(origin, 0):.3f})",
                        f"{destination} (freq: {model_results['feature_maps']['destination'].get(destination, 0):.3f})",
                        f"{X_pred['departure_time_sin'].values[0]:.3f}",
                        f"{X_pred['departure_time_cos'].values[0]:.3f}",
                        f"{tail_number} (freq: {model_results['feature_maps']['tail_number'].get(tail_number, 0):.3f})"
                    ],
                    'Importance': [
                        model_results['feature_importances']['origin_freq'],
                        model_results['feature_importances']['destination_freq'],
                        model_results['feature_importances']['departure_time_sin'],
                        model_results['feature_importances']['departure_time_cos'],
                        model_results['feature_importances']['tail_number_freq']
                    ]
                }
                
                contrib_df = pd.DataFrame(contribution_data)
                contrib_df['Weighted Contribution'] = contrib_df['Importance'] / contrib_df['Importance'].sum()
                
                # Display as a sortable table
                st.dataframe(contrib_df.sort_values('Importance', ascending=False))
    
    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Accuracy")
            st.markdown(f"""
            - **Training Accuracy:** {model_results['train_score']:.4f}
            - **Test Accuracy:** {model_results['test_score']:.4f}
            """)
            
            st.subheader("Classification Report")
            report_df = pd.DataFrame(model_results['classification_report']).transpose()
            st.dataframe(report_df)
        
        with col2:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(model_results['fpr'], model_results['tpr'], 
                   label=f'ROC Curve (AUC = {model_results["roc_auc"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(model_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        st.subheader("Feature Importance")
        sorted_importances = dict(sorted(model_results['feature_importances'].items(), 
                                         key=lambda x: x[1], reverse=True))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sorted_importances.keys(), sorted_importances.values())
        ax.set_xticklabels([k for k in sorted_importances.keys()], rotation=45, ha='right')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
    
    # Tab 3: Data Exploration
    with tab3:
        st.header("Data Exploration")
        
        st.subheader("Raw Data")
        st.dataframe(df)
        
        st.subheader("Incident Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x='incident', ax=ax)
        ax.set_title('Distribution of Flight Incidents')
        ax.set_xlabel('Incident (0 = No, 1 = Yes)')
        ax.set_ylabel('Count')
        
        # Add percentage labels
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.1,
                    '{:1.1f}%'.format(height/total*100),
                    ha="center") 
            
        st.pyplot(fig)
        
        st.subheader("Departure Time vs Incident")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='incident', y='departure_time', ax=ax)
        ax.set_title('Departure Time by Incident Status')
        ax.set_xlabel('Incident (0 = No, 1 = Yes)')
        ax.set_ylabel('Departure Time (24hr format)')
        st.pyplot(fig)
        
        # Add correlation of cyclical time features
        st.subheader("Correlation with Cyclical Time Features")
        temp_df = df.copy()
        temp_df = cyclical_encode(temp_df, 'departure_time', 2400)
        corr_data = temp_df[['departure_time_sin', 'departure_time_cos', 'incident']].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation with Cyclical Time Features')
        st.pyplot(fig)

if __name__ == "__main__":
    main()