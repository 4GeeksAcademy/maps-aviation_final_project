import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load your processed data
df = pd.read_csv("data/processed/combined_data.csv")

# Preprocessing
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
df['departure_time'] = pd.to_numeric(df['departure_time'], errors='coerce')

# Build frequency encodings
origin_freq_map = df['origin'].value_counts(normalize=True).to_dict()
destination_freq_map = df['destination'].value_counts(normalize=True).to_dict()
tail_freq_map = df['tail_number'].value_counts(normalize=True).to_dict()

def cyclical_encode(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def prepare_features(df):
    df = cyclical_encode(df, 'departure_time', 2400)
    df['origin_freq'] = df['origin'].map(origin_freq_map).fillna(min(origin_freq_map.values()))
    df['destination_freq'] = df['destination'].map(destination_freq_map).fillna(min(destination_freq_map.values()))
    df['tail_number_freq'] = df['tail_number'].map(tail_freq_map).fillna(min(tail_freq_map.values()))
    return df[['departure_time_sin', 'departure_time_cos', 'origin_freq', 'destination_freq', 'tail_number_freq']]

# Prepare features and target
X = prepare_features(df)
y = df['incident']  # assuming binary 0/1 label

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train the model
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# Metrics
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
feature_importances = dict(zip(X.columns, model.feature_importances_))

# Save model
joblib.dump(model, "models/model.pkl")

# Save feature maps
feature_maps = {
    'origin': origin_freq_map,
    'destination': destination_freq_map,
    'tail_number': tail_freq_map
}
joblib.dump(feature_maps, "models/feature_maps.pkl")

# Save results
results = {
    'train_score': train_score,
    'test_score': test_score,
    'classification_report': report,
    'confusion_matrix': cm,
    'fpr': fpr,
    'tpr': tpr,
    'roc_auc': roc_auc,
    'feature_importances': feature_importances
}
joblib.dump(results, "models/results.pkl")

print("✅ Model, feature maps, and results saved successfully.")