import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path

# Load training data
df = pd.read_csv('training_data.csv')

features = ['age', 'months_since_joined', 'join_month', 'visits_per_week', 'payment_delay_days']
X = df[features]
y = df['churned']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train classifier
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Model Evaluation ===")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

print("\n=== Feature Effects (abs standardized coefficients) ===")
coefs = model.named_steps['clf'].coef_[0]
for feat, imp in sorted(zip(features, np.abs(coefs)), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

# Save model
joblib.dump(model, 'churn_model.pkl')
print("\n✅ Model saved to churn_model.pkl")