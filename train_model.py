import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))
    print(f"ROC-AUC Score: {auc:.3f}")

    return auc


def print_logistic_effects(model, features):
    print("\n=== Logistic Regression Feature Effects (abs standardized coefficients) ===")
    coefs = model.named_steps['clf'].coef_[0]
    for feat, imp in sorted(zip(features, np.abs(coefs)), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")


def print_tree_importances(name, model, features):
    print(f"\n=== {name} Feature Importances ===")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")


def find_training_data():
    candidates = [
        Path('training_data.csv'),
        Path('GenDataset') / 'training_data.csv',
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find training_data.csv in the project root or GenDataset folder."
    )


# Load training data
training_data_path = find_training_data()
print(f"Loading training data from {training_data_path}")
df = pd.read_csv(training_data_path)

features = ['age', 'months_since_joined', 'join_month', 'visits_per_week', 'payment_delay_days']
X = df[features]
y = df['churned']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train logistic regression baseline
logistic_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
logistic_model.fit(X_train, y_train)

# Train random forest model
random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
random_forest_model.fit(X_train, y_train)

# Train gradient boosting model
gradient_boosting_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gradient_boosting_model.fit(X_train, y_train)

# Evaluate
model_scores = {}

model_scores['logistic'] = evaluate_model("Logistic Regression", logistic_model, X_test, y_test)
print_logistic_effects(logistic_model, features)

model_scores['random_forest'] = evaluate_model("Random Forest", random_forest_model, X_test, y_test)
print_tree_importances("Random Forest", random_forest_model, features)

model_scores['gradient_boosting'] = evaluate_model("Gradient Boosting", gradient_boosting_model, X_test, y_test)
print_tree_importances("Gradient Boosting", gradient_boosting_model, features)

# Save models
models = {
    'logistic': logistic_model,
    'random_forest': random_forest_model,
    'gradient_boosting': gradient_boosting_model,
}

model_files = {
    'logistic': 'churn_logistic_model.pkl',
    'random_forest': 'churn_random_forest_model.pkl',
    'gradient_boosting': 'churn_gradient_boosting_model.pkl',
}

for model_name, model in models.items():
    joblib.dump(model, model_files[model_name])

best_model_name = max(model_scores, key=model_scores.get)

# Keep app.py working with the newest default model.
joblib.dump(models[best_model_name], 'churn_model.pkl')

print("\nModels saved:")
print("  churn_logistic_model.pkl")
print("  churn_random_forest_model.pkl")
print("  churn_gradient_boosting_model.pkl")
print(f"  churn_model.pkl (default: {best_model_name})")
