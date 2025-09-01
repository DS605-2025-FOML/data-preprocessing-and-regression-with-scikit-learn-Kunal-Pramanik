from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import time

# load race results
results = pd.read_csv('results.csv')
# feature: grid position, constructor, driver
numeric_features = ['grid']
categorical_features = ['constructorId', 'driverId']
X = results[numeric_features + categorical_features]
y = (results['positionOrder'] <= 3).astype(int)  # 1 = podium finish

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()
# RF with grid search
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None]
}

clf = GridSearchCV(pipe, param_grid, cv=5)

# ⏱ Timing the fit

clf.fit(X_train, y_train)
end_time = time.time()

y_pred = clf.predict(X_test)

print("Best params:", clf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"GridSearchCV took {end_time - start_time:.2f} seconds")

# Make predictions with the best trained model
sample_input = pd.DataFrame({'grid': [6], 'constructorId': [6], 'driverId': [22]})
raw_prediction = clf.predict(sample_input)[0]
predicted_proba = clf.predict_proba(sample_input)[0][1]

print("Raw prediction (0=No podium, 1=Podium):", raw_prediction)
print("Predicted probability of podium finish:", predicted_proba)

import joblib

# Save the trained model
joblib.dump(clf, "f1_podium_model.pkl")