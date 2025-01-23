import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load processed data
X_test = pd.read_csv('../data/X_test.csv')
y_test = pd.read_csv('../data/y_test.csv').squeeze()

# Load model
model = joblib.load('../models/random_forest_model.pkl')

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
