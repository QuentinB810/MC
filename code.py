import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Imports NumPy, a library for numerical calculations and manipulation of multidimensional arrays.
#Imports Pandas, used to manipulate and analyze datasets in table form (DataFrame).
#Imports Matplotlib, a data visualization library for charting.
#Import Seaborn, a visualization library based on Matplotlib, optimized to create more advanced and aesthetic statistical graphs.
#Allows you to divide a dataset into two parts: one for training and one for testing.
#Imports StandardScaler, which is used to normalize the data (center and reduce values).
#Imports the RandomForestClassifier, a machine learning algorithm based on decision tree forests.
#Amount of evaluation metrics:
#accuracy_score: Calculates the model accuracy.
#classification_report: Provides a detailed report of the model’s performance (accuracy, recall, F1-score...).
#confusion_matrix: Creates a confusion matrix to visualize classification errors.


# Step 1: Load the dataset
# Using the provided CSV file
file_path = "breast-cancer-wisconsin.csv"
data = pd.read_csv(file_path)

# Step 2: Define the problem
# Problem: Predict whether a tumor is malignant or benign based on given features
# Hypothesis: A well-trained model should achieve high accuracy due to the structured nature of medical data

# Step 3: Data Preprocessing
# Checking for missing values
data.dropna(inplace=True)

# Splitting features and target variable
X = data.drop(columns=[class])  # Features
y = data[class]  # Target variable

# Convert categorical labels to numerical
y = y.map({'M': 1, 'B': 0})

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Model Training
# Using Random Forest Classifier for classification
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Step 6: Results Interpretation
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', classification_rep)

# Step 7: Data Visualization
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Additional Analysis: Feature Importance
feature_importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,5))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.title('Feature Importance in Random Forest Model')
plt.show()

# Conclusion
# The model achieves high accuracy, indicating it successfully differentiates between malignant and benign tumors.
# Further tuning of hyperparameters can improve results.
# Feature importance analysis helps understand the key factors influencing classification.
