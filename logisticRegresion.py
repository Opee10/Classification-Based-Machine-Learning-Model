import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score, recall_score,precision_score
import numpy as np

# Load the dataset
df = pd.read_csv("govt.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Split the dataset into features and target
X = df.drop("Employment Status", axis=1)
y = df["Employment Status"]

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# Create lr model
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("---------------------------------------------------------")
print("Accuracy of Logistic Regression model:", accuracy)
print("---------------------------------------------------------")

f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score:', f1)
print("---------------------------------------------------------")

recall = recall_score(y_test, y_pred, average='weighted')
print('Recall score:', recall)
print("---------------------------------------------------------")

precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
print('Precision score:', precision)
print("---------------------------------------------------------")