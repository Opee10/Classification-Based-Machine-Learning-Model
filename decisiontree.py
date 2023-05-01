import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, recall_score,precision_score

# Load the data
data = pd.read_csv("govt.csv")

X = data.drop("Employment Status", axis=1)
y = data["Employment Status"]

# Tranform categorical data
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

# Create a decision tree classifier
dtc = DecisionTreeClassifier()

# Train the classifier on the training data
dtc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dtc.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("---------------------------------------------------")
print("Accuracy of Decision tree model:", accuracy)
print("---------------------------------------------------")

f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score:', f1)
print("---------------------------------------------------")

recall = recall_score(y_test, y_pred, average='weighted')
print('Recall score:', recall)
print("---------------------------------------------------")

precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
print('Precision score:', precision)
print("---------------------------------------------------")