import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

# read in the dataset
df = pd.read_csv("govt.csv")

df.dropna(subset=['Employment Status'], inplace=True)

X = df.drop(['Employment Status'], axis=1)
y = df['Employment Status']

#transform
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# fit the model and make predictions
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("---------------------------------------------------")
print("Accuracy of KNN model:", accuracy)
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