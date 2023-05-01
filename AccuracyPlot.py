import matplotlib.pyplot as plt

models = ['SVM', 'Logistic Regression', 'Decision Tree', 'Naive Bayes', 'KNN']
accuracies = [0.9685658153241651, 0.9724950884086444, 0.9567779960707269, 0.9292730844793713, 0.9587426326129665]

colors =['orange', 'green', 'red', 'navy', 'purple']

plt.bar(models, accuracies, color=colors)
plt.title('Accuracy Scores of Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.88, 1.0)
plt.show()
