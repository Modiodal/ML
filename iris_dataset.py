import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import andrews_curves
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


#Step 1 - Import data from UCI Machine Learning Repository website, set column names according to UCI website
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class'])

#Visualizing data using Andrews Curves
plt.figure(figsize=(15, 10))
andrews_curves(df, 'Class', colormap='rainbow')
plt.title('Andrews Curves - Iris Flowers', fontsize=25)
plt.legend(loc=1, prop={'size': 20}, frameon=True, shadow=True, edgecolor='black')
plt.show()

#Step 2 - Set X to values regarding to length and width ...... and Set y to the types of iris flower
X = df.drop('Class', axis=1)
y = df['Class']

#Step 3 - Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Step 4 - Build the model and fit to data, finalize by predicting the labels
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)

#Step 5 - Evaluate predictions using Accuracy Score, Confusion Matrix and Classification reports
print('The Accuracy of this model is: {}%' .format(np.round(accuracy_score(y_test, y_predict), decimals=4) * 100))
print('-------------------------------------')
print('Confusion Matrix: \n{}'.format(confusion_matrix(y_test, y_predict)))
print('-------------------------------------')
print('Classification report: \n{}'.format(classification_report(y_test, y_predict)))

#Step 6 - Create k's for knn model (tuning), calculate cross validation scores and append to scores
k_list = np.arange(1, 30, 3)
scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    cross_score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    scores.append(np.round(1 - cross_score.mean(), decimals=5))

#Visualize 'scores' list alongside the Misclassification error
plt.figure(figsize=(15, 10))
plt.title('The optimal number of neighbors', fontsize=25)
plt.xlabel('Number of Neighbors (k)', fontsize=17)
plt.ylabel('Misclassification Error', fontsize=17)
plt.plot(k_list, scores)
plt.show()

#Step 7 - Finding the best number of neighbors(k) for the model
best_num = k_list[scores.index(min(scores))]
print('-------------------------------------')
print('The best number of neighbors for our model is: {}'.format(best_num))
