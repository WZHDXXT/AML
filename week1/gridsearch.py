import numpy as np
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = sklearn.datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3)

C = [0.001,0.01,0.1,1,10,100]
gamma = [0.001,0.01,0.1,1,10,100]
best_score = 0
best_parameter = {}
for c in C:
    for g in gamma:
        clf = SVC(C=c, gamma=g)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        if score > best_score:
            best_score = score
            best_parameter['C'] = c
            best_parameter['gamma'] = g

print("best score is ",best_score)
print("best parameter C is ",best_parameter['C'])
print("best parameter gamma is ",best_parameter['gamma'])

