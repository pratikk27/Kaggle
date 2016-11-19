from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv("train_cleaned.csv")
dataset = dataset.drop(['Cabin'], axis = 1)
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("test_clean.csv")
test = test.drop(['Cabin'], axis = 1)

rf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
#rf = RandomForestClassifier(n_estimators=100)
#rf = GaussianNB()
#rf = KNeighborsClassifier(n_neighbors = 3)
#rf = SVC()
rf.fit(train, target)
pred = rf.predict(test)
print(rf.score(train,target))

np.savetxt('submission_rand_forest.csv', np.c_[range(892,len(test)+892),pred], delimiter=',', header = 'PassengerId,Survived', comments = '', fmt='%d')
