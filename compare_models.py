#Aidan Martin CAP5771 Assignment 4 due 11/23/22
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

data=pd.read_csv('creditcard.csv')

Total_transactions = len(data)
normal = len(data[data.Class == 0])
fraudulent = len(data[data.Class == 1])
fraud_percentage = round(fraudulent/normal*100, 2)

print("Total transactions: "+str(Total_transactions))
print("Successful transactions: "+str(normal))
print("Fraudulent transactions: "+str(fraudulent))
print("Percent fraud: "+str(fraud_percentage))

sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

data.drop(['Time'], axis=1, inplace=True)
data.drop_duplicates(inplace=True)
X = data.drop('Class', axis = 1).values
y = data['Class'].values

def test(classifier, data, split):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=split, random_state = 1)
    classifier.fit(X_train, y_train)
    return f1_score(classifier.predict(X_test), y_test)

models={"Decision tree": DecisionTreeClassifier(max_depth=4, criterion='entropy'),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "LR": LogisticRegression(), #linear regression, naive bayes, linear discriminant
        "SVM": SVC(),
        "RF": RandomForestClassifier(max_depth = 4),
        "XGB": XGBClassifier(max_depth = 4)
        }

results=[]
for m in models:
    print("Running "+m, end="")
    data=[X,y]
    results.append([test(models[m],data,0.3),m])
    print(", F-score: "+str(results[-1][0]))

results.sort(key=lambda x: x[0], reverse=True)
print("\n RANKINGS \n")
for m in results:
    print(m[1]+" F-score: "+str(m[0]))
