import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('../input/train.csv', header=0)
col_y = ['Survived']
col_x = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
df = df[col_y + col_x].dropna(axis=0)

le2 = LabelEncoder()
le2.fit(df.Survived.unique())

le = LabelEncoder()
le.fit(df.Sex.unique())
df.Sex = le.transform(df.Sex)

X = df.drop('Survived', axis=1)
y = le2.transform(df['Survived'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

tree = DecisionTreeClassifier(criterion='entropy')
tree = tree.fit(X=X_train, y=y_train)
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
title = str(str(datetime.datetime.now().date()) + ' - Titanic Survival Decision tree - Accuracy: {}'.format(accuracy))

print('Titanic - Machine Learning from Disaster')
print(str(datetime.datetime.now().date()))
print()
print('Train size: {}'.format(X_train[['Pclass']].count(axis=0).values))
print('Test size: {}'.format(X_test[['Pclass']].count(axis=0).values))
print('Accuracy: {}'.format(accuracy))

label_names = ['0', '1']
graph_data = export_graphviz(tree, feature_names=col_x,
class_names=label_names, filled=True, rounded=True, out_file=None)
graph = graphviz.Source(graph_data)
graph.render('Titanic Decision Tree Classifier')
graph
print()
print('File "Titanic Decision Tree Classifier created')