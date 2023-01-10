'''
Used for classification of datasets
'''

import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import preprocessing, linear_model

data = pd.read_csv("car.data");
print(data.head())

le = preprocessing.LabelEncoder() #object for converting string labels to int values

#auto convert all string values into int values

buying = le.fit_transform(list(data["buying"])) #each one is a list of int values
maint = le.fit_transform(list(data["maint"])) #contains an entire column
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

x = list(zip(buying, maint, door, persons, lug_boot, safety))
#zip makes all elements of arrays into tupples
y = list(cls)

predict = "class"

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

num_neigh = 7

model = KNeighborsClassifier(n_neighbors=num_neigh)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], num_neigh, True)
    print ("N: ", n)