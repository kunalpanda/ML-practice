import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model  #Line of best fit
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";") #read the data set

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "traveltime", "freetime", "goout", "Dalc"]] #keep the follwoing attributes, drop the rest

predict = "G3" #label we want to predit

x = np.array(data.drop([predict], 1)) #the data set on the x axis without prediction var (all features)
y = np.array(data[predict]) #prediction variable in the y axis (all labels)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

#training algo to find the model with best accuracy
best = 0
'''
training_cycles = 10000
for _ in range(training_cycles):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)
    """split 10%(0.1) of the data and assign it to the test, assign the rest of train"""

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train) #put a line of best fit

    acc = linear.score(x_test, y_test) #check the accracy of the model


    if acc > best:
        best = acc
        print ("Accuracy: ", best)
        with open("studentmodel.pickle", "wb") as f: #writting a pickle file to save model
            pickle.dump(linear, f)
#end of training algo
'''
pickle_in = open("studentmodel.pickle", "rb") #open the saved model in the pickle file

linear = pickle.load(pickle_in) #set saved model as the operating one

print("Co: \n",  linear.coef_)
print("Inter: \n",  linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print("Prediction: ", round(prediction[x]), "input: ", x_test[x], "true value: ", y_test[x])

p = "traveltime" #x var
style.use("ggplot") #plot styling
pyplot.scatter(data[p], data["G3"]) #make a scatter plot with given x and y var
pyplot.xlabel(p) #axis labels
pyplot.ylabel("Final grade")
pyplot.show() #show plot