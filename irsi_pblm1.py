

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.svm import SVC
    from sklearn import model_selection
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    
#names = [ 'sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv("D:\karthick\study\ML\MachineLearning\\try\Iris.csv")
print(data.shape)
print(data.head)
print(data.tail())
print(data.describe())
print(data.groupby('Species').size())
#pd.show_versions()
#data.plot(kind='box', columns=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
#plt.show()
#data.hist()
#plt.show()
data=data.drop(columns=['Id'])
data.hist()
scatter_matrix(data)
#plt.show()
#plt.close()
array = data.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#print(X_train)
#print(X_validation)
#print(Y_train)
#print(Y_validation)

scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM',SVC()))
#print(models)
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)



fig = plt.figure()
fig.suptitle('Algorithm Compatison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.draw()
plt.show(block=False)


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions)) 
      


    

