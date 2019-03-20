#!/usr/bin/env python

import os
import re
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import iplot, init_notebook_mode

import pandas as pd
import numpy as np
np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

from sklearn.datasets import load_wine
from sklearn import neighbors, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib import cm as cmap

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
X= wine.data
y = wine.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,random_state=0)
clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
y_pred = clf.predict(X_test)


def KNN_PredictWine():
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    print("\nconfusion_matrix")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("\nclassificatio_report")
    print(metrics.classification_report(y_test, y_pred))
    print("\nf1_score = {:.3f}".format(metrics.f1_score(y_test, y_pred, average="macro")))
    print("\n")
    expected = y_test
    matches = (y_pred == expected)
    print("performance of estimator = {:.3f}".format(matches.sum() / float(len(matches))))
    print("\n")
    
def Plot_Prediction(num_neigh):
    X = wine.data[:,5:7]
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    knn = neighbors.KNeighborsClassifier(n_neighbors=num_neigh)
    knn.fit(X, y)
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                        np.linspace(y_min, y_max, 10))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    expected = y_test
    xvals = np.linspace(0, 1, len(y_pred))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    custom_lines = [Line2D([0], [0], color='#FF0000', lw=4),
                    Line2D([0], [0], color='#00FF00', lw=4),
                    Line2D([0], [0], color='#0000FF', lw=4)]
    plt.legend(custom_lines, ['c1','c2','c3'])
    plt.title("Wine Classifier, K="+str(num_neigh))
    plt.xlabel('Total Phenols')
    plt.ylabel('Flavanoids')
    plt.axis('tight')
    #plt.show()
    fname="KNN"+str(num_neigh)+".png"
    print(fname)
    plt.savefig(fname)




def Wine():
    print("\n* * * * * * *   K-NEAREST NEIGHBOUR   * * * * * * * * *\n")
    KNN_PredictWine()
    from sklearn.model_selection import GridSearchCV
    knn = neighbors.KNeighborsClassifier()
    params = {'n_neighbors':[2,3,4,5,6,7,10,15,20,50]}
    model = GridSearchCV(knn, params, cv=5)
    model.fit(X_train,y_train)
    print("    Best value for K is ....:  ",model.best_params_)
    #Plot_Prediction(1)
    #Plot_Prediction(2)
    Plot_Prediction(3)
    #Plot_Prediction(4)
    #Plot_Prediction(5)
    #Plot_Prediction_K3()
    print("\n* * * * * * * * * * * ** * * *")
    print("*                            * ")
    print("*        THE END             * ")
    print("*                            * ")
    print("* * * * * * * * ** * * * * * *\n\n")



Wine()


