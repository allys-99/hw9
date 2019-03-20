# hw9  class  9 homework wine classifier

# CEBD 1160 Intro to Big Data

This is the final project.

| Name | Date |
|:-------|:---------------|
|Allyson Ramkissoon | 20 March 2019|

-----
### Resources
Your repository should include the following:

- Python script for your analysis: clfwine.py
- Results figure/saved file: K3.png
- Dockerfile for your experiment: Dockerfile
- runtime-instructions in a file: RUNME.md

-----

-----

## Research Question

Given a wine sample containing the 13 attributes, can we classify the wine in either of c1, c2 or c3.

### Abstract

Wine experts go through a time consuming process to categorize wines.  Automating to some extent parts of the process would be cost effective.  For this exercise we use supervised learning classification to predict the class of the wine. The K Nearest Neighbour algorithm identifies the k number of observations that are the most proximate to the test sample



### Introduction

The wine data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wine: Barolo, Grignolino, Barbera.(ref 4)

1) Type (Levels: Barolo, Grignolino, Barbera)

2) Alcohol

3) Malic acid

4) Ash

5) Alcalinity of ash

6) Magnesium

7) Total phenols

8) Flavanoids

9) Nonflavanoid phenols

10) Proanthocyanins

11) Color intensity

12) Hue

13) OD280/OD315 of diluted wines

14) Proline

Usage
### Methods

The KNN algorithm was ideal for this dataset because a particular feature out of all the others was distinguishing the cultivators.  The attributes Flavinoids and Total Phenols were clearly in specific ranges for each cultivator.
Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

kNN Pseudocode:

Classify(X, Y, x) # X: training data, Y: class labels of X, x: unknown sample
For each x in the test set:
- Compute the distance between x and each observation in the train set.
- Sort the distances in ascending order and obtain the classes of the k-nearest neighbors.
- Using majority rule, assign x to the predicted class.



### Results

For the features, flavanoids and total phenols, the class distributions have different means.


![Alt text](./KNN3.png?raw=true "Wine dataset")

performance of estimator = 0.778


### Discussion
The three classes seem to naturally separate as low/mid/high alcohol distrubtions. But flavanoids and total phenols are the best indicators.
There are many classification algorithms to choose from.  As a next step, we could try out all the algorithms in scklearn and keep the model that has the highest cross-validation score.

### References


- https://www.makeareadme.com/
- http://building-babylon.net/2017/01/17/wine-dataset-demonstrates-importance-of-feature-scaling/
- https://medium.com/nyu-a3sr-data-science-team/k-nn-with-red-wines-quality-in-r-bd55dcba4fd7
- https://rdrr.io/github/marchtaylor/sinkr/man/wine.html
- https://jonathonbechtel.com/blog/2018/02/06/wines/


-------

