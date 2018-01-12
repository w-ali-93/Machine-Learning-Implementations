import scipy as scipy
from scipy import misc
import numpy as np
import matplotlib as matplotlib
import matplotlib.patches as mpatches
from matplotlib import pyplot as mpl
from matplotlib.lines import Line2D
from PIL import Image

X = []
Y = [[211],
     [271],
     [121],
     [31],
     [341],
     [401],
     [241],
     [181],
     [301],
     [301]]


path = "C:\\Users\\Wajahat\\OneDrive\\Work\\Aalto\\Year 1\\Period I\\CS-E3210 - Machine Learning - Basic Principles\\Homework\\Homework 2\\Problem 4\\Webcam\\"
files = ["MontBlanc1.png", "MontBlanc2.png", "MontBlanc3.png", "MontBlanc4.png", "MontBlanc5.png", "MontBlanc6.png", "MontBlanc7.png", "MontBlanc8.png", "MontBlanc9.png", "MontBlanc10.png"]
X = []
for item in files:
    w, h = Image.open(path+item).size
    img = Image.open(path + item)
    imgc = img.crop((0, 0, 100, 100))
    data = np.asarray(imgc)
    xg = []
    for i in range(0,100):
        for j in range(0, 100):
            xg.append(data[i, j, 1]/255)
    X.append(xg)
X = np.asarray(X)
mean = np.mean(X)
std = np.std(X)
X = (X - mean)*(1/std)
print(X)
Xt = np.transpose(X)
Y = np.asarray(Y)
XtX = np.matmul(Xt,X)
XtY = np.matmul(Xt,Y)

alpha_coll = [0.000005, 0.00001, 0.000135] #0.0001;
lambda_coll = [2, 5]

for alpha in alpha_coll:
    for penalty in lambda_coll:
        err_coll = []
        steps = []
        step = 0
        error = 99999999999999999
        w = np.zeros((10000, 1))
        print("alpha = " +str(alpha))
        while error > 1:
            step += 1
            gradient = (1/10)*(2*np.matmul(XtX, w) - 2*XtY + 2*penalty*w)
            w = w - alpha*gradient
            Y_ = np.matmul(X, w)
            err_prev = error
            error = (1/10)*np.matmul(np.transpose(Y_ - Y), (Y_ - Y))
            print(error)
            if step > 10000: break
            err_coll.append(error)
            steps.append(step)
        err_coll = np.asarray(err_coll).flatten()
        steps = np.asarray(steps).flatten()
        mpl.plot(steps, err_coll, label="alpha= " + str(alpha) + "; lambda= " + str(penalty) +"; final error= " + str(error[0][0]))

mpl.xlabel("k")
mpl.ylabel("f(w)")
mpl.legend(loc = 'best')
mpl.show()

"""Terminating conditions for regression loop
Specify a lower limit for the error, when this limit is met, terminate the loop
Specify the maximum number of steps that can be taken, when this limit is met, terminate the loop
Terminate the loop if current error becomes greater than the previous error (divergence)
"""