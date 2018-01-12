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

def pixel_count(filepath):
    width, height = Image.open(filepath).size
    return width*height

path = "C:\\Users\\Wajahat\\OneDrive\\Work\\Aalto\\Year 1\\Period I\\CS-E3210 - Machine Learning - Basic Principles\\Homework\\Homework 2\\Problem 4\\Webcam\\"
files = ["MontBlanc1.png", "MontBlanc2.png", "MontBlanc3.png", "MontBlanc4.png", "MontBlanc5.png", "MontBlanc6.png", "MontBlanc7.png", "MontBlanc8.png", "MontBlanc9.png", "MontBlanc10.png"]

for item in files:
    img = misc.imread(path + item)
    X.append([(np.sum(img[:, :, 1]))* (1 / pixel_count(path + item))])
X = np.asarray(X)
Y = np.asarray(Y)
Y_ = np.zeros((10,1))

mpl.plot(X, Y,'g.', label = "Original y")
color = ['r', 'b', 'y']
symbol = ['.', '.', '.']

sigma_coll = [1, 5, 10]
MSE = []
j=0
for sigma in sigma_coll:
    beta = 1/(2*(sigma^2))
    for i in range(0, 10):
        K = np.squeeze(np.asarray(np.exp(-beta*(np.square((X-X[i]))))))
        Y_[i] = np.sum(np.dot(K, Y))/np.sum(K)
    MSE.append((1/10)*(np.matmul(np.transpose(Y_-Y),(Y_-Y))))
    mpl.plot(X, Y_, str(color[j])+str(symbol[j]), label = "Regression y with sigma = " + str(sigma))
    j+=1
MSE = np.asarray(MSE).flatten().transpose()
sigma_coll = np.asarray(sigma_coll).flatten().transpose()
print("MSE for each sigma is:")
print(np.column_stack((sigma_coll, MSE)))
print("From this, the optimum choice of sigma is: " + str(sigma_coll[np.argmin(MSE)]))

mpl.xlabel("x_g")
mpl.ylabel("y")
mpl.legend(loc = 'best')
mpl.show()