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
    X.append([(np.sum(img[:,:,1]))/pixel_count(path+item),1])

X = np.array(X)
xg = X[:, 0]

"""mean = np.mean(xg)
std = np.std(xg)
print(X)
X[:, 0] = (X[:, 0]-mean)*(1/std)
print(X)"""
lambda1 = 2
lambda2 = 5
w1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X) + lambda1),np.transpose(X)),Y)
w2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X) + lambda2),np.transpose(X)),Y)

Y1_ = np.matmul(X,w1)#w1[0]*xg + w1[1]
Y2_ = np.matmul(X,w2)#w2[0]*xg + w2[1]

mpl.plot(xg,np.array(Y), 'go')
mpl.plot(xg,np.array(Y1_))
mpl.plot(xg,np.array(Y2_))
mpl.xlabel("x_g")
mpl.ylabel("y")
mpl.legend(['Actual y', 'Regression y with lambda=2', 'Regression y with lambda=5','.'])
print(np.asarray(w1))
print(np.asarray(w2))
mpl.show()

