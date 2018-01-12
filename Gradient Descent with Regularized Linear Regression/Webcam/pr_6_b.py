from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat;
from numpy.linalg import inv
from scipy import misc

""" GET THE Left 100x100 Green intensity Pixels and stack them int the vector """
def get_feature_vector():
    x = np.zeros((10,100*100));
    width = 100; height=100;

    for i in range(N): #FOR ALL IMAGES
        im = Image.open("MontBlanc{}.png".format(i+1));
        rgb_im = im.convert('RGB'); # Converts the image into RGB
        count = 0;

        for row in range(width):
            for col in range(height):
                r,g,b = rgb_im.getpixel((row,col));
                x[i,count] = g/(255);
                count +=1;

    """ standardized the values"""
    _mean = np.mean(x);
    _std_dev = np.std(x);
    x = (x - _mean)/_std_dev;
    return x;

""" Gradient Descent Algorithm"""
"""
Equation to find is:
Gradw = 2/N * [xTxw * x - xTy];
which will give Gradw with dimensions: 1x10000
w = 10000x1;
X = 10x10000;
y = 10x1"""

""" gradient_descent(feature_vector,trainng_data,error_margin)"""
def gradient_descent(x,y,a,error_margin):
    ER = [];
    a = a;
    N = 10;
    w = np.zeros((10**4,1));
    xT = np.transpose(x);
    xTx = np.dot(xT,x);
    xTy = np.dot(xT,y);
    epsilon = error_margin;
    e = 9999999999;
    _iter = 0;
    #error_margin = 10*3;
    _lambda = 5;

    for i in range(6000):
        """ Calculate Gradw """
        Gradw = (2/10)*(np.dot(xTx,w) - xTy) + 2*_lambda*w;
        w = w - a*Gradw;
        """ CALCULATE EMPIRICAL RISK = 1/N [y(i) - w^T x(i)]^2"""
        xw_y = np.dot(x,w) - y;
        e = (1/N)*np.dot(np.transpose(xw_y),xw_y);
        print (e[0,0]);
        ER.append(e[0,0]);
    return ER;

if __name__ == '__main__':

    """ Original Vector Dimensions
    x = [10x10^4];
    w = [10000x1];
    y = [10x1];
    """
    ER = [];
    y = np.array([[211],[271],[121],[31],[341],[401],[241],[181],[301],[301]]);
    N = 10; # sample size
    x = get_feature_vector();
    k = 10; # ITERATIONS...
    a = [10**-4,7*10**-5];

    ER1 = gradient_descent(x,y,a[0],6000);
    ER2 = gradient_descent(x,y,a[1],6000);

    x_range = list(range(6000));

    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Empirical Risk', fontsize=14);

    plt.plot(x_range,ER1,label='lambda = 5, sigma=10^-4');
    plt.plot(x_range,ER2,label='lambda = 5, sigma=7x10^-5');

    plt.legend(loc = 'best');
    plt.show();
