import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import stats
import random
from itertools import chain
import math
import sys


def get_feature_vector():  # first 10 are summer, next 10 are winter
    N = 20;
    x_vec = [];
    for i in range(N):
        if i < 10:
            im = misc.imread("summer{}.jpeg".format(i+1));
        else:
            im = misc.imread("winter{}.jpeg".format((i-10)+1));
        x_vec.append(((np.sum(im[:, :, 0])/im.size)/255, (np.sum(im[:, :, 1])/im.size)/255));
    return x_vec


def soft_cluster(m0_init, m1_init, C0_init, C1_init, n_iter, dataset):
    C0=C0_init
    C1=C1_init
    m0=m0_init
    m1=m1_init
    acc_purity = [];

    for ctr in range(n_iter+1):
        Y = [];
        c0_idx = [];
        c1_idx = [];
        x = 0;
        for datapoint in dataset:
            # Gaussian Normal Random Vector normally distributed about x(i) with mean m0 and covariance C0
            norm_xi_m0_C0 = stats.multivariate_normal.pdf((datapoint[0, 0], datapoint[0, 1]), (m0[0, 0], m0[0, 1]),C0);
            # Gaussian Normal Random Vector normally distributed about x(i) with mean m1 and covariance C1
            norm_xi_m1_C1 = stats.multivariate_normal.pdf((datapoint[0,0], datapoint[0,1]),(m1[0,0], m1[0,1]),C1);
            # Update degree of belonging
            y_i = norm_xi_m0_C0/(norm_xi_m0_C0+norm_xi_m1_C1);

            # Assign cluster to datapoint
            if y_i > (1 - y_i):
                c0_idx.append(x)
            else:
                c1_idx.append(x)
            Y.append(y_i)
            x+=1

        # Update GMM parameters:
        N0 = np.sum(Y);
        N1 = 20 - N0;

        # Update m1 and m0
        sum_m0_inner = (0, 0);
        sum_m1_inner = (0, 0);
        for k in range(20):
            sum_m0_inner+=Y[k]*dataset[k];
            sum_m1_inner+=(1-Y[k])*dataset[k];
        m0 = sum_m0_inner/N0;
        m1 = sum_m1_inner/N1;

        # Update C1 and C0
        sum_C0_inner = np.asmatrix([[0.0 for x in range(2)] for y in range(2)])
        sum_C1_inner = np.asmatrix([[0.0 for x in range(2)] for y in range(2)])

        for i in range(20):
            addend_C0=np.multiply(Y[i], np.transpose(dataset[i]-m0)*(dataset[i]-m0));
            np.add(sum_C0_inner, addend_C0, out=sum_C0_inner, casting="unsafe");
            addend_C1=np.multiply((1-Y[i]), np.transpose(dataset[i]-m1)*(dataset[i]-m1));
            np.add(sum_C1_inner, addend_C1, out=sum_C1_inner, casting="unsafe");
        C0 = sum_C0_inner/N0;
        C1 = sum_C1_inner/N1;

        #get purity
        purity = get_purity(Y, c1_idx, c0_idx);
        acc_purity.append(purity);

        #print("Iteration: " + str(ctr))
        #print("N1:" + str(N1) + ", N0: " + str(N0))
        #print("Y: " + str(np.ndarray.flatten(np.asarray(Y))))
        #print("c1_idx: "+ str(c1_idx))
        #print("c0_idx: " + str(c0_idx))
        #print("C1: " + str(C1) + ", m1: " + str(m1))
        #print("C0: " + str(C0) + ", m0: " + str(m0))
        #print("m1: " + str(m1))
        #print("m0: " + str(m0))
        #print("****************")
        #print("$$$$$$$$$$$$$$$$")
    return np.asarray(Y).flatten(), np.asarray(c0_idx).flatten(), np.asarray(c1_idx).flatten(), m0, C0, m1, C1, acc_purity

def get_purity(Y, c1_idx, c0_idx):
    sum_s = 0
    sum_w = 0

    for i in c0_idx:
        sum_s+=Y[i]
    for i in c1_idx:
        sum_w+=Y[i]

    param_s = sum_s * (2/20);
    if not(param_s>1):
        purity_summer = 1 + param_s*np.log2(param_s) + (1-param_s)*np.log2(1-param_s);
    else:
        purity_summer = 1;
    param_w = sum_w * (2/20);
    purity_winter = 1 + param_w*np.log2(param_w) + (1-param_w)*np.log2(1-param_w);
    purity_average = 0.5*(purity_summer + purity_winter)
    return purity_average


if __name__ == '__main__':
    iter_coll = range(1, 100);
    acc_acc_purity = [];
    purity_average_coll = [];

    # read features
    X = get_feature_vector();
    X_ = np.asmatrix(X);

    #print(str(X_));

    n_reps = 1

    # operational parameters
    use_random_every_time = 1;
    first_run = True;

    # set up blank containers for the means and covariances returned from previous run of soft clustering algorithm
    ret_m0 = np.asmatrix((0, 0))
    ret_m1 = np.asmatrix((0, 0))
    ret_C0 = np.asmatrix(np.identity(2));
    ret_C1 = np.asmatrix(np.identity(2));

    # select initial centroids randomly from dataset for the case where
    # random picking is done only *ONCE* at the start and *NOT* for
    # every value of M
    r1 = random.randint(0, 10)
    r2 = random.randint(10, 19)
    m0_i = (X_[r1]);
    m1_i = (X_[r2]);
    C0_i = np.asmatrix(np.identity(2));
    C1_i = np.asmatrix(np.identity(2));


    for iter in iter_coll:

        if use_random_every_time:
            #select initial centroids randomly from dataset at every run
            random.seed()
            r1 = random.randint(0,10)
            r2 = random.randint(10,19)
            m0_i = (X_[r1]);
            m1_i = (X_[r2]);
            C0_i = np.asmatrix(np.identity(2));
            C1_i = np.asmatrix(np.identity(2));
            #print("randoms:" + str(r1)+";"+str(r2))
        else:
            if not first_run:
                m0_i = ret_m0
                m1_i = ret_m1
                C0_i = ret_C0
                C1_i = ret_C1
        first_run = False;

        #run soft clustering
        Y, clust0_idx, clust1_idx, ret_m0, ret_C0, ret_m1, ret_C1, acc_purity = soft_cluster(m0_i, m1_i, C0_i, C1_i, iter, X_)
        #acc_acc_purity.append(acc_purity);
        acc_acc_purity = acc_acc_purity + acc_purity
        #print("accumulated purities:" + str(acc_acc_purity))
        purity_average_coll.append(np.mean(acc_acc_purity))

        #print("Cluster 1:" + str(clust1_idx))
        #print("Cluster 0:" + str(clust0_idx))

    #print(str(purity_average_coll));
    fig = plt.figure();
    fig.suptitle('Plot of Average Purity vs Number of Iterations of Soft Clustering Algorithm', fontsize=12);
    plt.xlabel('M', fontsize=12)
    plt.ylabel('P-bar', fontsize=14);
    plt.plot(iter_coll,purity_average_coll, color="green");
    plt.show();