import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import random
import math


def xrange(x,y):

    return iter(range(x,y))


def get_feature_vector():  # first 10 are summer, next 10 are winter
    N = 20;
    x_vec = [];
    for i in range(N):
        if i < 10:
            im = misc.imread("summer{}.jpeg".format(i+1));
        else:
            im = misc.imread("winter{}.jpeg".format((i-10)+1));
        x_vec.append((np.sum(im[:, :, 0])/im.size, np.sum(im[:, :, 1])/im.size));
    return x_vec


def k_means(centroids, n_iter, dataset):  # returns Nx1 array of cluster indices, where dataset is N x d numpy array
    for ctr in range(n_iter+1):  # do _iter times
        # *********** cluster assignment
        cluster_idx0 = [];
        cluster_idx1 = [];
        sum_c0 = np.asarray((0,0));
        num_el_c0 = 0;
        sum_c1 = np.asarray((0,0));
        num_el_c1 = 0;

        k = 0;
        for datapoint in dataset:  # for every data point in dataset

            d0 = np.linalg.norm(datapoint - centroids[0])
            d1 = np.linalg.norm(datapoint - centroids[1])

            if d0 < d1:
                cluster_idx0.append(k);
                np.add(sum_c0, datapoint, out=sum_c0,
                       casting="unsafe"); # do a running total of the feature values of the datapoints in cluster 0
                num_el_c0 += 1;  # count the number of datapoints in cluster 0
            else:
                cluster_idx1.append(k);
                np.add(sum_c1, datapoint, out=sum_c1,
                       casting="unsafe");  # do a running total of the feature values of the datapoints in cluster 1
                num_el_c1 += 1;  # count the number of datapoints in cluster 1
            k+=1

        # *********** cluster movement
        new_centroid0 = centroids[0]
        new_centroid1 = centroids[1]
        if not(num_el_c0 == 0):
            new_centroid0 = np.asarray(sum_c0)/num_el_c0
        if not(num_el_c1 == 0):
            new_centroid1 = np.asarray(sum_c1)/num_el_c1
        centroids=np.asarray((new_centroid0, new_centroid1));

    return cluster_idx0, cluster_idx1


def get_purity(cluster_idx0, cluster_idx1):
    # calculate param as ratio of number of data points belonging to both cluster 1 and the season's subset, to
    # the total number of data points belonging to the season
    num_s = 0
    num_w = 0
    #print("Cluster 0:" + str(cluster_idx0))
    for ele in cluster_idx0:
        if ele <= 9:
            num_s+=1; #found a data point belonging to cluster 0 *and* summer subset
        else:
            num_w+=1; #found a data point belonging to cluster 0 *and* winter subset
    param_s = num_s / 10; #10 is the number of datapoints in summer subset
    #print("param_s:" + str(param_s))
    purity_summer = 1 + param_s*math.log(param_s,2) + (1-param_s)*math.log(1-param_s);
    param_w = num_w / 10; #10 is the number of datapoints in summer subset
    #print("param_w:" + str(param_w))
    purity_winter = 1 + param_w*math.log(param_w,2) + (1-param_w)*math.log(1-param_w);

    purity_average = 0.5*(purity_summer + purity_winter)
    return purity_average


if __name__ == '__main__':
    iter_coll = range(1,200);
    purity_average_coll = [];

    # read features
    X = get_feature_vector();
    X_ = np.asarray(X);

    for iter in iter_coll:

        purity_acc = [];
        n_reps = 10;

        for j in range(0, n_reps):

            #select initial centroids randomly from dataset
            random.seed();
            centroid1 = (X_[random.randint(0,10)]);
            random.seed();
            centroid2 = (X_[random.randint(10,19)]);
            centroid_coll = np.asarray((centroid1, centroid2));
            #print("Centroids Init: " + str(centroid_coll));

            #run k-mean-sq
            c_idx0, c_idx1 = k_means(centroid_coll, iter, X_)

            #get purity
            purity = get_purity(c_idx0, c_idx1);
            purity_acc.append(purity);

        purity_average = np.mean(purity_acc);
        purity_average_coll.append(purity_average);

    fig = plt.figure();
    fig.suptitle('Plot of Average Purity vs Number of Iterations of K-Means Algorithm', fontsize=12);
    plt.xlabel('M', fontsize=12)
    plt.ylabel('P-bar', fontsize=14);
    plt.plot(iter_coll,purity_average_coll, color="green");
    plt.show();