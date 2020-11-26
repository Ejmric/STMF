import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster.bicluster import SpectralCoclustering
import numpy.ma as ma
from operator import or_
import operator
import math
from numpy import genfromtxt
import random
from scipy.stats import pearsonr, spearmanr
import distance_correlation as dc
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering, SpectralBiclustering
from sklearn.metrics import consensus_score
import copy
from statistics import mean
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
#polo
from polo import optimal_leaf_ordering
from scipy.spatial.distance import pdist
from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list
from sklearn.metrics import mean_squared_error
np.random.seed(0)
random.seed(0)

def plot_silh(range_n_clusters, predictions, original_data, folder_name):
    cluster_labels = copy.deepcopy(predictions)
    X = copy.deepcopy(original_data)
    range_n_clusters = [range_n_clusters]
    subtypes = ["LumA", "LumB", "Basal", "Her2", "Normal"]
    for n_clusters in range_n_clusters:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(9, 5)

        ax1.set_xlim([-0.3, 0.3])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(1, n_clusters+1):
            #print(i)
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            #print(size_cluster_i)
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            #ax1.text(-0.1, y_lower + 0.5 * size_cluster_i, str(i))
            ax1.text(-0.2, y_lower + 0.5 * size_cluster_i, subtypes[i-1])

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        #ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_xticks([-0.3, -0.2, 0.1, 0.0, 0.1, 0.2, 0.3])
        
    plt.savefig("results/" + folder_name + "/" + "silh_plot.png")
    plt.show()

def scatter_hist(x, y, x_nmf, y_nmf, folder_name, x_title, y_title, corr_name, num_bins=20):#, ax, ax_histx, ax_histy):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    # the scatter plot:
    ax.scatter(x, y, c='b', marker='o', label='STMF')
    ax.scatter(x_nmf, y_nmf, c='r', marker='x', label='NMF')
    ax.legend(['STMF', 'NMF'], loc='lower left')
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    # now determine nice limits by hand:
    # Za X-os
    xmax = np.max([x, x_nmf])
    xmin = np.min([x, x_nmf])
    binwidth = (xmax - xmin) / num_bins
    binsx = np.arange(xmin, xmax + binwidth, binwidth)
    
    ax_histx.hist(x, bins=binsx, label='STMF', alpha=0.5) # diff stmf
    ax_histx.hist(x_nmf, bins=binsx, label='NMF', alpha=0.5, color="red") # diff nmf
    
    ax_histx.legend()
    # Za Y-os
    ymax = np.max([y, y_nmf])
    ymin = np.min([y, y_nmf])
    binwidth = (ymax - ymin) / num_bins
    binsy = np.arange(ymin, ymax + binwidth, binwidth)
    ax_histy.hist(y, bins=binsy, orientation='horizontal', label='STMF', alpha=0.5) # corr stmf
    ax_histy.hist(y_nmf, bins=binsy, orientation='horizontal', label='NMF', alpha=0.5, color="red") # corr nmf
    ax_histy.legend()
    fig.savefig("results/" + folder_name + "/" + corr_name + "_hist.png")
    
def scatter_hist_clusters(first_cluster, second_cluster, third_cluster, x, y, x_nmf, y_nmf, folder_name, x_title, y_title, corr_name, num_bins=20):#, ax, ax_histx, ax_histy):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    # the scatter plot:
    ax.scatter(x, y, color='blue', marker='o', label='STMF', alpha=0.5)
    #ax.scatter(x_nmf, y_nmf, c='r', marker='x', label='nmf')
    ax.scatter(x_nmf[first_cluster], y_nmf[first_cluster], color='red', marker='*', label='NMF', alpha=0.5)
    ax.scatter(x_nmf[second_cluster], y_nmf[second_cluster], color='red', marker='o', label='NMF', alpha=0.5)
    ax.scatter(x_nmf[third_cluster], y_nmf[third_cluster], color='red', marker='x', label='NMF', alpha=0.5)
    #ax.legend(['STMF', 'NMF'], loc='lower left')
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    # now determine nice limits by hand:
    # Za X-os
    xmax = np.max([x, x_nmf])
    xmin = np.min([x, x_nmf])
    binwidth = (xmax - xmin) / num_bins
    binsx = np.arange(xmin, xmax + binwidth, binwidth)
    
    ax_histx.hist(x, bins=binsx, label='STMF', alpha=0.5, color='blue') # diff stmf
    ax_histx.hist(x_nmf, bins=binsx, label='NMF', alpha=0.5, color="red") # diff nmf
    
    ax_histx.legend()
    # Za Y-os
    ymax = np.max([y, y_nmf])
    ymin = np.min([y, y_nmf])
    binwidth = (ymax - ymin) / num_bins
    binsy = np.arange(ymin, ymax + binwidth, binwidth)
    ax_histy.hist(y, bins=binsy, orientation='horizontal', label='STMF', alpha=0.5, color='blue') # corr stmf
    ax_histy.hist(y_nmf, bins=binsy, orientation='horizontal', label='NMF', alpha=0.5, color="red") # corr nmf
    ax_histy.legend()
    fig.savefig("results/" + folder_name + "/" + corr_name + "_clusters_hist.png")
    
def plot_diff_corr(distances, correlation, distances_nmf, correlation_nmf, folder_name, corr_name):
    # spearman
    plt.scatter(distances, correlation, c='b', marker='o', label='STMF')
    plt.scatter(distances_nmf, correlation_nmf, c='r', marker='x', label='NMF')
    plt.xlabel("difference")
    plt.ylabel(corr_name + " correlation")
    plt.legend(loc='upper right')
    plt.savefig("results/" + folder_name + "/" + corr_name + ".png")
    plt.show();

def get_column_indices(patients, pat):
    indices = []
    for element in pat:
        if element in patients:
            indices.append((element, patients.index(element)))
    return [x[1] for x in indices] #sorted(indices, key=lambda x: patients.index(x[0]))]


def check_gene_names(data):
    new_list = [i.split('|', 1)[0] for i in data]
    return new_list

def return_permData_colPerm(data):
    # col perm
    data = data.T  # transpose
    D = pdist(data, 'euclidean')  # distance
    Z = linkage(D, 'ward')
    optimal_Z = optimal_leaf_ordering(Z, D)
    opt_order_columns = leaves_list(optimal_Z)
    data = data[opt_order_columns]
    return data.T, opt_order_columns

def return_permData_rowPerm(data):
    # row perm
    D = pdist(data, 'euclidean')  # distance
    Z = linkage(D, 'ward')
    optimal_Z = optimal_leaf_ordering(Z, D)
    opt_order = leaves_list(optimal_Z)
    data = data[opt_order]
    return data, opt_order

def scale_matrix(A, NINF = -1000000):
    m = A.shape[0]  # rows
    n = A.shape[1]  # columns
    left_matrix = ma.masked_array(np.full((m, m), NINF, float), mask=np.zeros((m, m)))  # left_matrix = np.full((m, m), np.NINF)
    right_matrix = ma.masked_array(np.full((n, n), NINF, float), mask=np.zeros((n, n)))  # right_matrix = np.full((n, n), np.NINF)
    # left matrix
    max_rows = []
    for j in range(m):
        max_value = np.max(A[j, :])
        max_rows.append(max_value)
    np.fill_diagonal(left_matrix, np.negative(max_rows))
    A_left = max_plus(left_matrix, A)
    # right matrix
    max_columns = []
    for i in range(n):
        max_value = np.max(A_left[:, i])
        max_columns.append(max_value)
    np.fill_diagonal(right_matrix, np.negative(max_columns))
    A_new = max_plus(A_left, right_matrix)

    return A_new, left_matrix, right_matrix


def inverse_scaling(U_scaled, V_scaled, left_matrix, right_matrix):
    r = left_matrix.shape[1]
    n = right_matrix.shape[1]
    for i in range(r):
        left_matrix[i, i] *= -1
    for j in range(n):
        right_matrix[j, j] *= -1
    
    U = max_plus(left_matrix, U_scaled)
    V = max_plus(V_scaled, right_matrix)
    return U, V


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Permutation')
    
def set_axis_style_y(ax, labels):
    ax.get_yaxis().set_tick_params(direction='out')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_ylim(0.25, len(labels) + 0.75)
    ax.set_ylabel('ordering')
    
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violin_plot(data, labels, title):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), sharey=True)

    ax1.set_ylabel('Correlation')

    ax1.set_title(title)
    parts = ax1.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    set_axis_style(ax1, labels)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#3A8CD4')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax1.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

def violin_plot_all(data, labels, title, name):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), sharey=True)

    ax1.set_xlabel('distance correlation')

    ax1.set_title(title)
    parts = ax1.violinplot(data, showmeans=False, showmedians=True, showextrema=True, vert=False)
    
    set_axis_style_y(ax1, labels) 
    
    for pc in parts['bodies']:
        pc.set_facecolor('#3A8CD4')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax1.scatter(medians, inds, marker='o', color='white', s=30, zorder=3)
    ax1.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax1.hlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    
    fig.savefig(name, dpi=300, bbox_inches = 'tight')
    
    

def generate_matrix(n, type='tropical'):
    # n > 4, r = floor(n/2)
    r = int(np.floor(n/2))
    b = np.random.randint(4, 3+2*r*n)
    low = -b
    high = b
    U_matrix = low + (high - low) * np.random.rand(n, r)
    V_matrix = low + (high - low) * np.random.rand(r, n)
    if type=='tropical':
        A = max_plus(U_matrix, V_matrix)
    else: # type = subtropical
        A = max_times(U_matrix, V_matrix)
    return A


def gen_mat(m, n, low, high, r, type='tropical'):
    B = low + (high-low)*np.random.rand(m, r)  # shape (m, r)
    C = low + (high-low)*np.random.rand(r, n)  # shape (r, n)
    if type=='tropical':
        A = max_plus(B, C)  # shape (m, n)
    else: # type = subtropical
        A = max_times(B, C)  # shape (m, n)
    return A


def max_plus(B, W):
    """
        :param B: numpy ndarray
        :param W: numpy ndarray
        :return:
        output: (max,+) multiplication of matrices B and W
        """
    rows_B, columns_B, columns_W = B.shape[0], B.shape[1], W.shape[1]
    B_size = np.size(B)
    W_size = np.size(W)
    
    if B_size * W_size != 0:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
        for i in range(0, rows_B):
            x = ma.expand_dims(ma.transpose(B[i, :]), axis=1)
            product = ma.array(x.data+W.data,mask=list(map(or_,x.mask,W.mask)))
            output[i, :]=get_max(product)
    else:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
    return output


def min_plus(B, W):
    """
        :param B: numpy ndarray
        :param W: numpy ndarray
        :return:
            output: (min,+) multiplication of matrices B and W
    """
    rows_B, columns_B, columns_W = B.shape[0], B.shape[1], W.shape[1]
    B_size = np.size(B)
    W_size = np.size(W)

    if B_size * W_size != 0:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
        for i in range(0, rows_B):
            x = ma.expand_dims(ma.transpose(B[i, :]), axis=1)
            product = ma.array(x.data+W.data,mask=list(map(or_,x.mask,W.mask)))
            output[i, :]=get_min(product)
    else:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
    return output


def min_plus_2(B, W, param):
    """
        :param B: numpy ndarray
        :param W: numpy ndarray
        :return:
            output: (min,+) multiplication of matrices B and W
    """
    rows_B, columns_B, columns_W = B.shape[0], B.shape[1], W.shape[1]
    B_size = np.size(B)
    W_size = np.size(W)

    if B_size * W_size != 0:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
        for i in range(0, rows_B):
            x = ma.expand_dims(ma.transpose(B[i, :]), axis=1)
            product = ma.array(x.data+W.data,mask=list(map(or_,x.mask,W.mask)))
            output[i, :]=get_min_2(product, param)
    else:
        output = ma.masked_array(np.zeros((rows_B, columns_W)), mask=np.zeros((rows_B, columns_W)))
    return output



def square_root(A):
    return np.multiply(0.5, A)  # multiply => element-wise multiplication


def generate_matrices(m, n, r):
    # generate data matrix by multiplying two matrices from normal distribution
    B = np.random.normal(size=(m, r))  # shape (m, r)
    C = np.random.normal(size=(r, n))  # shape (r, n)
    A = max_plus(B, C)  # shape (m, n)
    return A


def generate_integer_matrix(m, n, low, high):
    A = np.random.random_integers(low=low, high=high, size=(m, n))
    return A


def generate_float_matrix(m, n):
    b = np.random.randint(5, 15)
    low = b
    high = 2*b
    A = low + (high-low)*np.random.rand(m, n)
    return A


def plot_original_model(original, model):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    a = ax[0].pcolor(original)  # original data matrix
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Original")

    b = ax[1].pcolor(model)  # our model
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Model")

    plt.show()


def plot_original_model_U_V(original, model, U, V):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))

    a = ax[0].pcolor(original)  # original data matrix
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Original")

    b = ax[1].pcolor(model)  # our model
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Model")

    c = ax[2].pcolor(U)  # matrix U
    fig.colorbar(b, ax=ax[2])
    ax[2].set_title("Matrix U")

    d = ax[3].pcolor(V)  # matrix V
    fig.colorbar(b, ax=ax[3])
    ax[3].set_title("Matrix V")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)

    # plt.axis('equal')  # try or 'scaled'
    plt.show()


def plot_errors(trop, nmf, title):
    m = 11
    dic_trop = trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    fig.suptitle(title)
    
    ax1.boxplot(data_trop)
    med_trop = np.median(data_trop, axis=1)
    ax1.plot(range(1, m), med_trop)
    ax1.set_xlabel('rank')
    ax1.set_ylabel('error')
    ax1.set_title("STMF")
    ax1.set_xticks(np.arange(0, m, 5))
    ax1.set_xticklabels(['1', '5', '10'])
    
    dic_nmf = nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    ax2.boxplot(data_nmf)
    med_nmf = np.median(data_nmf, axis=1)
    ax2.plot(range(1, m), med_nmf)
    ax2.set_xlabel('rank')
    ax2.set_title("NMF")
    ax2.set_xticks(np.arange(0, m, 5))
    ax2.set_xticklabels(['1', '5', '10'])
    
    ax3.plot(med_trop)
    ax3.plot(med_nmf)
    ax3.legend(['STMF', 'NMF'], loc='lower left')
    ax3.set_xlabel('rank')
    ax3.set_xticks(np.arange(0, m, 5))
    ax3.set_xticklabels(['1', '5', '10'])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.84);

def plote(trop, nmf, title):
    m = 11
    dic_trop = trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)

    fig, ax = plt.subplots()
    ax.plot(med_trop)
    ax.plot(med_nmf)
    ax.legend(['STMF', 'NMF'], loc='lower left')
    ax.set_ylabel("error")
    ax.set_xlabel('rank')
    ax.set_xticks(np.arange(0, m, 1))
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']);
    fig.savefig('err.png', bbox_inches = 'tight', dpi=300);
    
def plotc(trop, nmf, title):
    m = 11
    dic_trop = trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)

    fig, ax = plt.subplots()
    ax.plot(med_trop)
    ax.plot(med_nmf)
    ax.legend(['STMF', 'NMF'], loc=2)
    ax.set_ylabel("error")
    ax.set_xlabel('rank')
    ax.set_xticks(np.arange(0, m, 1))
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    fig.savefig('err.png', bbox_inches = 'tight', dpi=300);
    
def plot_corr_four(trop, names):
    m = 5
    dic_trop = trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    fig, (ax1) = plt.subplots(1, 1, sharey=True)
    
    ax1.boxplot(data_trop)
    med_trop = np.median(data_trop, axis=1)
    #ax1.plot(range(1, m), med_trop)
    ax1.set_xlabel('permutations')
    ax1.set_ylabel('correlation')
    ax1.set_title("STMF")
    ax1.set_xticks(np.arange(0, m, 5))
    ax1.set_xticklabels([names[0], names[1], names[2], names[3]])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82);
    
    
def plot_corr(trop, nmf, title):
    m = 11
    dic_trop = trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    fig.suptitle(title)
    
    ax1.boxplot(data_trop)
    med_trop = np.median(data_trop, axis=1)
    ax1.plot(range(1, m), med_trop)
    ax1.set_xlabel('rank')
    ax1.set_ylabel('correlation')
    ax1.set_title("STMF")
    ax1.set_xticks(np.arange(0, m, 5))
    ax1.set_xticklabels(['1', '5', '10'])
    
    dic_nmf = nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    ax2.boxplot(data_nmf)
    med_nmf = np.median(data_nmf, axis=1)
    ax2.plot(range(1, m), med_nmf)
    ax2.set_xlabel('rank')
    ax2.set_title("NMF")
    ax2.set_xticks(np.arange(0, m, 5))
    ax2.set_xticklabels(['1', '5', '10'])
    
    ax3.plot(med_trop)
    ax3.plot(med_nmf)
    ax3.legend(['STMF', 'NMF'], loc='lower left')
    ax3.set_xlabel('rank')
    ax3.set_ylabel('correlation')
    ax3.set_xticks(np.arange(0, m, 5))
    ax3.set_xticklabels(['1', '5', '10'])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82);
    
def plot_errors_twenty(rmse_trop, rmse_nmf, errors_trop, errors_nmf, title, m, location):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = rmse_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = rmse_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='upper left')
    ax1.set_xlabel('rank')
    ax1.set_ylabel('approximation error')
    ax1.set_xticks(np.arange(0, m, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'])
    
    ################
    dic_trop = errors_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = errors_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax2.plot(med_trop)
    ax2.plot(med_nmf)
    ax2.legend(['STMF', 'NMF'], loc='upper left')
    ax2.set_xlabel('rank')
    ax2.set_ylabel('prediction error')
    ax2.set_xticks(np.arange(0, m, 1))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def plot_approx_pred_ten(error_trop, error_nmf, corr_trop, corr_nmf, title, m, location, ylabel_1, ylabel_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='upper right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    ################
    dic_trop = corr_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = corr_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax2.plot(med_trop)
    ax2.plot(med_nmf)
    ax2.legend(['STMF', 'NMF'], loc='upper left')
    ax2.set_xlabel('rank')
    ax2.set_ylabel(ylabel_2)
    ax2.set_xticks(np.arange(0, m-1, 1))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def plot_approx_pred_fifteen(error_trop, error_nmf, corr_trop, corr_nmf, title, m, location, ylabel_1, ylabel_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='upper right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    
    ################
    dic_trop = corr_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = corr_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax2.plot(med_trop)
    ax2.plot(med_nmf)
    ax2.legend(['STMF', 'NMF'], loc='upper left')
    ax2.set_xlabel('rank')
    ax2.set_ylabel(ylabel_2)
    ax2.set_xticks(np.arange(0, m-1, 1))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def plot_corr_silh_ten(error_trop, error_nmf, corr_trop, corr_nmf, orig, title, m, location, ylabel_1, ylabel_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='lower right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    ################
    dic_trop = corr_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = corr_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    y = [orig] * (m-1)
    
    ax2.plot(med_trop)
    ax2.plot(med_nmf)
    ax2.plot(y)
    ax2.legend(['STMF', 'NMF', "Original data"], loc='lower right')
    ax2.set_xlabel('rank')
    ax2.set_ylabel(ylabel_2)
    ax2.set_xticks(np.arange(0, m-1, 1))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def plot_corr_silh_fifteen(error_trop, error_nmf, corr_trop, corr_nmf, orig, title, m, location, ylabel_1, ylabel_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='lower right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    
    ################
    dic_trop = corr_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = corr_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    y = [orig] * (m-1)
    
    ax2.plot(med_trop)
    ax2.plot(med_nmf)
    ax2.plot(y)
    ax2.legend(['STMF', 'NMF', "Original data"], loc='lower right')
    ax2.set_xlabel('rank')
    ax2.set_ylabel(ylabel_2)
    ax2.set_xticks(np.arange(0, m-1, 1))
    ax2.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def plot_corr_ten(error_trop, error_nmf, title, m, location, ylabel_1):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = error_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = error_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='lower right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', "6", "7", "8", "9", "10"])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return


def plot_corr_twenty(corr_trop, corr_nmf, title, m, location, ylabel_1):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3), dpi=200)
    fig.suptitle(title)
    
    dic_trop = corr_trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    dic_nmf = corr_nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    ax1.plot(med_trop)
    ax1.plot(med_nmf)
    ax1.legend(['STMF', 'NMF'], loc='lower right')
    ax1.set_xlabel('rank')
    ax1.set_ylabel(ylabel_1)
    ax1.set_xticks(np.arange(0, m-1, 1))
    ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'])
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(location, bbox_inches = 'tight', dpi=300);
    
    return

def plot_medians(trop, nmf):
    dic_trop = trop
    labels_trop, data_trop = [*zip(*dic_trop.items())]
    med_trop = np.median(data_trop, axis=1)
    
    dic_nmf = nmf
    labels_nmf, data_nmf = [*zip(*dic_nmf.items())]
    med_nmf = np.median(data_nmf, axis=1)
    
    plt.plot(med_trop)
    plt.plot(med_nmf)
    plt.legend(['TMF', 'NMF'], loc='lower right', dpi=300)


def plot_original_trop_nmf(A, trop, nmf, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    fig.suptitle(title)
    
    a = ax[0].pcolor(A)  # original data matrix
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Original")
    plt.ylabel("gene")
    
    b = ax[1].pcolor(np.array(trop))  # tmf
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("STMF")
    plt.xlabel("patient")
    
    c = ax[2].pcolor(np.array(nmf))  # nmf
    fig.colorbar(c, ax=ax[2])
    ax[2].set_title("NMF")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    
    plt.savefig(location, dpi=500)
    plt.show();
    
def plot_original_trop_nmf_one_colorbar(A, trop, nmf, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    fig.suptitle(title)

    min_a, max_a = np.min(A), np.max(A)
    D = np.array(trop)
    min_d, max_d = np.min(D), np.max(D)
    E = np.array(nmf)
    min_e, max_e = np.min(E), np.max(E)
    final_min = min([min_a, min_d, min_e])
    final_max = max([max_a, max_d, max_e])

    a = ax[0].pcolor(A, vmin=final_min, vmax=final_max)  # original data matrix
    ax[0].set_title("Original")
    ax[0].set_ylabel("patient", fontsize=16)

    d = ax[1].pcolor(D, vmin=final_min, vmax=final_max)  # our model
    ax[1].set_title("STMF")
    ax[1].set_xlabel("gene", fontsize=16)

    e = ax[2].pcolor(E, vmin=final_min, vmax=final_max)  # nmf
    ax[2].set_title("NMF")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(e, cax=cbar_ax)
    plt.savefig(location, bbox_inches = 'tight', dpi=300)
    plt.show();
    

def plot_UI(A, trop, nmf, title):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    fig.suptitle(title)
    
    a = ax[0].pcolor(A)  # original data matrix
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Original")
    
    b = ax[1].pcolor(np.array(trop))  # tmf
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("TMF")
    
    c = ax[2].pcolor(np.array(nmf))  # nmf
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("NMF")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig("UI.png", dpi=300)
    plt.show();


def plot_original_missing_trop_nmf_one_colorbar(A, missing, trop, nmf, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))
    fig.suptitle(title)
    
    min_a, max_a = np.min(A), np.max(A)
    
    B = np.array(missing)
    min_b, max_b = np.min(B), np.max(B)
    
    D = np.array(trop)
    min_d, max_d = np.min(D), np.max(D)
    
    E = np.array(nmf)
    min_e, max_e = np.min(E), np.max(E)
    
    final_min = min([min_a, min_b, min_d, min_e])
    final_max = max([max_a, max_b, max_d, max_e])
    
    a = ax[0].pcolor(A, vmin=final_min, vmax=final_max)  # original data matrix
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Original")
    
    b = ax[1].pcolor(np.array(missing), vmin=final_min, vmax=final_max)  # matrix with missing values
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Missing values")
    
    c = ax[2].pcolor(np.array(trop), vmin=final_min, vmax=final_max)  # our model
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("STMF")

    d = ax[3].pcolor(np.array(nmf), vmin=final_min, vmax=final_max)  # nmf
    #fig.colorbar(d, ax=ax[3])
    ax[3].set_title("NMF")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(d, cax=cbar_ax)
    plt.savefig(location, bbox_inches = 'tight', dpi=300)
    plt.show();

def plot_synthetic_matrices(first, second, third, fourth, fifth, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
    fig.suptitle(title)

    A, B, C, D, E = np.array(first), np.array(second), np.array(third), np.array(fourth), np.array(fifth)
    min_a, max_a = np.min(A), np.max(A)
    min_b, max_b = np.min(B), np.max(B)
    min_c, max_c = np.min(C), np.max(C)
    min_d, max_d = np.min(D), np.max(D)
    min_e, max_e = np.min(E), np.max(E)
    final_min = min([min_a, min_b, min_c, min_d, min_e])
    final_max = max([max_a, max_b, max_c, max_d, max_e])

    a = ax[0].pcolor(A, vmin=final_min, vmax=final_max)  
    ax[0].set_title("First")

    b = ax[1].pcolor(B, vmin=final_min, vmax=final_max)  
    ax[1].set_title("Second")

    c = ax[2].pcolor(C, vmin=final_min, vmax=final_max) 
    ax[2].set_title("Third")

    d = ax[3].pcolor(D, vmin=final_min, vmax=final_max) 
    ax[3].set_title("Fourth")

    e = ax[4].pcolor(E, vmin=final_min, vmax=final_max) 
    ax[4].set_title("Fifth")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(e, cax=cbar_ax)
    plt.savefig(location, bbox_inches = 'tight', dpi=300)
    plt.show();

def plot_original_divided_trop_nmf(A, divided, missing, trop, nmf, title):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    fig.suptitle(title)

    min_a, max_a = np.min(A), np.max(A)
    D = np.array(trop)
    min_d, max_d = np.min(D), np.max(D)
    E = np.array(nmf)
    min_e, max_e = np.min(E), np.max(E)
    final_min = min([min_a, min_d, min_e])
    final_max = max([max_a, max_d, max_e])

    a = ax[0].pcolor(A, vmin=final_min, vmax=final_max)  # original data matrix
    ax[0].set_title("Original")

    d = ax[1].pcolor(D, vmin=final_min, vmax=final_max)  # our model
    ax[1].set_title("STMF")

    e = ax[2].pcolor(E, vmin=final_min, vmax=final_max)  # nmf
    ax[2].set_title("NMF")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(e, cax=cbar_ax)
    plt.show();


def plot_divided_missing(divided, missing):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 4))

    B = np.array(divided)
    min_b, max_b = np.min(B), np.max(B)
    C = np.array(missing)
    min_c, max_c = np.min(C), np.max(C)
    final_min = min([min_b, min_c])
    final_max = max([max_b, max_c])

    b = ax[0].pcolor(B, vmin=final_min, vmax=final_max)  # divided matrix
    ax[0].set_title("Divided")

    c = ax[1].pcolor(C, vmin=final_min, vmax=final_max)  # matrix with missing values
    ax[1].set_title("Missing values")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(c, cax=cbar_ax)
    plt.show();


def cluster_matrix(X):
    model_X = SpectralCoclustering(n_clusters=5, random_state=0)
    model_X.fit(X)
    row_perm = np.argsort(model_X.row_labels_)
    column_perm = np.argsort(model_X.column_labels_)
    fit_data_X = X[row_perm]
    fit_data_X = fit_data_X[:, column_perm]
    return fit_data_X, row_perm, column_perm

def polo_clustering(data_param):
    data = copy.deepcopy(data_param)
    D = pdist(data, 'euclidean')  # distance
    Z = linkage(D, 'ward')
    optimal_Z = optimal_leaf_ordering(Z, D)
    opt_order = leaves_list(optimal_Z)
    data = data[opt_order]
    
    data = data.T  # transpose
    D = pdist(data, 'euclidean')  # distance
    Z = linkage(D, 'ward')
    optimal_Z = optimal_leaf_ordering(Z, D)
    opt_order_columns = leaves_list(optimal_Z)
    data = data[opt_order_columns]
    return data.T, opt_order, opt_order_columns

def mean_absolute_error_all(X_orig, approx, missing_value):
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    for i in range(rows):
        for j in range(columns):
            error = abs(X_orig[i, j] - approx[i, j])
            errors.append(error)
    return sum(errors)/len(errors)

def mean_absolute_error(X_orig, X_with_missing_values, approx, missing_value):
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    for i in range(rows):
        for j in range(columns):
            if X_with_missing_values[i, j] == missing_value:
                error = (abs(X_orig[i, j] - approx[i, j]))
                errors.append(error)
    return sum(errors)/len(errors)

def rmse(X_orig, X_with_missing_values, approx, missing_value):
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    for i in range(rows):
        for j in range(columns):
            if X_with_missing_values[i, j] == missing_value:
                #error = (abs(X_orig[i, j] - approx[i, j]))
                error = (X_orig[i, j] - approx[i, j])**2 
                errors.append(error)
    return np.sqrt(sum(errors)/len(errors))

def rmse_approx(X_orig, X_with_missing_values, approx, missing_value):
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    for i in range(rows):
        for j in range(columns):
            if X_with_missing_values[i, j] != missing_value:
                #error = (abs(X_orig[i, j] - approx[i, j]))
                error = (X_orig[i, j] - approx[i, j])**2 
                errors.append(error)
    return np.sqrt(sum(errors)/len(errors))

def mean_relative_error(X_orig, X_with_missing_values, approx, missing_value):
    # divide the absolute error with actual value
    rows = X_orig.shape[0]
    columns = X_orig.shape[1]
    errors = []
    for i in range(rows):
        for j in range(columns):
            if X_with_missing_values[i, j] == missing_value:
                error = (abs(X_orig[i, j] - approx[i, j]))/abs(X_orig[i, j])
                errors.append(error)
    return sum(errors)/len(errors)

def create_matrix_with_missing_values(X, percentage, missing_value):
    rows = X.shape[0]
    columns = X.shape[1]
    elements = rows * columns
    zero_elements = int(percentage * elements)
    for i in range(zero_elements):
        random_row = random.randint(0, rows-1)
        random_column = random.randint(0, columns-1)
        X[random_row, random_column] = missing_value
    return X

def check_zeros(X):
    rows = X.shape[0]
    columns = X.shape[1]
    for i in range(rows):
        for j in range(columns):
            if X[i, j] == 0:
                print("there is a zero element")
    return

def get_max(product):
    data = product.data
    mask = product.mask
    rows = data.shape[0]
    columns = data.shape[1]
    result = []
    for j in range(columns):
        column_elements = []
        for i in range(rows):
            if mask[i, j] == False:
                column_elements.append(data[i, j])
        result.append(max(column_elements))
    return result


def get_min(product):
    data = product.data
    mask = product.mask
    rows = data.shape[0]
    columns = data.shape[1]
    result = []
    for j in range(columns):
        column_elements = []
        for i in range(rows):
            if not mask[i, j]:
                column_elements.append(data[i, j])
        if len(column_elements) == 0:  # only missing values
            raise Exception("there is an empty row/column in data")
        result.append(min(column_elements))
    return result


def get_min_2(product, param):
    data = product.data
    mask = product.mask
    rows = data.shape[0]
    columns = data.shape[1]
    result = []
    for j in range(columns):
        column_elements = []
        for i in range(rows):
            if not mask[i, j]:
                column_elements.append(data[i, j])
        if len(column_elements) == 0:  # only missing values
            raise Exception("there is an empty row/column in data")
        result.append(max(sorted(column_elements)[:param])) 
    return result


def get_coordinates(A):
    mask = A.mask
    rows = mask.shape[0]
    columns = mask.shape[1]
    coordinates = []
    for i in range(rows):
        for j in range(columns):
            if not mask[i, j]:
                coordinates.append((i, j))
    return coordinates


def create_dict(X):
    n_lines = len(X)
    temp = 0
    dictionary = dict()
    for i in range(10, n_lines + 10, 10):
        list = np.asarray([line for line in X][i-10:i])
        dictionary[temp] = list
        temp += 1
    return dictionary

def create_dict_two(X):
    n_lines = len(X)
    temp = 0
    dictionary = dict()
    for i in range(2, n_lines + 2, 2):
        list = np.asarray([line for line in X][i-2:i])
        dictionary[temp] = list
        temp += 1
    return dictionary

def create_dict_nmf(X):
    n_lines = len(X)
    temp = 0
    dictionary = dict()
    for i in range(1, n_lines + 1, 1):
        list = np.asarray([line for line in X][i-1:i])
        dictionary[temp] = list
        temp += 1
    return dictionary

def create_dict_five(X):
    n_lines = len(X)
    temp = 0
    dictionary = dict()
    for i in range(5, n_lines + 5, 5):
        list = np.asarray([line for line in X][i-5:i])
        dictionary[temp] = list
        temp += 1
    return dictionary


def plot_statistics(original_path, transformed_path):
    log = lambda x: math.log(1 + x, 2)
    with open(original_path, 'r') as f:
        num_cols = len(f.readline().split())
        f.seek(0)
        original = np.genfromtxt(f, usecols=range(1, num_cols))
    original = np.delete(original, 0, axis=0)
    original_shape = original.shape
    original = original.ravel()
    all_originals_to_transformed = np.asarray(list(map(log, original)))

    transf = genfromtxt(transformed_path, delimiter="\t")
    transf = np.delete(transf, 0, axis=0)
    transf = np.delete(transf, 0, axis=1)
    transf_shape = transf.shape
    transf = transf.ravel()

    transf_without_zeros = transf[transf != 0]

    minmax_original = "(" + str(round(min(original), 3)) + ', ' + str(round(max(original), 3)) + ')'
    minmax_transf_all = "(" + str(round(min(all_originals_to_transformed), 3)) + ', ' + str(
        round(max(all_originals_to_transformed), 3)) + ')'
    minmax_transf = "(" + str(round(min(transf), 3)) + ', ' + str(round(max(transf), 3)) + ')'
    minmax_transf_with_0 = "(" + str(round(min(transf_without_zeros), 3)) + ', ' + str(
        round(max(transf_without_zeros), 3)) + ')'


    columns = ('Original', 'Transf. all', 'Transf. subset', 'Transf. subset without 0')
    rows = ['(rows, columns)', '(min, max)', 'number of 0 elements']

    cell_text = [[original_shape, ' ', transf_shape, ''],
                 [minmax_original, minmax_transf_all, minmax_transf, minmax_transf_with_0],
                 [np.count_nonzero(original == 0), np.count_nonzero(all_originals_to_transformed == 0),
                  np.count_nonzero(transf == 0), np.count_nonzero(transf_without_zeros == 0)]]

    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    the_table.set_fontsize(36)
    the_table.scale(7, 7)
    plt.show();

def plot_dataset(original_path, transformed_path):
    log = lambda x: math.log(1 + x, 2)
    with open(original_path, 'r') as f:
        num_cols = len(f.readline().split())
        f.seek(0)
        original = np.genfromtxt(f, usecols=range(1, num_cols))
    original = np.delete(original, 0, axis=0)
    original_shape = original.shape
    original = original.ravel()
    all_originals_to_transformed = np.asarray(list(map(log, original)))

    transf = genfromtxt(transformed_path, delimiter="\t")
    transf = np.delete(transf, 0, axis=0)
    transf = np.delete(transf, 0, axis=1)
    transf_shape = transf.shape
    transf = transf.ravel()

    transf_without_zeros = transf[transf != 0]  # list(filter(lambda a: a != 0, transf))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 6))
    ax1.hist(original)
    ax1.set_title("Original")

    ax2.hist(all_originals_to_transformed)
    ax2.set_title("Transformed (all)")

    ax3.hist(transf)
    ax3.set_title("Transformed (subset)")

    ax4.hist(transf_without_zeros)
    ax4.set_title("Transformed (subset) without 0 elements")

    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.subplots_adjust(wspace=0.3);


def plot_all_approximations(approxs):
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
    fig.suptitle('All approximations')

    min_a, max_a = np.min(approxs[0]), np.max(approxs[0])
    min_b, max_b = np.min(approxs[1]), np.max(approxs[1])
    min_c, max_c = np.min(approxs[2]), np.max(approxs[2])
    min_d, max_d = np.min(approxs[3]), np.max(approxs[3])
    min_e, max_e = np.min(approxs[4]), np.max(approxs[4])
    final_min = min([min_a, min_b, min_c, min_d, min_e])
    final_max = max([max_a, max_b, max_c, max_d, max_e])

    a = ax[0].pcolor(approxs[0], vmin=final_min, vmax=final_max)
    ax[0].set_title("Approx. 1")

    b = ax[1].pcolor(approxs[1], vmin=final_min, vmax=final_max)
    ax[1].set_title("Approx. 2")

    c = ax[2].pcolor(approxs[2], vmin=final_min, vmax=final_max)
    ax[2].set_title("Approx. 3")

    d = ax[3].pcolor(approxs[3], vmin=final_min, vmax=final_max)
    ax[3].set_title("Approx. 4")

    e = ax[4].pcolor(approxs[4], vmin=final_min, vmax=final_max)
    ax[4].set_title("Approx. 5")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(e, cax=cbar_ax)
    plt.show();

def asc_dist_corr(dist, corr):
    x = dist
    y = corr
    L = sorted(zip(x,y), key=operator.itemgetter(0))
    new_x, new_y = zip(*L)
    return new_x, new_y

def dist_corr(orig, approx): 
    # calculating correlations of column vectors
    orig = copy.deepcopy(orig.T) # we transpose so that we have patients in columns
    approx = copy.deepcopy(approx.T)
    distances = []
    dist_correlation = []
    pearson_correlation = []
    spearman_correlation = []
    n_col = orig.shape[1]  # number of columns
    #print(n_col)
    dis_corr, A_cent, B_cent = dc.dcor(orig.T, approx.T) # A_cent and B_cent are centered matrix, dcor computes by rows, we need here patients to be in rows
    #print(orig.shape)
    #print(approx.shape)
    for i in range(n_col):
        column_origin = orig[:, i] 
        column_approx = approx[:, i]
        # those columns are from dist_corr function
        column_origin_A = A_cent[:, i] 
        column_approx_B = B_cent[:, i]
        
        dist = np.linalg.norm(column_origin-column_approx)
        distances.append(dist)
        
        shape_orig = column_origin.shape[0]
        shape_approx = column_approx.shape[0]
        a = np.reshape(column_origin, (shape_orig, 1)) # np.matrix
        b = np.reshape(column_approx, (shape_approx, 1))
        
        #dis_corr, A_vec, B_vec = dc.dcor(a, b)
        dist_correlation.append(np.linalg.norm(column_origin_A - column_approx_B))
        
        pear_corr = pearsonr(column_origin, column_approx)[0]
        pearson_correlation.append(pear_corr)
        
        spear_corr = spearmanr(column_origin, column_approx).correlation
        spearman_correlation.append(spear_corr)
        
        # order values on x-axis and corresponding y-values
        new_dist, new_dc = asc_dist_corr(distances, dist_correlation)
        new_pears = asc_dist_corr(distances, pearson_correlation)[1]
        new_spear = asc_dist_corr(distances, spearman_correlation)[1]
        
    return new_dist, new_dc, new_pears, new_spear


def plot_original_divided_missing_trop_nmf(A, divided, missing, trop, nmf, title):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    fig.suptitle(title)

    min_a, max_a = np.min(A), np.max(A)
    D = np.array(trop)
    min_d, max_d = np.min(D), np.max(D)
    E = np.array(nmf)
    min_e, max_e = np.min(E), np.max(E)
    final_min = min([min_a, min_d, min_e])
    final_max = max([max_a, max_d, max_e])

    a = ax[0].pcolor(A, vmin=final_min, vmax=final_max)  # original data matrix
    ax[0].set_title("Original")

    d = ax[1].pcolor(D, vmin=final_min, vmax=final_max)  # our model
    ax[1].set_title("STMF")

    e = ax[2].pcolor(E, vmin=final_min, vmax=final_max)  # nmf
    ax[2].set_title("NMF")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(e, cax=cbar_ax)
    plt.show()
    
def plot_latent_binary(approx_rank_1_binary, approx_rank_2_binary, approx_rank_3_binary, location):
    n = approx_rank_1_binary.shape[0] *  approx_rank_1_binary.shape[1]
    # binary visualization
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5)) 

    a = ax[0].pcolor(approx_rank_1_binary)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1 = " + str(round(np.count_nonzero(approx_rank_1_binary)*100/n,2)) + "%")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_binary)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2 = " + str(round(np.count_nonzero(approx_rank_2_binary)*100/n,2)) + "%")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_binary)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3 = " + str(round(np.count_nonzero(approx_rank_3_binary)*100/n,2)) + "%")
    #ax[2].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_binary_five(approx_rank_1_binary, approx_rank_2_binary, approx_rank_3_binary,  approx_rank_4_binary,  approx_rank_5_binary, location):
    n = approx_rank_1_binary.shape[0] *  approx_rank_1_binary.shape[1]
    # binary visualization
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 10)) 

    a = ax[0].pcolor(approx_rank_1_binary)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1 = " + str(round(np.count_nonzero(approx_rank_1_binary)*100/n,2)) + "%")
    ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_binary)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2 = " + str(round(np.count_nonzero(approx_rank_2_binary)*100/n,2)) + "%")
    ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_binary)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3 = " + str(round(np.count_nonzero(approx_rank_3_binary)*100/n,2)) + "%")
    ax[2].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[3].pcolor(approx_rank_4_binary)  # rank 4
    #fig.colorbar(c, ax=ax[2])
    ax[3].set_title("Latent matrix 4 = " + str(round(np.count_nonzero(approx_rank_4_binary)*100/n,2)) + "%")
    ax[3].set_aspect('equal', adjustable='box', anchor='C')
    
    e = ax[4].pcolor(approx_rank_5_binary)  # rank 5
    #fig.colorbar(c, ax=ax[2])
    ax[4].set_title("Latent matrix 5 = " + str(round(np.count_nonzero(approx_rank_5_binary)*100/n,2)) + "%")
    ax[4].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_binary_two(approx_rank_1_binary, approx_rank_2_binary, location):
    n = approx_rank_1_binary.shape[0] *  approx_rank_1_binary.shape[1]
    # binary visualization
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) 

    a = ax[0].pcolor(approx_rank_1_binary)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1 = " + str(round(np.count_nonzero(approx_rank_1_binary)*100/n,2)) + "%")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_binary)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2 = " + str(round(np.count_nonzero(approx_rank_2_binary)*100/n,2)) + "%")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_binary_four(approx_rank_1_binary, approx_rank_2_binary, approx_rank_3_binary,  approx_rank_4_binary, location):
    n = approx_rank_1_binary.shape[0] *  approx_rank_1_binary.shape[1]
    # binary visualization
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13, 8)) 

    a = ax[0].pcolor(approx_rank_1_binary)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1 = " + str(round(np.count_nonzero(approx_rank_1_binary)*100/n,2)) + "%")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_binary)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2 = " + str(round(np.count_nonzero(approx_rank_2_binary)*100/n,2)) + "%")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_binary)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3 = " + str(round(np.count_nonzero(approx_rank_3_binary)*100/n,2)) + "%")
    #ax[2].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[3].pcolor(approx_rank_4_binary)  # rank 4
    #fig.colorbar(c, ax=ax[2])
    ax[3].set_title("Latent matrix 4 = " + str(round(np.count_nonzero(approx_rank_4_binary)*100/n,2)) + "%")
    #ax[3].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_real(approx_rank_1_real, approx_rank_2_real, approx_rank_3_real, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_real)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3")
    #ax[2].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    
    plt.savefig(location, dpi=300)
    plt.show()
    
    
def plot_latent_real_two(approx_rank_1_real, approx_rank_2_real, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_real_five(approx_rank_1_real, approx_rank_2_real, approx_rank_3_real, approx_rank_4_real, approx_rank_5_real, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 10)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_real)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3")
    ax[2].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[3].pcolor(approx_rank_4_real)  # rank 4
    #fig.colorbar(c, ax=ax[2])
    ax[3].set_title("Latent matrix 4")
    ax[3].set_aspect('equal', adjustable='box', anchor='C')
    
    e = ax[4].pcolor(approx_rank_5_real)  # rank 5
    #fig.colorbar(c, ax=ax[2])
    ax[4].set_title("Latent matrix 5")
    ax[4].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_real_four(approx_rank_1_real, approx_rank_2_real, approx_rank_3_real, approx_rank_4_real, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13, 8)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_real)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3")
    #ax[2].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[3].pcolor(approx_rank_4_real)  # rank 4
    #fig.colorbar(c, ax=ax[2])
    ax[3].set_title("Latent matrix 4")
    #ax[3].set_aspect('equal', adjustable='box', anchor='C')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    
    plt.savefig(location, dpi=300)
    plt.show()

def plot_latent_real_approx(approx_rank_1_real, approx_rank_2_real, approx_rank_3_real, approx, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_real)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3")
    #ax[2].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[3].pcolor(approx)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[3].set_title("Approximation")
    #ax[3].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_real_approx_two(approx_rank_1_real, approx_rank_2_real, approx, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[2].pcolor(approx)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Approximation")
    #ax[3].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_real_approx_four(approx_rank_1_real, approx_rank_2_real, approx_rank_3_real, approx_rank_4_real, approx, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(13, 8)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    #ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    #ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_real)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3")
    #ax[2].set_aspect('equal', adjustable='box', anchor='C')
    
    f = ax[3].pcolor(approx_rank_4_real)  # rank 4
    #fig.colorbar(c, ax=ax[2])
    ax[3].set_title("Latent matrix 4")
    #ax[3].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[4].pcolor(approx)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[4].set_title("Approximation")
    #ax[5].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_latent_real_approx_five(approx_rank_1_real, approx_rank_2_real, approx_rank_3_real, approx_rank_4_real, approx_rank_5_real, approx, location):
    n = approx_rank_1_real.shape[0] *  approx_rank_1_real.shape[1]
    # real values visualization
    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(15, 10)) 

    a = ax[0].pcolor(approx_rank_1_real)  # rank 1
    #fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Latent matrix 1")
    ax[0].set_aspect('equal', adjustable='box', anchor='C')

    b = ax[1].pcolor(approx_rank_2_real)  # rank 2
    #fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Latent matrix 2")
    ax[1].set_aspect('equal', adjustable='box', anchor='C')

    c = ax[2].pcolor(approx_rank_3_real)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[2].set_title("Latent matrix 3")
    ax[2].set_aspect('equal', adjustable='box', anchor='C')
    
    f = ax[3].pcolor(approx_rank_4_real)  # rank 4
    #fig.colorbar(c, ax=ax[2])
    ax[3].set_title("Latent matrix 4")
    ax[3].set_aspect('equal', adjustable='box', anchor='C')
    
    g = ax[4].pcolor(approx_rank_5_real)  # rank 5
    #fig.colorbar(c, ax=ax[2])
    ax[4].set_title("Latent matrix 5")
    ax[4].set_aspect('equal', adjustable='box', anchor='C')
    
    d = ax[5].pcolor(approx)  # rank 3
    #fig.colorbar(c, ax=ax[2])
    ax[5].set_title("Approximation")
    ax[5].set_aspect('equal', adjustable='box', anchor='C')

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    
    plt.savefig(location, dpi=300)
    plt.show()

def plot_U_V(U, V):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 4))

    B = np.array(U)
    min_b, max_b = np.min(B), np.max(B)
    C = np.array(V)
    min_c, max_c = np.min(C), np.max(C)
    final_min = min([min_b, min_c])
    final_max = max([max_b, max_c])

    b = ax[0].pcolor(B, vmin=final_min, vmax=final_max)
    ax[0].set_title("Factor matrix U")

    c = ax[1].pcolor(C, vmin=final_min, vmax=final_max)
    ax[1].set_title("Factor matrix V")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.6])
    fig.colorbar(c, cax=cbar_ax)
    plt.show()


def plot_U_V_diff_scales(U, V, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.suptitle(title)
    
    a = ax[0].pcolor(U)
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Factor matrix U")

    b = ax[1].pcolor(V)
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Factor matrix V")
    ax[1].set_yticks([1, 2, 3])
    ax[1].set_yticklabels(["1", "2", "3"])
    
    
    plt.savefig(location, dpi=300)
    plt.show()

def plot_U_V_diff_scales_two(U, V, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.suptitle(title)
    
    a = ax[0].pcolor(U)
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Factor matrix U")
  
    b = ax[1].pcolor(V)
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Factor matrix V")
    ax[1].set_yticks([1, 2])
    ax[1].set_yticklabels(["1", "2"])
    
    plt.savefig(location, dpi=300)
    plt.show()
    
def plot_U_V_diff_scales_four(U, V, title, location):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.suptitle(title)
    
    a = ax[0].pcolor(U)
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Factor matrix U")

    b = ax[1].pcolor(V)
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Factor matrix V")
    ax[1].set_yticks([1, 2, 3, 4])
    ax[1].set_yticklabels(["1", "2", "3", "4"])
    
    plt.savefig(location, dpi=300)
    plt.show()

def plot_original_clustered(data, fit_data_X):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    a = ax[0].pcolor(data)
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("Original")

    b = ax[1].pcolor(fit_data_X)
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("Clustered")

    plt.show()


def plot_U_V_stmf_nmf(U, V, U_nmf, V_nmf):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    a = ax[0].pcolor(U)
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title("STMF - matrix U")

    b = ax[1].pcolor(V)
    fig.colorbar(b, ax=ax[1])
    ax[1].set_title("STMF - matrix V")

    c = ax[2].pcolor(U_nmf)
    fig.colorbar(c, ax=ax[2])
    ax[2].set_title("NMF - matrix U")

    d = ax[3].pcolor(V_nmf)
    fig.colorbar(d, ax=ax[3])
    ax[3].set_title("NMF - matrix V")

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)

    plt.show()
