from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_silhouette(n_clusters,dst_matrix, cluster_index, fig, ax, coord):
    ax1, ax2=ax
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(coord) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(dst_matrix,cluster_index,metric="precomputed")
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dst_matrix,cluster_index,metric="precomputed")
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_index == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.text(silhouette_avg-0.05,0,f"Avg. silhouette={np.round(silhouette_avg,2)}",rotation=90,color="red")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    coord_=pd.DataFrame()
    # 2nd Plot showing the actual clusters formed
    coord_['Projection 1'] = coord[:,0]
    coord_['Projection 2'] = coord[:,1]
    #transdict={0:1,1:2,2:3,3:0,4:4}
    #cluster_index = np.array([transdict[letter] for letter in cluster_index])
    coord_["Class"]=cluster_index
    sns.scatterplot(
        x="Projection 1", y="Projection 2",
        hue="Class",
        palette=sns.color_palette(cm.nipy_spectral([float(i)  / n_clusters for i in sorted(np.unique(cluster_index))])),
        data=coord_,
        legend="full",
        alpha=0.3,
        ax=ax2
    )
    sns.despine(fig)
    ax2.set_title("The visualization of the clustered data.")

    plt.suptitle(
        "Silhouette analysis for clustering on data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    
def get_prediction_strength(k, dist_mat_train_centroids_test, test_labels):
    '''
    Function for calculating the prediction strength of clustering
    
    Parameters
    ----------
    k : int
        The number of clusters
    train_centroids : array
        Centroids from the clustering on the training set
    test_labels : array
        Labels predicted for the test set
        
    Returns
    -------
    prediction_strength : float
        Calculated prediction strength
    '''
    if dist_mat_train_centroids_test is None:
        return 0
    closest_cent=np.argmin(dist_mat_train_centroids_test,axis=0)
    
    # populate the co-membership matrix
    D = np.equal(closest_cent.reshape(-1,1),closest_cent.reshape(1,-1))
    np.fill_diagonal(D,0.0)
    
    # calculate the prediction strengths for each cluster
    ss = []
    for j in range(k):
        s = 0
        n_examples_j = np.sum(test_labels == j)
        same_label = np.equal(test_labels.reshape(-1,1),test_labels.reshape(1,-1))
        row_label_j=same_label[test_labels==j,:]
        row_D_j=D[test_labels==j,:]
        s=(row_label_j*row_D_j).sum()
        ss.append(s / (1+ n_examples_j * (n_examples_j - 1))) 

    prediction_strength = min(ss)

    return prediction_strength
    
  