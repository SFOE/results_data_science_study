import pandas as pd
import os
import numpy as np
from functools import reduce
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from cluster_metric import plot_silhouette, get_prediction_strength
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)

import geopandas as gpd


from settings import folder_data


def get_features():
    final = pd.read_csv(os.path.join(folder_data, "combined.csv")).set_index(
        "BFS_NUMMER"
    )
    final = final.drop("CH")
    return final


def get_mapping():
    mapping = pd.read_excel(
        os.path.join(
            folder_data,
            "Schweizerische Gemeinden und zuständige Stromnetzbetreiber.xlsx",
        ),
        header=2,
    )
    mapping = mapping.astype({"Gde-Nr.": str}).set_index(["Gde-Nr."])
    return mapping


def get_mapping_commune():
    # Manually tracking the municipalities that are merged together
    # dict-like file indicating the changes in the GDE code of newly merged municipalities
    mapping_commune = pd.read_csv(
        os.path.join(folder_data, "mapping_commune.csv")
    ).astype({"From": "category", "To": "category"})
    return mapping_commune


def fusion_commune(df, cols_to_agg=None):
    mapping_commune = get_mapping_commune()
    for i, row in mapping_commune.iterrows():
        from_ = str(row["From"])
        to = str(row["To"])
        if from_ in df.index:
            if to in df.index:
                # Add to the current municipality
                if cols_to_agg is None:
                    df.loc[to, :] += df.loc[from_, :]
                else:
                    df.loc[to, cols_to_agg] += df.loc[from_, cols_to_agg]
            else:
                # If creation of a new municipality, we add a new row
                if cols_to_agg is None:
                    df.loc[to, :] = df.loc[from_, :]
                else:
                    df.loc[to, cols_to_agg] = df.loc[from_, cols_to_agg]
            # Remove old municipalities
            df = df.drop(from_, axis=0)
    return df


def get_infra():
    all_data = []

    for path in glob.glob(os.path.join(folder_data, "*_closest.csv")):
        colname = "com_BFS_NU"
        if "prod_plant" in path:
            colname = "closest_BF"
        name_file = path.split("/")[-1].split("_closest")[0]
        if "prod_plant" in path or "consumption" in path:
            # In this case we are interested in the total sum of power, load, etc...
            stats = "sum"
        else:
            # In this case we are interested in the number of critical sites
            stats = "count"
        data = pd.read_csv(path, usecols=[colname, stats]).astype({colname: str})
        data = data.rename(
            columns={
                colname: "Regions-ID",
                stats: name_file,
            }
        )
        all_data.append(data)
    reduce_func = lambda left, right: pd.merge(
        left, right, on=["Regions-ID"], how="outer"
    )
    critical_infra = (
        reduce(
            reduce_func,
            all_data,
        )
        .fillna(0)
        .set_index("Regions-ID")
    )
    return critical_infra


def read_xlsx_from_atlas(path, nrows=2223):
    df = pd.read_excel(path, header=[2, 3], skiprows=[4], nrows=nrows, na_values="X")
    df.columns = [
        df.columns[i][1] if i in [0, 1] else df.columns[i][0]
        for i in range(len(df.columns))
    ]
    # set to zero to change the type to int and then str
    df.loc[df.Regionsname == "Suisse", "Regions-ID"] = 0
    df = df.astype({"Regions-ID": int}).astype({"Regions-ID": str})
    # set to ch
    df.loc[df.Regionsname == "Suisse", "Regions-ID"] = "CH"
    df = df.set_index("Regions-ID")
    return df


def aggregate_by_grd(features, mapping, columns_to_keep):
    merged = pd.merge(mapping, features, how="inner", left_index=True, right_index=True)
    # use absolute number in order to merge the data/GRD
    merged["Population - age 0-19 ans"] = (
        merged["Population - Part du groupe d'âge 0-19 ans"]
        * merged["Population - Habitants"]
    )
    merged["Population - age 20-64 ans"] = (
        merged["Population - Part du groupe d'âge 20-64 ans"]
        * merged["Population - Habitants"]
    )
    merged["Population - age 65+ ans"] = (
        merged["Population - Part du groupe d'âge 65+ ans"]
        * merged["Population - Habitants"]
    )
    merged["Revenu_nb_contribuable"] = (
        merged["Revenu imposable, en mio. de francs"]
        / merged["Revenu imposable par contribuable, en francs"]
        * 1e6
    )
    merged["Revenu_nb_habitant"] = (
        merged["Revenu imposable, en mio. de francs"]
        / merged["Revenu imposable par habitant/-e, en francs"]
        * 1e6
    )
    merged["Aide_social_pop"] = (
        merged["Nombre de bénéficiaires"] / merged["Taux d'aide sociale"]
    )

    column_filtered = merged[columns_to_keep]

    agg_dict = {}
    col_add = [
        "Population - Habitants",
        "Population - age 0-19 ans",
        "Population - age 20-64 ans",
        "Population - age 65+ ans",
        "Surface - Surface, total",
        "Economie - Emplois, total",
        "Nombre de bénéficiaires",
        "Aide_social_pop",
        "Revenu imposable, en mio. de francs",
        "Revenu_nb_contribuable",
        "Revenu_nb_habitant",
        "train_station",
        "bank",
        "hospital",
        "prod_plant_50MW",
        "official_gov",
        "consumption",
        "supermarket",
        "airport_filtered",
    ]
    discard = ["Name", "REGION"]
    for c in columns_to_keep:
        if c in discard:
            continue
        if c in col_add:
            agg_dict[c] = "sum"

    aggregated_by_GRD = column_filtered.groupby("Name").agg(agg_dict)
    aggregated_by_GRD["Population - Part du groupe d'âge 0-19 ans"] = (
        aggregated_by_GRD["Population - age 0-19 ans"]
        / aggregated_by_GRD["Population - Habitants"]
    )
    aggregated_by_GRD["Population - Part du groupe d'âge 20-64 ans"] = (
        aggregated_by_GRD["Population - age 20-64 ans"]
        / aggregated_by_GRD["Population - Habitants"]
    )
    aggregated_by_GRD["Population - Part du groupe d'âge 65+ ans"] = (
        aggregated_by_GRD["Population - age 65+ ans"]
        / aggregated_by_GRD["Population - Habitants"]
    )
    aggregated_by_GRD["Taux d'aide sociale"] = (
        aggregated_by_GRD["Nombre de bénéficiaires"]
        / aggregated_by_GRD["Aide_social_pop"]
    )
    aggregated_by_GRD["Revenu imposable par contribuable, en francs"] = (
        aggregated_by_GRD["Revenu imposable, en mio. de francs"]
        / aggregated_by_GRD["Revenu_nb_contribuable"]
        * 1e6
    )
    aggregated_by_GRD["Revenu imposable par habitant/-e, en francs"] = (
        aggregated_by_GRD["Revenu imposable, en mio. de francs"]
        / aggregated_by_GRD["Revenu_nb_habitant"]
        * 1e6
    )
    aggregated_by_GRD = aggregated_by_GRD.drop(
        columns=[
            "Population - age 0-19 ans",
            "Population - age 20-64 ans",
            "Population - age 65+ ans",
            "Nombre de bénéficiaires",
            "Revenu imposable, en mio. de francs",
            "Aide_social_pop",
            "Revenu_nb_contribuable",
            "Revenu_nb_habitant",
        ]
    )

    return aggregated_by_GRD


def weighting_scheme(scheme_name, col_names):
    assert scheme_name in [None, "economic", "vuln", "energie"]
    weights = np.ones((1, 14))

    weighting_scheme_economic = [
        "Economie - Emplois, total",
        "bank",
        "train_station",
        "supermarket",
        "Revenu imposable par contribuable, en francs",
    ]
    weighting_scheme_vuln = ["hospital", "Population - Part du groupe d'âge 65+ ans"]
    weighting_scheme_energie = ["prod_plant_50MW"]

    # choose which kind of
    if scheme_name is not None:
        if scheme_name == "economic":
            for i, c in enumerate(col_names):
                if c in weighting_scheme_economic:
                    weights[0, i] = 2

        if scheme_name == "vuln":
            for i, c in enumerate(col_names):
                if c in weighting_scheme_vuln:
                    weights[0, i] = 2

        if scheme_name == "energie":
            for i, c in enumerate(col_names):
                if c in weighting_scheme_energie:
                    weights[0, i] = 2
    return weights


def plot_pairwise(X, columns, against, name_col):
    fig, ax = plt.subplots(2, 5, figsize=(14, 10))
    ax = ax.flatten()
    for i, c in enumerate(columns):
        ax[i].scatter(X[c], X[against])
        ax[i].set_ylabel(name_col)
        ax[i].set_xlabel(c)
        # find line of best fit
        a, b = np.polyfit(X[c], X[against], 1)

        # add line of best fit to plot
        x = X[c].sort_values()
        ax[i].plot(
            x,
            a * x + b,
            linestyle="--",
            linewidth=2,
            color="red",
            label="best linear relation",
        )

        mask = X[c] > 0
        a, b = np.polyfit(np.log(X[c][mask]), np.log(X[against][mask]), 1)
        ax[i].plot(
            x,
            np.exp(b) * x**a,
            linestyle="--",
            linewidth=2,
            color="black",
            label="best log-log relation",
        )
        ax[i].loglog(nonpositive="mask")
    plt.legend()
    plt.tight_layout()
    sns.despine(fig)
    return fig


def compute_prediction_strength(n_clusters, constr, params, X, metric):

    X_train, X_test, index_train, index_test = train_test_split(
        X, np.arange(len(X)), test_size=0.2, shuffle=True
    )
    # training and testing model
    model_tr = constr(n_clusters, **params)
    model_t = constr(n_clusters, **params)

    # fit train model
    _ = model_tr.fit_predict(X_train)  # need the centroid
    # fit test
    cluster_index_test = model_t.fit_predict(X_test)

    if isinstance(model_tr, KMedoids):
        train_centroids = index_train[model_tr.medoid_indices_]
        dst_matrix = pairwise_distances(X, metric=metric)
        dst_matrix[np.arange(X.shape[0]), np.arange(X.shape[0])] = 0
        dst_matrix_train_test = dst_matrix[train_centroids, :][:, index_test]

    elif isinstance(model_tr, KMeans):
        dst_matrix_train_test = pairwise_distances(
            model_tr.cluster_centers_, X_test, metric=metric
        )
    elif isinstance(model_tr, GaussianMixture):
        dst_matrix_train_test = pairwise_distances(
            model_tr.means_, X_test, metric=metric
        )
    else:
        dst_matrix_train_test = None
    pred_strength = get_prediction_strength(
        n_clusters, dst_matrix_train_test, cluster_index_test
    )
    return pred_strength


def compute_cluster(
    n_clusters,
    X,
    name_model="KMeans",
    metric="l2",
    compute_figure=True,
    save_figure=False,
    filename=None,
    X_vis=None,
    return_metrics=True,
):
    constr, params = choose_clustering_algo(name_model, metric)
    model = constr(n_clusters, **params)

    # Fit the model on all the data
    cluster_index_ = model.fit_predict(X)
    dst_matrix = pairwise_distances(X, metric=metric)
    dst_matrix[np.arange(X.shape[0]), np.arange(X.shape[0])] = 0

    if X_vis is None:
        # Take the first 2 dimensions
        X_vis = X[:, :2]
    if compute_figure:

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        plot_silhouette(n_clusters, dst_matrix, cluster_index_, fig, ax, X_vis)
        if save_figure:
            if filename is None:
                filename = (
                    f"{name_model}_{metric}_{n_clusters}_no_airport_conso_weighting"
                )
            plt.savefig(os.path.join("../figure", f"{filename}.png"), dpi=150)

    pred_strength = compute_prediction_strength(n_clusters, constr, params, X, metric)
    sil = silhouette_score(dst_matrix, cluster_index_, metric="precomputed")
    davies = davies_bouldin_score(X, cluster_index_)
    calinski = calinski_harabasz_score(X, cluster_index_)
    bic = 0
    if name_model == "GMM":
        bic = model.bic(X)
    if return_metrics:
        return pred_strength, sil, davies, calinski, bic
    else:
        return cluster_index_


def order_clustering(cluster_index_, classified, n_clusters):
    ordered = (
        classified.groupby("cluster_index").mean().sort_values("Population - Habitants")
    )
    ordered["new_order"] = np.arange(n_clusters)
    ordered = ordered.sort_index()
    permutation_label = ordered.new_order.to_dict()
    cluster_index_ = cluster_index_.replace(
        {"cluster_index": permutation_label}
    ).astype({"cluster_index": "category"})
    return cluster_index_


def map_visualization(mapping, cluster_index_, simplified=True):
    # Load map
    folder_gis = os.path.join("../data_gis", "SHAPEFILE_LV95_LN02-2")
    swiss_map = gpd.read_file(os.path.join(folder_gis, "group_commune.shp")).astype(
        {"BFS_NUMMER": str}
    )
    # Each GRD is assigned to a category
    tmp = pd.merge(
        mapping.reset_index(),
        cluster_index_,
        left_on="Name",
        right_on="Name",
        how="inner",
    )
    if not simplified:
        # Some municipality can be served by 2 GRDs or more. These are colored by another color
        # We add the name of the GRDs (separated by /) that serve the same municipality
        tmp = pd.merge(
            tmp.drop(columns="Name"),
            tmp.groupby("Gde-Nr.")
            .Name.apply(list)
            .apply(lambda x: " / ".join(sorted(x))),
            left_on="Gde-Nr.",
            right_index=True,
        )
        #  Make sure that there is only one value per municipality and cluster index
        tmp = tmp[~tmp[["Gde-Nr.", "cluster_index"]].duplicated()]
        tmp = tmp.astype({"cluster_index": str})
        # Replace the cluster index, the new one is separated by - if it belongs to multiple clusters
        # e.g. 1-2, or 2, etc...
        tmp = pd.merge(
            tmp.drop(columns="cluster_index"),
            tmp.groupby("Gde-Nr.")
            .cluster_index.apply(list)
            .apply(lambda x: "-".join(sorted(x))),
            left_on="Gde-Nr.",
            right_index=True,
        )
        # Drop duplicated rows
        tmp = tmp[~tmp[["Gde-Nr.", "cluster_index"]].duplicated()]
        tmp = tmp.astype({"cluster_index": "category"})
    else:
        # We add the name of the GRDs (separated by /) that serve the same municipality
        tmp = pd.merge(
            tmp.drop(columns="Name"),
            tmp.groupby("Gde-Nr.")
            .Name.apply(list)
            .apply(lambda x: " / ".join(sorted(x))),
            left_on="Gde-Nr.",
            right_index=True,
        )
        tmp = tmp[~tmp[["Gde-Nr.", "cluster_index"]].duplicated()]
        tmp = tmp.astype({"cluster_index": int})

        # Assign the maximal cluster index on the map
        tmp = pd.merge(
            tmp.drop(columns="cluster_index"),
            tmp.groupby("Gde-Nr.").cluster_index.max(),
            left_on="Gde-Nr.",
            right_index=True,
        )
        tmp = tmp[~tmp[["Gde-Nr.", "cluster_index"]].duplicated()]
        tmp = tmp.astype({"cluster_index": "category"})

    swiss_map = pd.merge(
        tmp, swiss_map, left_on="Gde-Nr.", right_on="BFS_NUMMER", how="right"
    )
    swiss_map = swiss_map[["NAME", "Name", "geometry", "BFS_NUMMER", "cluster_index"]]
    swiss_map = gpd.GeoDataFrame(swiss_map)
    swiss_map.loc[swiss_map["Name"].isnull(), "Name"] = "Unknown"
    return swiss_map


def plot_metrics(
    smallest,
    largest,
    name_model,
    pred_strength_,
    avg_silhouette,
    davies_bouldin_,
    bic,
    calinski_harabasz_,
):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(18, 7)
    if name_model == "GMM":
        ax[0, 0].errorbar(
            np.arange(smallest, largest + 1), bic.mean(axis=0), yerr=bic.std(axis=0)
        )
        ax[0, 0].set_xlabel("N. of clusters")
        ax[0, 0].set_ylabel("BIC")
    else:
        ax[0, 0].errorbar(
            np.arange(smallest, largest + 1),
            avg_silhouette.mean(axis=0),
            yerr=avg_silhouette.std(axis=0),
        )
        ax[0, 0].set_xlabel("N. of clusters")
        ax[0, 0].set_ylabel("Average silhouette")
    ax[0, 1].errorbar(
        np.arange(smallest, largest + 1),
        pred_strength_.mean(axis=0),
        yerr=pred_strength_.std(axis=0),
    )
    ax[0, 1].set_xlabel("N. of clusters")
    ax[0, 1].set_ylabel("Prediction strength")
    ax[1, 0].errorbar(
        np.arange(smallest, largest + 1),
        davies_bouldin_.mean(axis=0),
        yerr=davies_bouldin_.std(axis=0),
    )
    ax[1, 0].set_xlabel("N. of clusters")
    ax[1, 0].set_ylabel("Davies-Bouldin Score")
    ax[1, 1].errorbar(
        np.arange(smallest, largest + 1),
        calinski_harabasz_.mean(axis=0),
        yerr=calinski_harabasz_.std(axis=0),
    )
    ax[1, 1].set_xlabel("N. of clusters")
    ax[1, 1].set_ylabel("Calinski_Harabasz Score")
    return fig


def plot_kdeplot(classified, save=False, filename=None, only_hab=False):
    cmap = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[5:, :-1])
    if only_hab:

        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        classified = classified.astype({"cluster_index": "int"})
        sns.kdeplot(
            x="Population - Habitants",
            hue="cluster_index",
            data=classified,
            log_scale=True,
            common_norm=False,
            palette=cmap,
        )
        ax.set_title("Population - Habitants")
        sns.despine(fig)
    else:
        include_zero = [
            "train_station",
            "bank",
            "hospital",
            "official_gov",
            "consumption",
            "supermarket",
            "airport_filtered",
            "prod_plant_50MW",
        ]

        fig, ax = plt.subplots(4, 4, figsize=(25, 18))
        axs = ax.flatten()
        for i, c in enumerate(classified.columns[2:]):
            ax = axs[i]
            if c in include_zero:
                classified[f"{c}_1"] = classified[c] + 1
                sns.kdeplot(
                    x=c + "_1",
                    hue="cluster_index",
                    data=classified,
                    log_scale=True,
                    common_norm=False,
                    ax=ax,
                    palette=cmap,
                )
                classified = classified.drop(f"{c}_1", axis=1)
            else:
                sns.kdeplot(
                    x=c,
                    hue="cluster_index",
                    data=classified,
                    log_scale=True,
                    common_norm=False,
                    ax=ax,
                    palette=cmap,
                )
            ax.set_title(c)
            sns.despine(fig)
    plt.tight_layout()
    if save:
        if filename is None:
            raise ValueError("filename must be provided")
        plt.savefig(os.path.join("../figure", f"{filename}.png"), dpi=150)
    return fig


def choose_clustering_algo(name_model="KMeans", metric="l2"):
    if name_model == "KMedoids":
        return KMedoids, dict(init="k-medoids++", max_iter=600, metric=metric)
    elif name_model == "KMeans":
        return KMeans, dict(init="k-means++", max_iter=600)
    elif name_model == "Spectral":
        affinity = "nearest_neighbors"
        n_neighbors = 30
        return SpectralClustering, dict(affinity=affinity, n_neighbors=n_neighbors)
    elif name_model == "GMM":
        return GaussianMixture, dict(max_iter=600)
    else:
        raise ValueError("Unknown model")


def circle_of_correlations(pc_infos, ebouli):
    # https://github.com/mazieres/analysis/blob/master/analysis.py#L19-34
    plt.Circle((0, 0), radius=10, color="g", fill=False)
    circle1 = plt.Circle((0, 0), radius=1, color="g", fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pc_infos["PC-0"])):
        x = pc_infos["PC-0"][idx]
        y = pc_infos["PC-1"][idx]
        plt.plot([0.0, x], [0.0, y], "k-")
        plt.plot(x, y, "rx")
        plt.annotate(pc_infos.index[idx], xy=(x, y))
    plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.title("Circle of Correlations")


def my_PCA(df, n_components, clusters=None):
    # https://github.com/mazieres/analysis/blob/master/analysis.py#L19-34
    # Normalize data
    df_norm = (df - df.mean()) / df.std()
    # PCA
    pca = PCA(n_components=n_components)
    pca_res = pca.fit_transform(df_norm.values)
    sing_val = pd.Series(pca.explained_variance_ratio_)
    sing_val.plot(kind="bar", title="Singular values")
    plt.show()
    # Circle of correlations
    # http://stackoverflow.com/a/22996786/1565438
    coef = np.transpose(pca.components_)  # n_features x n_components
    cols = [f"PC-{x}" for x in range(len(sing_val))]
    pc_infos = pd.DataFrame(coef, columns=cols, index=df_norm.columns)
    circle_of_correlations(pc_infos, sing_val)
    plt.show()
    # Plot PCA
    dat = pd.DataFrame(pca_res, columns=cols)
    if isinstance(clusters, np.ndarray):
        for clust in set(clusters):
            colors = list("bgrcmyk")
            plt.scatter(
                dat["PC-0"][clusters == clust],
                dat["PC-1"][clusters == clust],
                c=colors[clust],
            )
    else:
        plt.scatter(dat["PC-0"], dat["PC-1"])
    plt.xlabel("PC-0 (%s%%)" % str(sing_val[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(sing_val[1])[:4].lstrip("0."))
    plt.title("PCA")
    plt.show()
    return pc_infos, sing_val
