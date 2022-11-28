from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def make_correction_imports(import_df_tot):

    # -----------------------------------------------------------------------------------------
    # Germany
    # -----------------------------------------------------------------------------------------
    germany = "Allemagne (jusqu'en 1990, ancien territoire de la RFA)"
    import_df_tot_sub = import_df_tot.loc[germany].copy()
    # Impute values
    wrong_years = [2016, 2017, 2018, 2019]
    correct_years = [2010, 2011, 2012, 2013, 2014, 2015, 2020]
    df = import_df_tot_sub[import_df_tot_sub["Year"].isin(correct_years)]
    df = df[["Year", "Non spécifié"]]
    poly = PolynomialFeatures(3)
    X = poly.fit_transform(df["Year"].to_numpy().reshape(-1, 1))
    # Fit a linear model and predict on 2016-2019
    reg = LinearRegression().fit(X, df["Non spécifié"])
    X = poly.fit_transform(np.array(wrong_years).reshape(-1, 1))
    val = reg.predict(X)
    wrong_data = import_df_tot_sub[import_df_tot_sub["Year"].isin(wrong_years)].copy()
    # Move the predicted quantity from "Non spécifié" to "Pays-Bas"
    wrong_data.loc[:, "Pays-Bas"] = wrong_data.loc[:, "Non spécifié"] - val
    wrong_data.loc[:, "Non spécifié"] = val
    import_df_tot_sub[import_df_tot_sub["Year"].isin(wrong_years)] = wrong_data
    import_df_tot.loc[germany] = import_df_tot_sub

    # -----------------------------------------------------------------------------------------
    # Austria
    # -----------------------------------------------------------------------------------------

    import_df_tot_sub = import_df_tot.loc["Autriche"].copy()
    wrong_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    correct_years = [2010, 2011, 2012, 2013]
    # Impute values
    df = import_df_tot_sub[import_df_tot_sub["Year"].isin(correct_years)]
    df = df[["Year", "Non spécifié"]]
    poly = PolynomialFeatures(1)
    X = poly.fit_transform(df["Year"].to_numpy().reshape(-1, 1))
    # Fit a linear model and predict on 2014-2020
    reg = LinearRegression().fit(X, df["Non spécifié"])
    X = poly.fit_transform(np.array(wrong_years).reshape(-1, 1))
    val = reg.predict(X)
    wrong_data = import_df_tot_sub[import_df_tot_sub["Year"].isin(wrong_years)].copy()
    # Move the predicted quantity from "Non spécifié" to "Russie"
    wrong_data.loc[:, "Russie"] = wrong_data.loc[:, "Non spécifié"] - val
    wrong_data.loc[:, "Non spécifié"] = val
    import_df_tot_sub[import_df_tot_sub["Year"].isin(wrong_years)] = wrong_data
    import_df_tot.loc["Autriche"] = import_df_tot_sub

    # -----------------------------------------------------------------------------------------
    # Slovakia
    # -----------------------------------------------------------------------------------------
    import_df_tot_sub = import_df_tot.loc["Slovaquie"].copy()
    import_df_tot_sub.loc[:, "Russie"] += import_df_tot_sub.loc[:, "Non spécifié"]
    import_df_tot_sub.loc[:, "Non spécifié"] = 0
    import_df_tot.loc["Slovaquie"] = import_df_tot_sub

    # -----------------------------------------------------------------------------------------
    # Luxembourg
    # -----------------------------------------------------------------------------------------

    import_df_tot_sub = import_df_tot.loc["Luxembourg"].copy()
    wrong_years = [2017, 2018, 2019, 2020]
    correct_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
    # Impute values
    df = import_df_tot_sub[import_df_tot_sub["Year"].isin(correct_years)]
    df = df[["Year", "Non spécifié"]]
    poly = PolynomialFeatures(3)
    X = poly.fit_transform(df["Year"].to_numpy().reshape(-1, 1))
    # Fit a linear model and predict on 2017-2020
    reg = LinearRegression().fit(X, df["Non spécifié"])
    X = poly.fit_transform(np.array(wrong_years).reshape(-1, 1))
    val = reg.predict(X)
    wrong_data = import_df_tot_sub[import_df_tot_sub["Year"].isin(wrong_years)].copy()
    # Move the predicted quantity from "Non spécifié" to "Norvège"
    wrong_data.loc[:, "Norvège"] = wrong_data.loc[:, "Non spécifié"] - val
    wrong_data.loc[:, "Non spécifié"] = val
    import_df_tot_sub[import_df_tot_sub["Year"].isin(wrong_years)] = wrong_data
    import_df_tot.loc["Luxembourg"] = import_df_tot_sub

    return import_df_tot
