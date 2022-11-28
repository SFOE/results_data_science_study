import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import geopandas as gpd
import folium
import statsmodels.api as sm
from bs4 import BeautifulSoup
from settings import data_folder, preprocessed_folder


def set_plotting():
    plt.rcParams["figure.figsize"] = (12, 6)
    sns.set(font_scale=1.5)


def load_database_init():
    renaming_dict = {
        "Status (FIX=ausbezahlt/abgeschlossen, PROV=verpflichtet/in Auszahlung, SIST=sistiert)": "Status"
    }
    types_column = {
        "Kanton": "category",
        "Status (FIX=ausbezahlt/abgeschlossen, PROV=verpflichtet/in Auszahlung, SIST=sistiert)": "category",
        "EGID": "category",
        "Ort": "category",
        "PLZ": int,
        "UID": "category",
        "Baubewilligungsjahr": "Int64",
        "Nr. HFM 2015": str,
        "Verpflichteter Beitrag": float,
        "Jahr Verpflichtung": "Int64",
        "Auszahlung 1 (CHF)": float,
        "Jahr Auszahlung 1": "Int64",
        "Auszahlung 2 (CHF)": float,
        "Jahr Auszahlung 2": "Int64",
        "Auszahlung 3 (CHF)": float,
        "Jahr Auszahlung 3": "Int64",
        "Auszahlung 4 (CHF)": float,
        "Jahr Auszahlung 4": "Int64",
        "Auszahlung 5 (CHF)": float,
        "Jahr Auszahlung 5": "Int64",
        "Auszahlung 6 (CHF)": float,
        "Jahr Auszahlung 6": "Int64",
        "Energiebezugsfläche": float,
        "Hauptheizsystem vor Massnahme": "category",
        "Hauptheizsystem nach Massnahme": "category",
        "Wärmegedämmte Fläche Fassade": "Int64",
        "Wärmegedämmte Fläche Dach": "Int64",
        "Wärmegedämmte Fläche Wand und Boden gegen Erdreich": "Int64",
        "Anzahl Stückholz-/Pelletsfeuerungen mit Tagesbehälter": "Int64",
        "Thermische Nennleistung": float,
        "Anzahl Klassen/Stufen, um die das Gebäude verbessert wurde": "Int64",
        "Erreichter Sanierungsstandard": str,
        "Gültigkeitsdauer": "Int64",
        "Wirkungsdauer in Jahren": "Int64",
        "Gesamtinvestition": float,
        "Mehrinvestition ggü. nicht energetische Massnahme (Instandhaltung)": float,
        "Mehrinvestition ggü. Referenz HFM 2015": float,
        "Energiewirkung über die Massnahmenlebensdauer ggü. nicht energetischer Massnahme (Instandhaltung) (MWh)": float,
        "Energiewirkung über die Massnahmenlebensdauer ggü. Referenz HFM 2015 (MWh)": float,
        "CO2-Wirkung über die Massnahmenlebensdauer ggü. nicht energetischer Massnahme (Instandhaltung) (t CO2)": float,
        "CO2-Wirkung über ie Massnahmenlebensdauer ggü. Referenz HFM 2015 (t CO2)": float,
    }
    db = pd.read_csv(
        os.path.join(data_folder, "2022-07-08-09-50-40-db-rohdaten_(1)_v2_extract.csv"),
        skiprows=1,
        sep=";",
        encoding="iso-8859-1",
        dtype=types_column,
    ).rename(columns=renaming_dict)
    return db


def load_database():
    types_column = {
        "Kanton": "category",
        "EGID": "category",
        "Ort": "category",
        "PLZ": int,
        "Baubewilligungsjahr": "Int64",
        "Nr. HFM 2015": str,
        "Verpflichteter Beitrag": float,
        "Auszahlung 1 (CHF)": float,
        "Jahr Auszahlung 1": "Int64",
        "Hauptheizsystem vor Massnahme": "category",
        "Hauptheizsystem nach Massnahme": "category",
        "Wärmegedämmte Fläche Fassade": "Int64",
        "Wärmegedämmte Fläche Dach": "Int64",
        "Wärmegedämmte Fläche Wand und Boden gegen Erdreich": "Int64",
        "Anzahl Stückholz-/Pelletsfeuerungen mit Tagesbehälter": "Int64",
        "Thermische Nennleistung": float,
        "Wirkungsdauer in Jahren": "Int64",
        "Gesamtinvestition": float,
        "Mehrinvestition ggü. nicht energetische Massnahme (Instandhaltung)": float,
        "Mehrinvestition ggü. Referenz HFM 2015": float,
        "Energiewirkung über die Massnahmenlebensdauer ggü. nicht energetischer Massnahme (Instandhaltung) (MWh)": float,
        "Energiewirkung über die Massnahmenlebensdauer ggü. Referenz HFM 2015 (MWh)": float,
        "CO2-Wirkung über die Massnahmenlebensdauer ggü. nicht energetischer Massnahme (Instandhaltung) (t CO2)": float,
        "CO2-Wirkung über ie Massnahmenlebensdauer ggü. Referenz HFM 2015 (t CO2)": float,
    }
    use_columns = [
        "Kanton",
        "EGID",
        "Ort",
        "PLZ",
        "Baubewilligungsjahr",
        "Nr. HFM 2015",
        "Regions-ID",
        "Verpflichteter Beitrag",
        "Auszahlung 1 (CHF)",
        "Jahr Auszahlung 1",
        "Hauptheizsystem vor Massnahme",
        "Hauptheizsystem nach Massnahme",
        "Wärmegedämmte Fläche Fassade",
        "Wärmegedämmte Fläche Dach",
        "Wärmegedämmte Fläche Wand und Boden gegen Erdreich",
        "Anzahl Stückholz-/Pelletsfeuerungen mit Tagesbehälter",
        "Thermische Nennleistung",
        "Wirkungsdauer in Jahren",
        "Gesamtinvestition",
        "Mehrinvestition ggü. nicht energetische Massnahme (Instandhaltung)",
        "Mehrinvestition ggü. Referenz HFM 2015",
        "Energiewirkung über die Massnahmenlebensdauer ggü. nicht energetischer Massnahme (Instandhaltung) (MWh)",
        "Energiewirkung über die Massnahmenlebensdauer ggü. Referenz HFM 2015 (MWh)",
        "CO2-Wirkung über die Massnahmenlebensdauer ggü. nicht energetischer Massnahme (Instandhaltung) (t CO2)",
        "CO2-Wirkung über die Massnahmenlebensdauer ggü. Referenz HFM 2015 (t CO2)",
        "Alpine",
        "Typology",
    ]
    db_with_terrain_class = pd.read_csv(
        os.path.join(preprocessed_folder, "db_with_terrain_class.csv"),
        usecols=use_columns,
        dtype=types_column,
    )
    return db_with_terrain_class


def combined_rows_db(db, col):
    prop_hab = [
        "Population - Taux brut de nuptialité",
        "Population - Taux brut de divortialité",
        "Population - Taux brut de natalité",
        "Population - Taux brut de mortalité",
        "Population - Part du groupe d'âge 0-19 ans",
        "Population - Part du groupe d'âge 20-64 ans",
        "Population - Part du groupe d'âge 65+ ans",
        "Population - Etrangers",
        "Construction, logement - Nouveaux logements construits",
    ]

    prop_hab_var = ["Population - Variation en %"]
    prop_beneficiaire = ["Taux d'aide sociale"]
    prop_surface = ["Population - Densité de la population"]
    prop_surface_hab = ["Surface - Surfaces d'habitat et d'infrastructure, variation"]
    prop_surface_agricole = ["Surface - Surfaces agricoles, variation"]
    prop_nb_contr = ["Revenu imposable par contribuable, en francs"]
    prop_hab_1 = ["Revenu imposable par habitant/-e, en francs"]
    prop_nb_menage = ["Population - Taille moyenne des ménages"]

    db["Revenu_nb_contribuable"] = (
        db["Revenu imposable, en mio. de francs"]
        / db["Revenu imposable par contribuable, en francs"]
        * 1e6
    )
    db["Revenu_nb_habitant"] = (
        db["Revenu imposable, en mio. de francs"]
        / db["Revenu imposable par habitant/-e, en francs"]
        * 1e6
    )

    db["hab_old"] = db["Population - Habitants"] / (
        1 + db["Population - Variation en %"] / 100
    )  # i.e. (current_pop-old)/old*100=variation -> old=current_pop/(1+variation/100)
    db["surf_hab_old"] = db["Surface - Surfaces d'habitat et d'infrastructure"] / (
        1 + db["Surface - Surfaces d'habitat et d'infrastructure, variation"] / 100
    )
    db["surf_agr_old"] = db["Surface - Surfaces agricoles"] / (
        1 + db["Surface - Surfaces agricoles, variation"] / 100
    )

    agg_dict = {}

    def weight_sum(x, name_col):
        w = db.loc[x.index, name_col]
        if np.sum(w) == 0:
            w = None
        return np.average(x, weights=w)

    for columns in db.columns:

        if columns in [
            "BFS_NUMMER",
            "Construction, logement - Taux de logements vacants",
        ]:
            # todo find the total number of accomodations
            continue
        elif columns == "REGION":
            agg_dict[columns] = lambda x: " / ".join(x)
        elif columns in prop_hab:
            agg_dict[columns] = lambda x: weight_sum(x, "Population - Habitants")
        elif columns in prop_beneficiaire:
            agg_dict[columns] = lambda x: weight_sum(x, "Nombre de bénéficiaires")
        elif columns in prop_surface:
            agg_dict[columns] = lambda x: weight_sum(x, "Surface - Surface, total")
        elif columns in prop_surface_hab:
            agg_dict[columns] = lambda x: weight_sum(x, "surf_hab_old")
        elif columns in prop_surface_agricole:
            agg_dict[columns] = lambda x: weight_sum(x, "surf_agr_old")
        elif columns in prop_nb_contr:
            agg_dict[columns] = lambda x: weight_sum(x, "Revenu_nb_contribuable")
        elif columns in prop_hab_1:
            agg_dict[columns] = lambda x: weight_sum(x, "Revenu_nb_habitant")
        elif columns in prop_nb_menage:
            agg_dict[columns] = lambda x: weight_sum(x, "Population - Ménages privés")
        elif columns in prop_hab_var:
            agg_dict[columns] = lambda x: weight_sum(x, "hab_old")
        else:
            agg_dict[columns] = "sum"
    return db.groupby(col).agg(agg_dict).reset_index()


def get_code_translation_regbl():
    name = pd.read_csv(
        os.path.join(data_folder, "ch", "kodes_codes_codici.csv"),
        dtype={"CECODID": "category"},
        usecols=["CECODID", "CMERKM", "CODTXTLF", "CODTXTKF", "CEXPDAT"],
        sep="\t",
    ).set_index(["CECODID", "CMERKM"])
    return name


def read_xlsx_from_atlas(path, nrows=2212):
    df = pd.read_excel(path, header=[2, 3], skiprows=[4], nrows=nrows, na_values="X")
    df.columns = [
        df.columns[i][1] if i in [0, 1] else df.columns[i][0]
        for i in range(len(df.columns))
    ]
    df = df.astype({"Regions-ID": "category"}).set_index("Regions-ID")
    return df


def get_mapping_commune():
    # Manually tracking the municipalities that are merged together
    # dict-like file indicating the changes in the GDE code of newly merged municipalities
    mapping_commune = pd.read_csv(
        os.path.join(data_folder, "mapping_commune.csv")
    ).astype({"From": "category", "To": "category"})
    return mapping_commune


def mapping_com(df, rule_merging=None):
    mapping_commune = get_mapping_commune()
    for i, row in mapping_commune.iterrows():
        if row["From"] in df.index:
            if row["To"] in df.index:
                # add to the current municipality
                to = df.loc[row["To"], df.columns[1:]]
                from_ = df.loc[row["From"], df.columns[1:]]
                if not (to == from_).all():
                    df.loc[row["To"], :] = rule_merging(
                        df.loc[row["To"], :], df.loc[row["From"], :]
                    )
            else:
                # if creation of a new municipality, we add a new row
                df.loc[row["To"], :] = df.loc[row["From"], :]
            # remove old municipalities
            df = df.drop(row["From"], axis=0)
    return df


def prepare_regbl():

    name = get_code_translation_regbl()
    columns_to_use = [
        "EGID",
        "GDEKT",
        "GGDENR",
        "GGDENAME",
        "GKODE",
        "GKODN",
        "GSTAT",
        "GKAT",
        "GKLAS",
        "GBAUJ",
        "GBAUP",
        "GABBJ",
        "GAREA",
        "GVOL",
        "GVOLNORM",
        "GASTW",
        "GANZWHG",
        "GAZZI",
        "GSCHUTZR",
        "GEBF",
        "GWAERZH1",
        "GENH1",
        "GWAERDATH1",
        "GWAERZH2",
        "GENH2",
        "GWAERDATH2",
        "GWAERZW1",
        "GENW1",
        "GWAERDATW1",
        "GWAERZW2",
        "GENW2",
        "GWAERDATW2",
    ]
    dtype = {
        "EGID": int,
        "GDEKT": "category",
        "GGDENR": "category",
        "GGDENAME": "category",
        "GKODE": float,
        "GKODN": float,
        "GSTAT": "category",
        "GKAT": "category",
        "GKLAS": "category",
        "GBAUJ": float,
        "GBAUP": "category",
        "GABBJ": float,
        "GAREA": float,
        "GVOL": float,
        "GVOLNORM": "category",
        "GASTW": float,
        "GANZWHG": float,
        "GAZZI": float,
        "GSCHUTZR": float,
        "GEBF": float,
        "GWAERZH1": "category",
        "GENH1": "category",
        "GWAERZH2": "category",
        "GENH2": "category",
        "GWAERZW1": "category",
        "GENW1": "category",
        "GWAERZW2": "category",
        "GENW2": "category",
    }
    regbl = pd.read_csv(
        os.path.join(data_folder, "ch", "gebaeude_batiment_edificio.csv"),
        sep="\t",
        usecols=columns_to_use,
        dtype=dtype,
        parse_dates=["GWAERDATH1", "GWAERDATH2", "GWAERDATW1", "GWAERDATW2"],
    )

    for x in regbl.columns:
        try:
            rename_codes = name.xs(x, level=1, drop_level=True).CODTXTKF.to_dict()
        except KeyError:
            continue
        regbl.replace({x: rename_codes}, inplace=True)
    trad = {
        "GDEKT": "Canton",
        "GGDENR": "BFS_NUMBER",
        "GGDENAME": "Name",
        "GKODE": "Coord_x",
        "GKODN": "Coord_y",
        "GSTAT": "Statut_bat",
        "GKAT": "Cat_bat",
        "GKLAS": "Classe_bat",
        "GBAUJ": "Annee_constr",
        "GBAUP": "Epoque_constr",
        "GABBJ": "Annee_destr",
        "GAREA": "Superficie",
        "GVOL": "Volume",
        "GVOLNORM": "Norme_vol",
        "GASTW": "Nb_etage",
        "GANZWHG": "Nb_habitation",
        "GAZZI": "Nb_loc_hab",
        "GSCHUTZR": "Protection_civile",
        "GEBF": "Superficie_energ",
        "GWAERZH1": "Gen1",
        "GENH1": "Source_energie_Gen1",
        "GWAERDATH1": "Date_Gen1",
        "GWAERZH2": "Gen2",
        "GENH2": "Source_energie_Gen2",
        "GWAERDATH2": "Date_Gen2",
        "GWAERZW1": "Eau1",
        "GENW1": "Source_energie_Eau1",
        "GWAERDATW1": "Date_Eau1",
        "GWAERZW2": "Eau2",
        "GENW2": "Source_energie_Eau2",
        "GWAERDATW2": "Date_Eau2",
    }
    regbl.rename(
        columns=trad,
        inplace=True,
    )
    return regbl


def tensor_index_canton_year(df, possible_year, measure):
    name_abbr = get_name_abbr()
    cantons = name_abbr["Kanton"].unique()
    index = (
        pd.MultiIndex.from_product(
            [
                possible_year,
                cantons,
            ],
            names=["Jahr Auszahlung 1", "Kanton"],
        )
        .to_frame()
        .reset_index(drop=True)
    )
    sub = df[df["Nr. HFM 2015"] == measure] if measure is not None else df
    tensor_product = sub.set_index(["Jahr Auszahlung 1", "Kanton"]).merge(
        index,
        left_index=True,
        right_on=["Jahr Auszahlung 1", "Kanton"],
        how="outer",
    )
    return tensor_product


def precentage_renovated_rooms(measure, db, year, cumul=True):
    if cumul:
        db = db[
            (db["Jahr Auszahlung 1"].astype(float) <= year)
            | db["Jahr Auszahlung 1"].isnull()
        ].copy()
    else:
        db = db[
            (db["Jahr Auszahlung 1"].astype(float) == year)
            | db["Jahr Auszahlung 1"].isnull()
        ].copy()
    nb_renovated_rooms = db.groupby(["Canton", measure]).WAZIM.sum().reset_index()
    nb_renovated_rooms = pd.pivot_table(
        nb_renovated_rooms, "WAZIM", "Canton", measure
    ).rename(columns={True: "Renovated", False: "Not renovated"})
    ratio = (
        nb_renovated_rooms["Renovated"]
        / (nb_renovated_rooms["Renovated"] + nb_renovated_rooms["Not renovated"])
        * 100
    )
    ratio = ratio.to_frame("renov_rooms")
    ratio["year"] = year
    return ratio


def explode_db(db):
    # repeat some rows since some values of the EGID column of the db_with_terrain_class database contains multiple EGID in the same cell.
    db = db.copy().astype({"EGID": str})
    db["EGID"] = db.EGID.str.split(";")
    db = db.explode("EGID").replace({"EGID": {"": np.nan, "191 792 494": np.nan}})
    db = db.astype({"EGID": "float"}).astype({"EGID": "Int64"})
    return db


def add_renov_indicator(combined_regbl):
    combined_regbl["renov"] = ~combined_regbl.PLZ.isnull()
    combined_regbl["Envelope_renov"] = combined_regbl["Nr. HFM 2015"].isin(["M-01"])
    combined_regbl["Heating_renov"] = combined_regbl["Nr. HFM 2015"].isin(
        ["M-02", "M-03", "M-04", "M-05", "M-06", "M-07"]
    )
    combined_regbl.loc[
        combined_regbl["Nr. HFM 2015"].isnull(), "Nr. HFM 2015"
    ] = "no_measure"
    combined_regbl["ones"] = 1
    pivoted = (
        combined_regbl[["Nr. HFM 2015", "ones"]]
        .pivot(columns="Nr. HFM 2015", values="ones")
        .fillna(0)
    )
    combined_regbl = pd.concat((combined_regbl, pivoted), axis=1)
    return combined_regbl


def add_alpin_topo(db, on="BFS_NUMBER"):
    db = db.astype({on: "Int64"})
    alpin = pd.read_csv(
        os.path.join(preprocessed_folder, "alpin_fusion_com.csv"), dtype={"Regions-ID": "Int64"}
    )
    typology = pd.read_csv(
        os.path.join(preprocessed_folder, "typology_fusion_com.csv"),
        dtype={"Regions-ID": "Int64"},
    )
    return pd.merge(
        pd.merge(db, typology, left_on=on, right_on="Regions-ID"),
        alpin,
        left_on=on,
        right_on="Regions-ID",
    ).drop(columns=["Regions-ID_x", "Regions-ID_y"])


def get_economic_features(level="commune"):
    assert level in ["commune", "CH"]

    # load socio economic features and national average
    if level == "commune":
        socio_features = pd.read_csv(os.path.join(preprocessed_folder, "socio_economic.csv"))
        socio_features = socio_features.astype({"BFS_NUMMER": int})
    else:
        socio_features = pd.read_csv(os.path.join(data_folder, "combined.csv"))
        socio_features = socio_features[socio_features.BFS_NUMMER == "CH"].copy()
    return socio_features


def add_socio_economic_features(db):
    socio_features = get_economic_features("commune")
    combined = pd.merge(db, socio_features, left_on="Regions-ID", right_on="BFS_NUMMER")
    return combined


def build_design_matrix(X, mean):
    X = pd.get_dummies(X, drop_first=True)
    # normalize data by the population size (%)
    X[
        [
            "Economie - Emplois dans le secteur primaire",
            "Economie - Emplois dans le secteur secondaire",
            "Economie - Emplois dans le secteur tertiaire",
        ]
    ] = X[
        [
            "Economie - Emplois dans le secteur primaire",
            "Economie - Emplois dans le secteur secondaire",
            "Economie - Emplois dans le secteur tertiaire",
        ]
    ].div(
        X["Population - Habitants"], axis=0
    )
    X = X.drop(columns="Population - Habitants")

    # center the data with the national average
    mean[["Alpine_Yes", "Typology_Rural", "Typology_Urban"]] = 0
    mean[
        [
            "Economie - Emplois dans le secteur primaire",
            "Economie - Emplois dans le secteur secondaire",
            "Economie - Emplois dans le secteur tertiaire",
        ]
    ] = mean[
        [
            "Economie - Emplois dans le secteur primaire",
            "Economie - Emplois dans le secteur secondaire",
            "Economie - Emplois dans le secteur tertiaire",
        ]
    ].div(
        mean["Population - Habitants"], axis=0
    )
    mean = mean.drop(columns="Population - Habitants")
    X = X - mean.to_numpy().reshape(1, -1)

    # standardize data
    X = X / X.to_numpy().std(axis=0, keepdims=True)

    X = sm.add_constant(X, prepend=True)
    return X


def plot_param_glm(res, name_param):
    results_df = pd.concat((res.params, res.bse), axis=1).drop("const").reset_index()
    results_df = results_df.rename(
        columns={"index": "Variable", 0: "Parameter value", 1: "Std. err."}
    )
    plt.errorbar(
        results_df["Parameter value"],
        results_df["Variable"],
        fmt="o",
        xerr=2 * results_df["Std. err."],
        linestyle=None,
    )
    plt.gca().set_yticklabels(name_param)
    # plt.xticks(rotation=40)
    plt.axvline(0, linestyle="dashed", color="red")


def save(file, fig=None, tight_layout=True):
    if tight_layout:
        plt.gcf().tight_layout()
    if fig is None:
        plt.savefig(os.path.join("../figure", file + ".png"), dpi=150)
    else:
        fig.savefig(os.path.join("../figure", file + ".png"), dpi=150)


def set_ylim(ax, ymax, ymin=0, seaborn=False):
    if not seaborn:
        ax.set_ylim(ymin, ymax)
    else:
        for col_val, a in ax.axes_dict.items():
            a.set_ylim(ymin, ymax)


def _set_size_label(sup, func, msg, size):
    func = getattr(sup, func)
    if size is None:
        func(msg)
    else:
        func(msg, fontdict={"fontsize": size})


def _set_log_scale(ax, nticks, offset, x):
    if x:
        if ax is None:
            ax = plt.gca()
        ax.set_xscale("log")
        xmin, xmax = ax.get_xlim()
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks(
            (np.logspace(np.log10(xmin) + offset, np.log10(xmax), nticks) / 10).round()
            * 10
        )
    else:
        if ax is None:
            ax = plt.gca()
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yticks(
            (np.logspace(np.log10(ymin) + offset, np.log10(ymax), nticks) / 10).round()
            * 10
        )


def set_log_xscale(nticks, ax=None, offset=0):
    _set_log_scale(ax, nticks, offset, x=True)


def set_log_yscale(nticks, ax=None, offset=0):
    _set_log_scale(ax, nticks, offset, x=False)


def _set_size_title(sup, func, msg, size):
    func = getattr(sup, func)
    if size is None:
        func(msg)
    else:
        func(msg, fontdict={"fontsize": size})


def set_title(label, size=None, ax=None):
    if ax is not None:
        _set_size_title(ax, "set_title", label, size)
    else:
        _set_size_title(plt, "title", label, size)


def set_label(xlabel=None, ylabel=None, size=None, ax=None):
    if xlabel is not None:
        if ax is not None:
            _set_size_label(ax, "set_xlabel", xlabel, size)
        else:
            _set_size_label(plt, "xlabel", xlabel, size)
    if ylabel is not None:
        if ax is not None:
            _set_size_label(ax, "set_ylabel", ylabel, size)
        else:
            _set_size_label(plt, "ylabel", ylabel, size)


def get_nb_hab_per_region():
    # get number of inhabitants/region
    alpin = pd.read_csv(os.path.join(preprocessed_folder, "alpin_fusion_com.csv"))
    typology = pd.read_csv(os.path.join(preprocessed_folder, "typology_fusion_com.csv"))
    socio_economic = pd.read_csv(
        os.path.join(preprocessed_folder, "socio_economic.csv"),
        usecols=["Population - Habitants", "BFS_NUMMER"],
    )
    hab_alpin_typ = pd.merge(
        socio_economic, alpin, left_on="BFS_NUMMER", right_on="Regions-ID"
    )
    hab_alpin_typ = pd.merge(
        hab_alpin_typ, typology, left_on="BFS_NUMMER", right_on="Regions-ID"
    )
    # group per region and sum
    hab_alpin_typ_sum = hab_alpin_typ.groupby(["Alpine", "Typology"])[
        "Population - Habitants"
    ].sum()
    return hab_alpin_typ_sum


def get_nb_hab_per_canton():
    # load population in each canton
    nb_hab_per_canton = (
        pd.read_excel(
            os.path.join(data_folder, "stat_par_canton.xlsx"),
            skiprows=[0, 1, 2, 4, 5, 6],
            nrows=1,
            usecols=range(3, 29),
        )
        .rename({0: "Pop"})
        .T
        * 1000
    )
    nb_hab_per_canton = nb_hab_per_canton.reset_index().rename(
        columns={"index": "Kanton"}
    )
    return nb_hab_per_canton


def merge_swiss_map(df, how="outer"):
    # import swiss map with boundaries
    swiss_map = gpd.read_file(
        os.path.join(data_folder, "SHAPEFILE_LV95_LN02-2", "group_canton.shp")
    )
    name_abbr = get_name_abbr()
    swiss_map = pd.merge(swiss_map, name_abbr, on="NAME")
    return pd.merge(swiss_map, df, on="Kanton", how=how)


def get_name_abbr():
    return pd.read_csv(os.path.join(data_folder, "name_abbr.csv"))


def save_map(file, map):
    if isinstance(map, folium.folium.Map):
        map.save(os.path.join("../figure", f"{file}.html"))
    else:
        with open(os.path.join("../figure", f"{file}.html"), "w") as file:
            file.write(str(map))


def add_legend_map(fig, legend):
    soup = BeautifulSoup(fig.to_html(), features="html.parser")
    added_html = BeautifulSoup(
        f"""
            <div style="position: fixed;
                    left: 50%;
                    top: 50px;
                    transform: translate(-50%, -50%);
                    margin: 0 auto; 
                    padding: 10px;
                background-color:white; border:2px solid grey;z-index: 900;">
                <h5>{legend}</h5>
            </div>
            """,
        features="html.parser",
    )
    divtag = soup.find_all("div", class_="plotly-graph-div")[0]
    divtag.insert_after(added_html.div)
    return soup
