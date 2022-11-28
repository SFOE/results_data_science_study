import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd

from settings import data_folder, preprocessed_folder


matching_names_european = {
    "Suisse": "Switzerland",
    "Belgique": "Belgium",
    "Bulgarie": "Bulgaria",
    "Tchéquie": "Czech Republic",
    "Danemark": "Denmark",
    "Allemagne (jusqu'en 1990, ancien territoire de la RFA)": "Germany",
    "Estonie": "Estonia",
    "Irlande": "Ireland",
    "Grèce": "Greece",
    "Espagne": "Spain",
    "Croatie": "Croatia",
    "Italie": "Italy",
    "Lettonie": "Latvia",
    "Lituanie": "Lithuania",
    "Hongrie": "Hungary",
    "Pays-Bas": "Netherlands",
    "Autriche": "Austria",
    "Pologne": "Poland",
    "Roumanie": "Romania",
    "Slovénie": "Slovenia",
    "Slovaquie": "Slovakia",
    "Finlande": "Finland",
    "Suède": "Sweden",
    "Norvège": "Norway",
    "Royaume-Uni": "United Kingdom",
    "Macédoine du Nord": "North Macedonia",
    "Serbie": "Serbia",
    "Turquie": "Turkey",
    "Moldavie": "Moldova",
}

matching_names_non_euro = {
    "Afrique du Sud": "South Africa",
    "Albanie": "Albania",
    "Algérie": "Algeria",
    "Arabie Saoudite": "Saudi Arabia",
    "Argentine": "Argentina",
    "Azerbaïdjan": "Azerbaijan",
    "Cameroun": "Cameroon",
    "Gibraltar (UK)": "Gibraltar",
    "Guinée équatoriale": "Equatorial Guinea",
    "Libye": "Libya",
    "Malaisie": "Malaysia",
    "Malte": "Malta",
    "Nigéria": "Nigeria",
    "Non spécifié": "Other",
    "Ouzbékistan": "Uzbekistan",
    "Pérou": "Peru",
    "Russie": "Russia",
    "République dominicaine": "Dominican Republic",
    "Singapour": "Singapore",
    "Trinité-et-Tobago": "Trinidad and Tobago",
    "Turkménistan": "Turkmenistan",
    "Yémen": "Yemen",
    "Égypte": "Egypt",
    "Émirats arabes unis": "United Arab Emirates",
    "Équateur": "Ecuador",
    "États-Unis": "United States",
}


def load_stocks():
    stock = pd.read_csv(os.path.join(preprocessed_folder, "stocks.csv")).set_index(
        "Country_stock"
    )
    return stock


def load_production():
    production_euro_country = pd.read_csv(
        os.path.join(preprocessed_folder, "gas_prod.csv")
    ).set_index("Country")

    # Problem with the Ukrainian value of 2020
    # https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/energy-economics/statistical-review/bp-stats-review-2022-full-report.pdf
    prod_bp = pd.Series(
        [19.5, 19.4, 20.2, 20.2, 18.8, 19.0, 19.4, 19.7, 19.4, 19.1],
        index=[str(i) for i in range(2011, 2021)],
    )
    ratio = (production_euro_country.loc["Ukraine"].drop("2020") / prod_bp).mean()
    production_euro_country.loc["Ukraine", "2020"] = ratio * 19.1
    return production_euro_country


def add_stock_prod(stock, prod):
    # Add the stock to the production
    prod = stock + prod
    prod = pd.melt(
        prod, ignore_index=False, value_name="Production", var_name="Year"
    ).astype({"Year": int})
    return prod


def load_index(which):
    match_names = {
        "Russian Federation": "Russia",
        "Egypt, Arab Rep.": "Egypt",
        "Iran, Islamic Rep.": "Iran",
        "Yemen, Rep.": "Yemen",
        "Slovak Republic": "Slovakia",
    }
    cols = [str(i) for i in range(2010, 2021)]
    if which == "democracy":
        democracy_index_10_20 = pd.read_csv(
            os.path.join(preprocessed_folder, "democracy_index_10_20.csv"), index_col=0
        )
        return democracy_index_10_20
    elif which == "PoliticalStability":
        index = pd.read_csv(
            os.path.join(
                preprocessed_folder, "wgidataset_Political StabilityNoViolence.csv"
            ),
            index_col=0,
        )

    elif which == "ControlofCorruption":
        index = pd.read_csv(
            os.path.join(preprocessed_folder, "wgidataset_ControlofCorruption.csv"),
            index_col=0,
        )

    elif which == "GovernmentEffectiveness":
        index = pd.read_csv(
            os.path.join(preprocessed_folder, "wgidataset_GovernmentEffectiveness.csv"),
            index_col=0,
        )

    elif which == "RegulatoryQuality":
        index = pd.read_csv(
            os.path.join(preprocessed_folder, "wgidataset_RegulatoryQuality.csv"),
            index_col=0,
        )

    elif which == "RuleofLaw":
        index = pd.read_csv(
            os.path.join(preprocessed_folder, "wgidataset_RuleofLaw.csv"), index_col=0
        )

    elif which == "VoiceandAccountability":
        index = pd.read_csv(
            os.path.join(preprocessed_folder, "wgidataset_VoiceandAccountability.csv"),
            index_col=0,
        )
    else:
        raise KeyError(str(which))

    index = index[cols]
    index = index.rename(match_names)
    return index


def load_imports(which):
    if which == "origin":
        shares = pd.read_excel(
            os.path.join(data_folder, "gas_origin.xlsx"), sheet_name="origin", nrows=5
        ).set_index("Country")
        shares_10_20 = shares[[str(i) for i in range(2010, 2021)]]
        return shares_10_20
    elif which == "euro":
        import_df_tot = pd.read_csv(
            os.path.join(preprocessed_folder, "gas_import.csv"), index_col=0
        )
        import_df_tot = import_df_tot.drop(columns=["Total"])
        import_df_tot = (
            import_df_tot[import_df_tot.Type == "Gaz naturel"]
            .drop(columns="Type")
            .copy()
        )
        return import_df_tot
    else:
        raise ValueError("Which should be either origin or euro")


def load_exports_ch():
    """export_df = pd.read_csv(
        os.path.join(data_folder, "eurostat", "gas_export_to_ch.csv"),
        header=[0, 1],
        index_col=0,
    )
    # Drop countries that are not directly providing gas to switzerland
    export_df = export_df[export_df.sum(axis=1) > 0]["Gaz naturel"]
    # Data is missing prior to 2014
    export_df.loc["France", ["2010", "2011", "2012", "2013"]] = export_df.loc[
        "France", ["2014", "2015", "2016"]
    ].mean()"""
    # Use corrected imports
    export_df = pd.read_csv(
        os.path.join(preprocessed_folder, "export_to_ch.csv"),
    ).set_index("Year")
    return export_df


def risk(index):
    return 1 - index


def combine_imports_eu_export_ch(imports, exports):
    # Add switzerland to imports
    imports_swiss = exports.reset_index().copy()
    imports_swiss["Country_import"] = "Suisse"
    imports_swiss = imports_swiss.set_index("Country_import").astype({"Year": int})
    imports = pd.concat((imports_swiss, imports), axis=0).fillna(0)
    return imports


def compute_proportions(df):
    cols = df.columns.drop(["Year"])
    # Compute proportions
    import_df = df.copy()
    import_df[cols] = df[cols].div(df[cols].sum(axis=1), axis=0)
    return import_df


def _get_euro_countries(imports, Year):
    # European countries
    list_countries = imports.index.unique().to_list()
    # Select only european countries
    europe_system = imports.loc[list_countries, list_countries + ["Year"]].copy()
    A = europe_system[europe_system.Year == Year].drop(columns="Year").copy()
    # Renaming columns and indices
    A = A.rename(columns=matching_names_european).rename(matching_names_european)
    return A


def _get_non_euro_countries(imports, Year):
    list_countries = imports.index.unique().to_list()
    # Select only non-european countries
    not_europe = imports.columns[~imports.columns.isin(list_countries)]
    not_europe_system = imports.loc[:, not_europe].copy()
    # Remove columns full of zeros (no impact on the computation of the index)
    not_europe_system = not_europe_system.drop(
        columns=not_europe_system.columns[not_europe_system.sum(axis=0) == 0]
    )
    return not_europe_system


def compute_B(imports, Year):
    not_europe_system = _get_non_euro_countries(imports, Year)
    B = not_europe_system[not_europe_system.Year == Year].drop(columns="Year").copy()
    B = B.rename(columns=matching_names_non_euro)
    return B


def compute_A(imports, Year):
    A = _get_euro_countries(imports, Year)
    return A


def domestic_prod(imports, Year):
    A = _get_euro_countries(imports, Year)
    domestic_prod = np.diag(A)
    return domestic_prod


def compute_risk(
    A,
    B,
    domestic_prod,
    index,
    risk_domestic_prod=0,
    factor_euro=0.1,
    adjust=lambda x: x,
):
    # Voluntary disruption (non european countries)
    risk_noneuro_vol = risk(index.loc[B.columns])
    risk_noneuro_vol = adjust(risk_noneuro_vol)
    risk_noneuro_vol = risk_noneuro_vol.to_numpy().reshape(1, -1)
    # Total risk for voluntary disruption
    risk_non_europ = (B**2 * risk_noneuro_vol).sum(axis=1).to_numpy()
    A_num = A.to_numpy().copy()
    # Set diagonal to zero
    A_num[range(len(A)), range(len(A))] = 0
    # Voluntary disruption (european countries)
    risk_euro_vol = risk(index.loc[A.columns])
    risk_euro_vol = adjust(risk_euro_vol)
    risk_euro_vol = risk_euro_vol.to_numpy().reshape(1, -1)
    # Total risk for voluntary disruption
    risk_euro = factor_euro * (A_num**2 * risk_euro_vol).sum(axis=1)
    lhs = np.eye(A_num.shape[0]) - A_num**2
    rhs = domestic_prod**2 * risk_domestic_prod + risk_non_europ + risk_euro
    # Solve the linear equations
    return np.linalg.solve(lhs, rhs)


def domestic_derivative(
    A,
    B,
    domestic_prod,
    index,
    risk_domestic_prod=0,
    factor_euro=0.1,
    adjust=lambda x: x,
    power=2,
    i=0# CH
):
    # Voluntary disruption (non european countries)
    risk_noneuro_vol = risk(index.loc[B.columns])
    risk_noneuro_vol = adjust(risk_noneuro_vol)
    risk_noneuro_vol = risk_noneuro_vol.to_numpy().reshape(1, -1)

    A_num = A.to_numpy().copy()
    B_num = B.to_numpy().copy()
    # Voluntary disruption (european countries)
    risk_euro_vol = risk(index.loc[A.columns])
    risk_euro_vol = adjust(risk_euro_vol)
    risk_euro_vol = risk_euro_vol.to_numpy().reshape(1, -1)
    
    jac = jacfwd(_function_diff)
    risk_CH=jac(
        A_num,
        B_num,
        domestic_prod,
        risk_domestic_prod,
        risk_noneuro_vol,
        factor_euro,
        risk_euro_vol,
        power,
    )[i]
    return np.diag(risk_CH)

def imports_derivative(
    A,
    B,
    domestic_prod,
    index,
    risk_domestic_prod=0,
    factor_euro=0.1,
    adjust=lambda x: x,
    power=2,
):

    # Voluntary disruption (non european countries)
    risk_noneuro_vol = risk(index.loc[B.columns])
    risk_noneuro_vol = adjust(risk_noneuro_vol)
    risk_noneuro_vol = risk_noneuro_vol.to_numpy().reshape(1, -1)

    A_num = A.to_numpy().copy()
    B_num = B.to_numpy().copy()
    # Voluntary disruption (european countries)
    risk_euro_vol = risk(index.loc[A.columns])
    risk_euro_vol = adjust(risk_euro_vol)
    risk_euro_vol = risk_euro_vol.to_numpy().reshape(1, -1)

    jac = jacfwd(_function_diff)
    return jac(
        A_num,
        B_num,
        domestic_prod,
        risk_domestic_prod,
        risk_noneuro_vol,
        factor_euro,
        risk_euro_vol,
        power,
    )


def _function_diff(
    A_num,
    B_num,
    domestic_prod,
    risk_domestic_prod,
    risk_noneuro_vol,
    factor_euro,
    risk_euro_vol,
    power,
):
    # Total risk for voluntary disruption
    risk_non_europ = (B_num**power * risk_noneuro_vol).sum(axis=1)
    # Normalize proportions
    A_num = A_num / (
        A_num.sum(axis=1, keepdims=True) + B_num.sum(axis=1, keepdims=True)
    )
    # Set diagonal to zero
    diag_elements = jnp.diag_indices_from(A_num)
    A_num = A_num.at[diag_elements].set(0)
    risk_euro = factor_euro * (A_num**power * risk_euro_vol).sum(axis=1)
    lhs = np.eye(A_num.shape[0]) - A_num**power
    rhs = domestic_prod**power * risk_domestic_prod + risk_non_europ + risk_euro
    # Solve the linear equations
    return jnp.linalg.solve(lhs, rhs)
