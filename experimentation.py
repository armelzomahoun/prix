"""
experimentations.py
===================

Test A/B de différentes configurations de features et transformations.

CORRECTIONS v2 :
- Utilise uniquement le dataset PROPRE (6 colonnes)
- Ne touche pas aux colonnes inutiles (déjà supprimées)
- Aucune fuite possible
- Fix warnings matplotlib

Usage :
    python experimentations.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    DOSSIER_RACINE,
    DOSSIER_EXPER,
    FICHIER_CLEAN,
    CIBLE,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
)


# ============================================================
# CONFIGURATIONS À TESTER
# ============================================================

CONFIGS = [
    # ─── Référence ───
    {
        "nom": "baseline",
        "drop_colonnes": [],
        "log_cible": False,
        "modele": "GB",
    },
    {
        "nom": "avec_log",
        "drop_colonnes": [],
        "log_cible": True,
        "modele": "GB",
    },

    # ─── Sans nb_chambres (redondant avec type_bien) ───
    {
        "nom": "sans_nb_chambres",
        "drop_colonnes": ["nb_chambres"],
        "log_cible": False,
        "modele": "GB",
    },
    {
        "nom": "sans_nb_chambres+log",
        "drop_colonnes": ["nb_chambres"],
        "log_cible": True,
        "modele": "GB",
    },

    # ─── Sans variables géographiques ───
    {
        "nom": "sans_commune",
        "drop_colonnes": ["commune"],
        "log_cible": True,
        "modele": "GB",
    },
    {
        "nom": "sans_arrondissement",
        "drop_colonnes": ["arrondissement"],
        "log_cible": True,
        "modele": "GB",
    },
    {
        "nom": "sans_quartier",
        "drop_colonnes": ["quartier"],
        "log_cible": True,
        "modele": "GB",
    },
    {
        "nom": "sans_geo",
        "drop_colonnes": ["commune", "arrondissement", "quartier"],
        "log_cible": True,
        "modele": "GB",
    },

    # ─── Autres modèles ───
    {
        "nom": "log+ridge",
        "drop_colonnes": [],
        "log_cible": True,
        "modele": "Ridge",
    },
    {
        "nom": "log+rf",
        "drop_colonnes": [],
        "log_cible": True,
        "modele": "RF",
    },
]


# ============================================================
# UTILITAIRES
# ============================================================

def titre(texte: str) -> None:
    print("\n" + "=" * 78)
    print(texte)
    print("=" * 78)


def charger_dataset() -> pd.DataFrame:
    chemin = DOSSIER_RACINE / FICHIER_CLEAN
    if not chemin.exists():
        print(f"❌ Fichier introuvable : {chemin}")
        print("   Lance d'abord : python analyse.py")
        sys.exit(1)

    df = pd.read_csv(chemin)
    print(f"✓ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"  Colonnes : {list(df.columns)}")
    return df


# ============================================================
# CONSTRUCTION DES PIPELINES
# ============================================================

def construire_preprocesseur(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le préprocesseur (num + cat)."""
    # Forcer les dtypes corrects
    for col in X.select_dtypes(include=["string", "str", "category"]).columns:
        X[col] = X[col].astype("object")

    var_num = X.select_dtypes(include=np.number).columns.tolist()
    var_cat = X.select_dtypes(include=["object"]).columns.tolist()

    pipeline_num = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
    ])

    pipeline_cat = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    transformers = []
    if var_num:
        transformers.append(("num", pipeline_num, var_num))
    if var_cat:
        transformers.append(("cat", pipeline_cat, var_cat))

    return ColumnTransformer(transformers)


def obtenir_modele(nom: str):
    """Retourne le modèle selon son nom."""
    if nom == "GB":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        )
    if nom == "RF":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if nom == "Ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    raise ValueError(f"Modèle inconnu : {nom}")


def construire_pipeline(X_train: pd.DataFrame, config: dict) -> Pipeline:
    """Construit le pipeline complet selon la configuration."""
    preproc = construire_preprocesseur(X_train.copy())
    modele_base = obtenir_modele(config["modele"])

    if config["log_cible"]:
        modele = TransformedTargetRegressor(
            regressor=modele_base,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        modele = modele_base

    return Pipeline([
        ("preprocesseur", preproc),
        ("modele", modele),
    ])


# ============================================================
# ÉVALUATION D'UNE CONFIGURATION
# ============================================================

def evaluer_config(df: pd.DataFrame, config: dict) -> dict | None:
    """Évalue une configuration et retourne les métriques."""
    print(f"\n{'─' * 78}")
    print(f"  Configuration : {config['nom']}")
    print(f"{'─' * 78}")

    # ─── Préparation X / y ───
    drop = [CIBLE] + [c for c in config["drop_colonnes"] if c in df.columns]
    df_config = df.drop(columns=drop)

    y = df[CIBLE].reset_index(drop=True)
    X = df_config.reset_index(drop=True)

    # Vérification : au moins une feature
    if X.shape[1] == 0:
        print("  ❌ Aucune feature → config ignorée")
        return None

    # Sécurité dtypes
    for col in X.select_dtypes(include=["string", "str", "category"]).columns:
        X[col] = X[col].astype("object").where(X[col].notna(), np.nan)

    print(f"  Features ({X.shape[1]}) : {list(X.columns)}")
    print(f"  Cible : {'log1p(prix)' if config['log_cible'] else 'prix brut'}")
    print(f"  Modèle : {config['modele']}")

    # ─── Split ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    # ─── Entraînement ───
    t0 = time.time()
    try:
        pipeline = construire_pipeline(X_train, config)
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"  ❌ Erreur d'entraînement : {e}")
        return None
    duree = time.time() - t0

    # ─── Prédictions ───
    y_pred = pipeline.predict(X_test)

    # ─── Métriques ───
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  Résultats :")
    print(f"    MAE  = {mae:,.0f} FCFA")
    print(f"    RMSE = {rmse:,.0f} FCFA")
    print(f"    R²   = {r2:.4f}")
    print(f"    Temps d'entraînement : {duree:.2f} s")

    # ─── Validation croisée ───
    try:
        cv = cross_val_score(
            pipeline, X_train, y_train,
            cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mae_cv = -cv.mean()
        mae_cv_std = cv.std()
        print(f"    MAE CV ({CV_FOLDS} folds) = {mae_cv:,.0f} ± {mae_cv_std:,.0f} FCFA")
    except Exception as e:
        print(f"    ⚠ CV échouée : {e}")
        mae_cv = np.nan
        mae_cv_std = np.nan

    return {
        "config": config["nom"],
        "modele": config["modele"],
        "log_cible": config["log_cible"],
        "n_features": X.shape[1],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE_CV": mae_cv,
        "MAE_CV_std": mae_cv_std,
        "duree_s": duree,
    }


# ============================================================
# VISUALISATION
# ============================================================

def visualiser_resultats(df_res: pd.DataFrame) -> None:
    """Génère des graphiques comparatifs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_res = df_res.sort_values("MAE").reset_index(drop=True)

    # ─── Graphique 1 : MAE (test vs CV) ───
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_res))
    width = 0.35
    ax.bar(x - width/2, df_res["MAE"], width, label="MAE test", color="#2E86AB")
    ax.bar(x + width/2, df_res["MAE_CV"], width, label="MAE CV",
           color="#E63946", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["config"], rotation=20, ha="right")
    ax.set_ylabel("MAE (FCFA)")
    ax.set_title("Comparaison des configurations — MAE test vs CV",
                 fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(DOSSIER_EXPER / "01_comparaison_MAE.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Graphique 2 : R² ───
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(df_res)), df_res["R2"], color="#06A77D")
    ax.set_xticks(range(len(df_res)))
    ax.set_xticklabels(df_res["config"], rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Comparaison des configurations — R²", fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, r2 in zip(bars, df_res["R2"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(DOSSIER_EXPER / "02_comparaison_R2.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  💾 Graphiques sauvegardés dans {DOSSIER_EXPER}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("EXPÉRIMENTATIONS LOCAPAY — Test A/B des configurations")
    print("=" * 78)

    df = charger_dataset()

    titre("1 — ÉVALUATION DES CONFIGURATIONS")
    resultats = []
    for config in CONFIGS:
        res = evaluer_config(df, config)
        if res is not None:
            resultats.append(res)

    if not resultats:
        print("\n❌ Aucune configuration n'a fonctionné")
        sys.exit(1)

    df_res = pd.DataFrame(resultats).sort_values("MAE").reset_index(drop=True)

    titre("2 — CLASSEMENT FINAL")

    print("\n  Classement par MAE test (croissante) :\n")
    print(df_res[["config", "modele", "log_cible", "n_features",
                  "MAE", "MAE_CV", "R2"]].to_string(index=False))

    # Sauvegarde
    chemin = DOSSIER_EXPER / "resultats_experimentations.csv"
    df_res.to_csv(chemin, index=False)
    print(f"\n  💾 {chemin}")

    # Visualisation
    titre("3 — VISUALISATION")
    visualiser_resultats(df_res)

    # ─── Recommandation ───
    titre("4 — RECOMMANDATION")
    meilleur = df_res.iloc[0]
    baseline_rows = df_res[df_res["config"] == "baseline"]

    print(f"\n  🏆 Meilleure configuration : {meilleur['config']}")
    print(f"     MAE  = {meilleur['MAE']:,.0f} FCFA")
    print(f"     R²   = {meilleur['R2']:.4f}")
    print(f"     MAE CV = {meilleur['MAE_CV']:,.0f} FCFA")

    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[0]
        print(f"\n  📊 Comparaison avec baseline :")
        gain_mae = baseline['MAE'] - meilleur['MAE']
        gain_pct = 100 * gain_mae / baseline['MAE']
        print(f"     Gain MAE : {gain_mae:+,.0f} FCFA ({gain_pct:+.1f} %)")
        print(f"     Gain R²  : {meilleur['R2'] - baseline['R2']:+.4f}")

        if gain_pct > 3:
            print(f"\n  ✅ Gain significatif → adopter cette configuration")
        elif gain_pct > 0:
            print(f"\n  ⚠ Gain marginal → à considérer avec prudence")
        else:
            print(f"\n  ❌ Pas de gain → garder la baseline")

    print("\n" + "=" * 78)
    print("✓ EXPÉRIMENTATIONS TERMINÉES")
    print("=" * 78)


if __name__ == "__main__":
    main()