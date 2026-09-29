"""
entrainement.py
===============

Entraînement du modèle de prédiction de prix Locapay.

Stratégie :
- Teste 2 configurations : "complet" (5 features) et "sans_geo" (2 features)
- Compare 5 modèles : Ridge, Random Forest, Extra Trees, Gradient Boosting, CatBoost
- Bootstrap pour intervalles de confiance (90%)
- Sauvegarde automatique du meilleur modèle

Usage :
    python entrainement.py
"""

from __future__ import annotations

import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# CatBoost optionnel (installation : pip install catboost)
try:
    from catboost import CatBoostRegressor
    CATBOOST_DISPONIBLE = True
except ImportError:
    CATBOOST_DISPONIBLE = False
    print("⚠ CatBoost non installé → pip install catboost")

from config import (
    DOSSIER_MODELE,
    DOSSIER_RACINE,
    FICHIER_CLEAN,
    FICHIER_MODELE,
    CIBLE,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    N_BOOTSTRAP,
)


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
    for col in X.select_dtypes(include=["string", "str", "category"]).columns:
        X[col] = X[col].astype("object")

    var_num = X.select_dtypes(include=np.number).columns.tolist()
    var_cat = X.select_dtypes(include=["object"]).columns.tolist()

    transformers = []
    if var_num:
        transformers.append(("num", Pipeline([
            ("imputation", SimpleImputer(strategy="median")),
        ]), var_num))

    if var_cat:
        transformers.append(("cat", Pipeline([
            ("imputation", SimpleImputer(strategy="most_frequent")),
            ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), var_cat))

    return ColumnTransformer(transformers)


def obtenir_modeles() -> dict:
    """Retourne le dict des modèles à comparer."""
    modeles = {
        "Ridge (baseline)": Ridge(alpha=1.0, random_state=RANDOM_STATE),

        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }

    if CATBOOST_DISPONIBLE:
        modeles["CatBoost"] = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_state=RANDOM_STATE,
            verbose=0,
        )

    return modeles


def construire_pipeline(X_train: pd.DataFrame, modele, log_cible: bool) -> Pipeline:
    """Construit le pipeline complet avec ou sans log-transform."""
    preproc = construire_preprocesseur(X_train.copy())

    if log_cible:
        modele_final = TransformedTargetRegressor(
            regressor=modele,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        modele_final = modele

    return Pipeline([
        ("preprocesseur", preproc),
        ("modele", modele_final),
    ])


# ============================================================
# ÉVALUATION
# ============================================================

def evaluer(y_true, y_pred, nom: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n  {nom}")
    print(f"    MAE  = {mae:,.0f} FCFA")
    print(f"    RMSE = {rmse:,.0f} FCFA")
    print(f"    R²   = {r2:.4f}")
    return {"Modele": nom, "MAE": mae, "RMSE": rmse, "R2": r2}


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGS = [
    {
        "nom": "complet",          # 5 features
        "drop_colonnes": [],
        "log_cible": True,
    },
    {
        "nom": "sans_geo",         # 2 features
        "drop_colonnes": ["commune", "arrondissement", "quartier"],
        "log_cible": True,
    },
]


# ============================================================
# ENTRAÎNEMENT POUR UNE CONFIG
# ============================================================

def entrainer_config(df: pd.DataFrame, config: dict) -> dict:
    """Entraîne tous les modèles pour une configuration donnée."""
    titre(f"CONFIG : {config['nom']}")

    # ─── Préparation X / y ───
    drop = [CIBLE] + [c for c in config["drop_colonnes"] if c in df.columns]
    X = df.drop(columns=drop).reset_index(drop=True)
    y = df[CIBLE].reset_index(drop=True)

    for col in X.select_dtypes(include=["string", "str", "category"]).columns:
        X[col] = X[col].astype("object")

    print(f"  Features ({X.shape[1]}) : {list(X.columns)}")
    print(f"  Cible : {'log1p(prix)' if config['log_cible'] else 'prix brut'}")

    # ─── Split ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"\n  Split : train={len(X_train)} | test={len(X_test)}")

    # ─── Entraînement de tous les modèles ───
    modeles = obtenir_modeles()
    resultats = []
    pipelines = {}

    print(f"\n  Entraînement des {len(modeles)} modèles :")
    for nom, modele in modeles.items():
        t0 = time.time()
        pipeline = construire_pipeline(X_train, modele, config["log_cible"])
        pipeline.fit(X_train, y_train)
        duree = time.time() - t0

        y_pred = pipeline.predict(X_test)
        res = evaluer(y_test, y_pred, nom)
        res["duree_s"] = duree
        res["config"] = config["nom"]

        resultats.append(res)
        pipelines[nom] = pipeline

    df_res = (pd.DataFrame(resultats)
              .sort_values("MAE")
              .reset_index(drop=True))

    return {
        "config": config["nom"],
        "df_res": df_res,
        "pipelines": pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "log_cible": config["log_cible"],
    }


# ============================================================
# VALIDATION CROISÉE
# ============================================================

def validation_croisee(pipeline, X_train, y_train):
    try:
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mae_cv = -scores.mean()
        mae_std = scores.std()
        print(f"  MAE CV ({CV_FOLDS} folds) = {mae_cv:,.0f} ± {mae_std:,.0f} FCFA")
        return mae_cv, mae_std
    except Exception as e:
        print(f"  ⚠ CV échouée : {e}")
        return np.nan, np.nan


# ============================================================
# BOOTSTRAP POUR INTERVALLES DE CONFIANCE
# ============================================================

def entrainer_bootstrap(X_train, y_train, modele_base, log_cible: bool,
                       n_boot: int = N_BOOTSTRAP):
    """Entraîne n_boot modèles pour estimer les intervalles de confiance."""
    print(f"\n  Bootstrap ({n_boot} modèles)...")
    modeles_boot = []
    t0 = time.time()

    for i in range(n_boot):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train.iloc[idx].reset_index(drop=True)
        y_boot = y_train.iloc[idx].reset_index(drop=True)

        pipeline = construire_pipeline(X_boot, modele_base, log_cible)
        pipeline.fit(X_boot, y_boot)
        modeles_boot.append(pipeline)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_boot}...")

    print(f"  ✓ {n_boot} modèles entraînés en {time.time() - t0:.1f}s")
    return modeles_boot


# ============================================================
# SAUVEGARDE
# ============================================================

def sauvegarder_modele(resultat: dict, bootstrap_actif: bool = True):
    """Sauvegarde le meilleur modèle de la config gagnante."""
    df_res = resultat["df_res"]
    config_nom = resultat["config"]

    meilleur = df_res.iloc[0]
    nom_modele = meilleur["Modele"]
    X_train = resultat["X_train"]
    y_train = resultat["y_train"]
    pipeline = resultat["pipelines"][nom_modele]
    log_cible = resultat["log_cible"]

    print(f"\n  🏆 Meilleur modèle [{config_nom}] : {nom_modele}")
    print(f"     MAE  = {meilleur['MAE']:,.0f} FCFA")
    print(f"     RMSE = {meilleur['RMSE']:,.0f} FCFA")
    print(f"     R²   = {meilleur['R2']:.4f}")

    # Réentraînement sur tout le train (déjà fait normalement)
    pipeline.fit(X_train, y_train)

    # Bootstrap pour IC
    modeles_boot = []
    if bootstrap_actif and N_BOOTSTRAP > 0:
        # Récupérer le modèle non-wrapé
        modele_base = pipeline.named_steps["modele"]
        if hasattr(modele_base, "regressor_"):
            modele_base = modele_base.regressor_

        # Créer une copie fraîche pour le bootstrap
        from sklearn.base import clone
        modele_boot_base = clone(modele_base)

        try:
            modeles_boot = entrainer_bootstrap(
                X_train, y_train, modele_boot_base, log_cible,
            )
        except Exception as e:
            print(f"  ⚠ Bootstrap échoué : {e}")
            modeles_boot = []

    # Sauvegarde
    chemin = DOSSIER_MODELE / FICHIER_MODELE
    joblib.dump({
        "pipeline": pipeline,
        "modeles_bootstrap": modeles_boot,
        "nom_modele": nom_modele,
        "config": config_nom,
        "colonnes": X_train.columns.tolist(),
        "cible": CIBLE,
        "transformation_cible": "log1p" if log_cible else "aucune",
        "metriques": meilleur.to_dict(),
        "n_bootstrap": len(modeles_boot),
    }, chemin)

    taille_ko = chemin.stat().st_size / 1024
    print(f"\n  💾 Modèle sauvegardé : {chemin}")
    print(f"     Taille : {taille_ko:.1f} Ko")

    return nom_modele


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("ENTRAÎNEMENT DU MODÈLE LOCAPAY — v2")
    print("=" * 78)

    df = charger_dataset()

    # ─── Entraîner toutes les configurations ───
    resultats_par_config = {}
    for config in CONFIGS:
        resultats_par_config[config["nom"]] = entrainer_config(df, config)

    # ─── Comparer les configs ───
    titre("COMPARAISON DES CONFIGURATIONS")

    comparaison = []
    for nom, res in resultats_par_config.items():
        meilleur = res["df_res"].iloc[0]
        comparaison.append({
            "config": nom,
            "n_features": res["X_train"].shape[1],
            "meilleur_modele": meilleur["Modele"],
            "MAE": meilleur["MAE"],
            "R2": meilleur["R2"],
            "log_cible": res["log_cible"],
        })

    df_comp = pd.DataFrame(comparaison).sort_values("MAE").reset_index(drop=True)
    print("\n" + df_comp.to_string(index=False))

    # ─── Validation croisée du gagnant ───
    titre("VALIDATION CROISÉE DU GAGNANT")
    config_gagnante = df_comp.iloc[0]["config"]
    res_gagnant = resultats_par_config[config_gagnante]
    meilleur_modele_nom = res_gagnant["df_res"].iloc[0]["Modele"]
    pipeline_gagnant = res_gagnant["pipelines"][meilleur_modele_nom]

    validation_croisee(
        pipeline_gagnant,
        res_gagnant["X_train"],
        res_gagnant["y_train"],
    )

    # ─── Sauvegarde du meilleur ───
    titre("SAUVEGARDE DU MEILLEUR MODÈLE")
    nom_final = sauvegarder_modele(res_gagnant, bootstrap_actif=True)

    # ─── Résumé final ───
    titre("RÉSULTAT FINAL")

    meilleur_global = res_gagnant["df_res"].iloc[0]
    print(f"""
  Configuration  : {config_gagnante}
  Modèle         : {nom_final}
  MAE            : {meilleur_global['MAE']:,.0f} FCFA
  RMSE           : {meilleur_global['RMSE']:,.0f} FCFA
  R²             : {meilleur_global['R2']:.4f}
  Features       : {res_gagnant['X_train'].shape[1]}
  Log-transform  : {'Oui' if res_gagnant['log_cible'] else 'Non'}
""")

    # ─── Comparaison détaillée des 2 configs ───
    print("\n  Détail par configuration :\n")
    for nom, res in resultats_par_config.items():
        print(f"  ── {nom.upper()} ──")
        print(res["df_res"][["Modele", "MAE", "RMSE", "R2"]].to_string(index=False))
        print()

    print("=" * 78)
    print("✓ ENTRAÎNEMENT TERMINÉ")
    print("=" * 78)


if __name__ == "__main__":
    main()