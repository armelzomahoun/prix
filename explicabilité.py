"""
explicabilité.py
================

Explicabilité du modèle LocaPay (CatBoost + log-transform).

CORRECTIONS v2 :
- Fix : X_full passé à analyser_residus()
- Fix : impact_fcfa calculé correctement (base_value + shap, pas pred_log + shap)

Produit :
- Importances globales par variable
- SHAP summary plot (beeswarm)
- SHAP bar plot
- Explications locales (3 exemples contrastés)
- Analyse des résidus

Usage :
    python explicabilité.py
"""

from __future__ import annotations

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from config import (
    DOSSIER_RACINE,
    DOSSIER_MODELE,
    DOSSIER_EXPLIC,
    FICHIER_CLEAN,
    FICHIER_MODELE,
    CIBLE,
)

# Configuration affichage
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


# ============================================================
# UTILITAIRES
# ============================================================

def titre(texte: str) -> None:
    print("\n" + "=" * 78)
    print(texte)
    print("=" * 78)


def charger_modele():
    chemin = DOSSIER_MODELE / FICHIER_MODELE
    if not chemin.exists():
        print(f"❌ Fichier introuvable : {chemin}")
        print("   Lance d'abord : python entrainement.py")
        sys.exit(1)

    contenu = joblib.load(chemin)
    print(f"✓ Modèle chargé : {chemin}")
    print(f"  Taille : {chemin.stat().st_size / 1024:.1f} Ko")
    print(f"  Clés du dict : {list(contenu.keys())}")

    pipeline = contenu["pipeline"]
    nom_modele = contenu.get("nom_modele", "?")
    config = contenu.get("config", "?")
    metriques = contenu.get("metriques", {})
    n_bootstrap = contenu.get("n_bootstrap", 0)

    print(f"  Nom : {nom_modele}")
    print(f"  Config : {config}")
    print(f"  Bootstrap : {n_bootstrap} modèles")
    if metriques:
        print(f"  Métriques : MAE={metriques.get('MAE', 0):,.0f} | "
              f"R²={metriques.get('R2', 0):.4f}")

    return contenu


def charger_dataset():
    chemin = DOSSIER_RACINE / FICHIER_CLEAN
    df = pd.read_csv(chemin)
    print(f"✓ Dataset chargé : {df.shape}")
    return df


def extraire_preprocesseur_et_modele(pipeline):
    """Récupère le préprocesseur et le modèle réel (unwrap TransformedTarget)."""
    preproc = pipeline.named_steps["preprocesseur"]
    modele_wrapper = pipeline.named_steps["modele"]

    if hasattr(modele_wrapper, "regressor_"):
        modele_reel = modele_wrapper.regressor_
        log_cible = True
    else:
        modele_reel = modele_wrapper
        log_cible = False

    return preproc, modele_reel, log_cible


def ligne_dense(X, idx):
    """Extrait une ligne de X en array dense."""
    ligne = X[idx]
    if hasattr(ligne, "toarray"):
        return ligne.toarray().flatten()
    return np.asarray(ligne).flatten()


# ============================================================
# 1. IMPORTANCES GLOBALES
# ============================================================

def afficher_importances(model, feature_names):
    titre("1 — IMPORTANCES DES VARIABLES")

    if not hasattr(model, "feature_importances_"):
        print("  ⚠ Le modèle n'a pas d'attribut feature_importances_")
        return None

    importances = model.feature_importances_
    print(f"  Nombre de features : {len(importances)}")

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df_imp["pct"] = 100 * df_imp["importance"] / df_imp["importance"].sum()

    print("\n  Top 20 features :\n")
    print(df_imp.head(20).to_string(index=False))

    # Graphique (top 15 seulement pour lisibilité)
    top_n = 15
    df_plot = df_imp.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df_plot))
    bars = ax.barh(df_plot["feature"], df_plot["pct"], color=colors)
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Top {top_n} features — {type(model).__name__}",
                 fontweight='bold')
    for bar, pct in zip(bars, df_plot["pct"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(DOSSIER_EXPLIC / "01_importances.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  💾 {DOSSIER_EXPLIC / '01_importances.png'}")

    return df_imp


# ============================================================
# 2. SHAP
# ============================================================

def calculer_shap(model, X_transformed, feature_names):
    """Calcule les valeurs SHAP avec TreeExplainer."""
    print("\n  Calcul des valeurs SHAP...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        base_value = float(np.ravel(explainer.expected_value)[0])
        print(f"  ✓ SHAP calculé : shape = {shap_values.shape}")
        print(f"  ✓ Valeur de base : {base_value:.4f} (log-espace)")
        print(f"    → soit ~{np.expm1(base_value):,.0f} FCFA en échelle réelle")
        return explainer, shap_values, base_value
    except Exception as e:
        print(f"  ❌ Erreur SHAP : {e}")
        return None, None, None


def graphiques_shap(shap_values, X_transformed, feature_names,
                   base_value, log_cible: bool):
    """Génère les graphiques SHAP globaux."""

    # ─── Summary plot (beeswarm) ───
    print("\n  → Beeswarm plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        max_display=15, show=False,
    )
    titre_plot = "SHAP — Impact des variables"
    if log_cible:
        titre_plot += " (log-espace)"
    plt.title(titre_plot, fontweight='bold')
    plt.tight_layout()
    plt.savefig(DOSSIER_EXPLIC / "02_shap_beeswarm.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    💾 {DOSSIER_EXPLIC / '02_shap_beeswarm.png'}")

    # ─── Bar plot ───
    print("  → Bar plot SHAP...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        plot_type="bar", max_display=15, show=False,
    )
    plt.title("SHAP — Importance moyenne |SHAP|", fontweight='bold')
    plt.tight_layout()
    plt.savefig(DOSSIER_EXPLIC / "03_shap_bar.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    💾 {DOSSIER_EXPLIC / '03_shap_bar.png'}")


# ============================================================
# 3. EXPLICATIONS LOCALES
# ============================================================

def explications_locales(model, X_full, X_transformed, y_full,
                        shap_values, base_value, feature_names,
                        log_cible: bool):
    """Analyse 3 exemples contrastés (cher, abordable, médian)."""

    titre("3 — EXPLICATIONS LOCALES")

    y_pred_all = model.predict(X_transformed)
    idx_cher = int(np.argmax(y_pred_all))
    idx_pauvre = int(np.argmin(y_pred_all))
    idx_median = int(np.argsort(y_pred_all)[len(y_pred_all)//2])

    exemples = [
        ("LOGEMENT CHER", idx_cher),
        ("LOGEMENT ABORDABLE", idx_pauvre),
        ("LOGEMENT MÉDIAN", idx_median),
    ]

    # Valeur de base en FCFA (une seule fois)
    base_fcfa = float(np.expm1(base_value))

    for titre_ex, idx in exemples:
        print(f"\n{'─' * 78}")
        print(f"  {titre_ex} — Index {idx}")
        print(f"{'─' * 78}")

        print(f"  Caractéristiques :")
        for col in X_full.columns:
            print(f"    {col:20s} = {X_full.iloc[idx][col]}")

        pred_log = float(y_pred_all[idx])
        pred_reel = float(np.expm1(pred_log)) if log_cible else pred_log
        reel = float(y_full.iloc[idx])

        print(f"\n  Prix prédit     : {pred_reel:>10,.0f} FCFA")
        print(f"  Prix réel       : {reel:>10,.0f} FCFA")
        print(f"  Erreur          : {pred_reel - reel:>+10,.0f} FCFA")
        print(f"  Valeur de base  : {base_fcfa:>10,.0f} FCFA (moyenne)")

        # Contributions SHAP
        contributions = pd.DataFrame({
            "feature": feature_names,
            "shap": shap_values[idx],
        })
        contributions["abs_shap"] = contributions["shap"].abs()
        top = contributions.nlargest(5, "abs_shap")[["feature", "shap"]]

        print(f"\n  Top 5 contributions :")
        for _, row in top.iterrows():
            signe = "↑" if row["shap"] > 0 else "↓"

            # ✅ CALCUL CORRIGÉ : impact local en FCFA
            # diff = expm1(base + shap) - expm1(base)
            impact_fcfa = float(
                np.expm1(base_value + row["shap"]) - base_fcfa
            )
            print(f"    {signe} {row['feature']:45s}  "
                  f"shap={row['shap']:>+8.4f}  "
                  f"(~{impact_fcfa:>+10,.0f} FCFA)")


# ============================================================
# 4. ANALYSE DES RÉSIDUS
# ============================================================

def analyser_residus(model, X_transformed, X_full, y_full, log_cible: bool):
    """Analyse les résidus (corrigé : X_full passé en paramètre)."""
    titre("4 — ANALYSE DES RÉSIDUS")

    y_pred_log = model.predict(X_transformed)
    y_pred = np.expm1(y_pred_log) if log_cible else y_pred_log
    residus = y_full.values - y_pred

    mae = np.mean(np.abs(residus))
    rmse = np.sqrt(np.mean(residus**2))
    r2 = 1 - np.sum(residus**2) / np.sum((y_full.values - y_full.mean())**2)

    print(f"  MAE  = {mae:,.0f} FCFA")
    print(f"  RMSE = {rmse:,.0f} FCFA")
    print(f"  R²   = {r2:.4f}")

    # Graphique résidus vs prédit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred, residus, alpha=0.5, edgecolor='white')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Prix prédit (FCFA)")
    ax.set_ylabel("Résidu (réel - prédit)")
    ax.set_title("Analyse des résidus", fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DOSSIER_EXPLIC / "04_residus.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 {DOSSIER_EXPLIC / '04_residus.png'}")

    # Résidus par type_bien
    df_res = pd.DataFrame({
        "type_bien": X_full["type_bien"].values,
        "residu": residus,
        "abs_residu": np.abs(residus),
    })

    print("\n  Résidus par type_bien (triés par MAE) :\n")
    stats = (df_res.groupby("type_bien")
             .agg(nb=("residu", "count"),
                  MAE=("abs_residu", "mean"),
                  biais=("residu", "mean"))
             .round(0)
             .sort_values("MAE"))
    print(stats.to_string())

    # Résidus par commune (si dispo)
    if "commune" in X_full.columns:
        df_res["commune"] = X_full["commune"].values
        print("\n  Résidus par commune :\n")
        stats_com = (df_res.groupby("commune")
                     .agg(nb=("residu", "count"),
                          MAE=("abs_residu", "mean"),
                          biais=("residu", "mean"))
                     .round(0)
                     .sort_values("MAE"))
        print(stats_com.to_string())


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("EXPLICABILITÉ DU MODÈLE LOCAPAY")
    print("=" * 78)

    # ─── Chargement ───
    contenu = charger_modele()
    pipeline = contenu["pipeline"]

    preproc, model, log_cible = extraire_preprocesseur_et_modele(pipeline)
    print(f"\n  Préprocesseur : {type(preproc).__name__}")
    print(f"  Modèle réel   : {type(model).__name__}")
    print(f"  Log-transform : {'Oui' if log_cible else 'Non'}")

    try:
        feature_names = preproc.get_feature_names_out()
        print(f"  Features      : {len(feature_names)}")
    except Exception:
        feature_names = np.array(contenu.get("colonnes", []))
        print(f"  Features      : {len(feature_names)} (depuis métadonnées)")

    # ─── Dataset ───
    df = charger_dataset()
    cols_utilisees = contenu.get("colonnes", list(df.columns[:-1]))
    X_full = df[cols_utilisees].reset_index(drop=True)
    y_full = df[CIBLE].reset_index(drop=True)

    # Transformation
    X_transformed = preproc.transform(X_full)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    print(f"  X transformé  : {X_transformed.shape}")

    # ─── 1. Importances ───
    afficher_importances(model, feature_names)

    # ─── 2. SHAP ───
    titre("2 — SHAP GLOBAL")
    explainer, shap_values, base_value = calculer_shap(
        model, X_transformed, feature_names
    )

    if shap_values is not None:
        graphiques_shap(shap_values, X_transformed, feature_names,
                       base_value, log_cible)

        # ─── 3. Explications locales ───
        explications_locales(
            model, X_full, X_transformed, y_full,
            shap_values, base_value, feature_names, log_cible,
        )

    # ─── 4. Résidus ───
    analyser_residus(model, X_transformed, X_full, y_full, log_cible)

    # ─── Résumé ───
    titre("RÉSUMÉ FINAL")
    print(f"""
  Modèle       : {contenu.get('nom_modele', '?')}
  Config       : {contenu.get('config', '?')}
  Features     : {len(feature_names)}
  Log-transform: {'Oui' if log_cible else 'Non'}
  Bootstrap    : {contenu.get('n_bootstrap', 0)} modèles

  📁 Résultats dans : {DOSSIER_EXPLIC}
""")

    print("=" * 78)
    print("✓ ANALYSE TERMINÉE")
    print("=" * 78)


if __name__ == "__main__":
    main()