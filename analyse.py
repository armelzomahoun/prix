"""
analyse.py
==========

Analyse exploratoire complète du dataset Locapay.

IMPORTANT : pas de déduplication (on garde un maximum de lignes).

Génère :
- Statistiques descriptives
- Graphiques (20+)
- Tests statistiques
- locapay_clean.csv

Usage :
    python analyse.py
"""

from __future__ import annotations

import sys
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, shapiro, spearmanr
from matplotlib.ticker import FuncFormatter

from config import (
    DOSSIER_RACINE,
    DOSSIER_ANALYSE,
    FICHIER_SOURCE,
    FICHIER_CLEAN,              # ← AJOUTE CETTE LIGNE
    MOTS_EXCLUS,
    QUANTILE_OUTLIER,
    SEUIL_QUARTIER_RARE,
    SEUIL_ARRONDISSEMENT_RARE,
    COLONNES_A_SUPPRIMER,
    CIBLE,
)


# ============================================================
# UTILITAIRES
# ============================================================

def titre(texte: str) -> None:
    print("\n" + "=" * 78)
    print(texte)
    print("=" * 78)


def normaliser_texte(serie: pd.Series) -> pd.Series:
    """Strip + upper + suppression des accents. Retourne des 'object'."""
    def _clean(x):
        if pd.isna(x):
            return np.nan
        x = unicodedata.normalize("NFKD", str(x))
        x = x.encode("ascii", "ignore").decode("ascii")
        return x.strip().upper()
    return serie.map(_clean).astype("object")


def sauvegarder(fig, nom: str) -> None:
    chemin = DOSSIER_ANALYSE / nom
    fig.tight_layout()
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {chemin.name}")


def formatter_fcfa(x, pos=None):
    """Formate les valeurs en FCFA (k pour milliers)."""
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{int(x/1_000)}k"
    return f"{int(x)}"


# ============================================================
# CHARGEMENT
# ============================================================

def charger(fichier: str) -> pd.DataFrame:
    chemin = DOSSIER_RACINE / fichier
    if not chemin.exists():
        print(f"❌ Fichier introuvable : {chemin}")
        sys.exit(1)
    df = pd.read_csv(chemin)
    print(f"✓ Chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ============================================================
# 1. AUDIT
# ============================================================

def audit(df: pd.DataFrame) -> None:
    titre("1 — AUDIT DU DATASET")

    print("\nTypes et valeurs :")
    resume = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_nuls": df.notna().sum(),
        "manquants": df.isna().sum(),
        "pct_manquants": (df.isna().mean() * 100).round(2),
        "uniques": df.nunique(dropna=True),
    })
    print(resume.to_string())
    resume.to_csv(DOSSIER_ANALYSE / "resume_colonnes.csv")

    constantes = resume[resume["uniques"] <= 1].index.tolist()
    if constantes:
        print(f"\n⚠ Colonnes constantes : {constantes}")

    print(f"\nDoublons exacts : {df.duplicated().sum()}")

    # ─── NOUVEAU : Graphique valeurs manquantes ───
    manquants = df.isna().sum().sort_values(ascending=False)
    manquants = manquants[manquants > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(manquants) > 0:
        bars = ax.barh(manquants.index, manquants.values, color="#C53030")
        ax.set_xlabel("Nombre de valeurs manquantes")
        ax.set_title("Valeurs manquantes par colonne")
        ax.invert_yaxis()
        for bar, val in zip(bars, manquants.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Aucune valeur manquante",
                ha="center", va="center", fontsize=14, color="green")
        ax.set_axis_off()
    sauvegarder(fig, "00_valeurs_manquantes.png")


# ============================================================
# 2. EXTRACTION DE FEATURES OPTIONNELLES
# ============================================================

def extraire_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait des features depuis les colonnes textuelles."""
    titre("2 — EXTRACTION DE FEATURES OPTIONNELLES")

    motif_meuble = r"meubl|equip|furnished|climatis|clim"

    if "nom_maison" in df.columns:
        df["est_meuble"] = (
            df["nom_maison"].astype(str).str.lower()
            .str.contains(motif_meuble, regex=True, na=False)
            .astype(int)
        )
        n_meubles = df["est_meuble"].sum()
        print(f"  est_meuble  : {n_meubles} logements meublés "
              f"({n_meubles / len(df) * 100:.1f}%)")
        if n_meubles == 0:
            df = df.drop(columns=["est_meuble"])
            print("                → 0 valeur → colonne SUPPRIMÉE")
    else:
        print("  est_meuble  : colonne 'nom_maison' absente → ignoré")

    motif_surface = r"(\d+)\s*m[²2]"

    if "nom_maison" in df.columns:
        def _extract(x):
            if pd.isna(x):
                return np.nan
            m = re.search(motif_surface, str(x).lower())
            return float(m.group(1)) if m else np.nan

        df["surface_m2"] = df["nom_maison"].apply(_extract)
        n_surfaces = df["surface_m2"].notna().sum()
        print(f"  surface_m2  : {n_surfaces} surfaces extraites "
              f"({n_surfaces / len(df) * 100:.1f}%)")
        if n_surfaces == 0:
            df = df.drop(columns=["surface_m2"])
            print("                → 0 valeur → colonne SUPPRIMÉE")
        else:
            print(f"                médiane = {df['surface_m2'].median():.0f} m²")
    else:
        print("  surface_m2  : colonne 'nom_maison' absente → ignoré")

    return df


# ============================================================
# 3. NETTOYAGE
# ============================================================

def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    titre("3 — NETTOYAGE")

    df = df.copy()

    for col in ("type_bien", "commune", "arrondissement", "quartier"):
        if col in df.columns:
            df[col] = normaliser_texte(df[col])
    print("✓ Textes normalisés (strip, upper, accents)")

    if "type_bien" in df.columns:
        avant = len(df)
        masque = ~df["type_bien"].str.upper().str.contains(
            "|".join(MOTS_EXCLUS), na=False
        )
        df = df[masque]
        print(f"✓ Filtre logements (exclut villa, maison, etc.) : "
              f"{avant} → {len(df)}")

    avant = len(df)
    df = df[
        df[CIBLE].notna()
        & (df[CIBLE] > 0)
        & (df[CIBLE] < 350000)
    ]
    print(f"✓ Prix > 0 et < 350k : {avant} → {len(df)}")

    avant = len(df)
    df = df[
        df["commune"].notna() & (df["commune"].str.len() > 0)
        & df["quartier"].notna() & (df["quartier"].str.len() > 0)
    ]
    print(f"✓ Commune + quartier non vides : {avant} → {len(df)}")

    avant = len(df)
    seuil = df[CIBLE].quantile(QUANTILE_OUTLIER)
    df = df[df[CIBLE] <= seuil]
    print(f"✓ Outliers (>{seuil:,.0f} FCFA) : {avant} → {len(df)}")

    if "quartier" in df.columns and SEUIL_QUARTIER_RARE > 0:
        avant = df["quartier"].nunique()
        comptes = df["quartier"].value_counts()
        rares = comptes[comptes < SEUIL_QUARTIER_RARE].index
        df["quartier"] = df["quartier"].where(
            ~df["quartier"].isin(rares), "AUTRE"
        )
        apres = df["quartier"].nunique()
        n_autres = (df["quartier"] == "AUTRE").sum()
        print(f"✓ Quartiers rares (< {SEUIL_QUARTIER_RARE}) : "
              f"{avant} → {apres} modalités ({n_autres} en AUTRE)")

    if "arrondissement" in df.columns and SEUIL_ARRONDISSEMENT_RARE > 0:
        avant = df["arrondissement"].nunique()
        comptes = df["arrondissement"].value_counts()
        rares = comptes[comptes < SEUIL_ARRONDISSEMENT_RARE].index
        df["arrondissement"] = df["arrondissement"].where(
            ~df["arrondissement"].isin(rares), "AUTRE"
        )
        apres = df["arrondissement"].nunique()
        print(f"✓ Arrondissements rares (< {SEUIL_ARRONDISSEMENT_RARE}) : "
              f"{avant} → {apres} modalités")

    if "nb_chambres" in df.columns and "type_bien" in df.columns:
        masque_entree = df["type_bien"].str.contains("ENTREE COUCHEE", na=False)
        df.loc[masque_entree & df["nb_chambres"].isna(), "nb_chambres"] = 0
        df["nb_chambres"] = df["nb_chambres"].fillna(
            df.groupby("type_bien")["nb_chambres"].transform("median")
        )
        df["nb_chambres"] = df["nb_chambres"].fillna(df["nb_chambres"].median())
        print("✓ nb_chambres imputé (entrée couchée = 0, autres = médiane par type)")

    df = df.reset_index(drop=True)
    print(f"\n→ Dataset après nettoyage : {len(df)} lignes")
    return df


# ============================================================
# 4. SUPPRESSION DES COLONNES INUTILES
# ============================================================

def supprimer_colonnes_inutiles(df: pd.DataFrame) -> pd.DataFrame:
    titre("4 — SUPPRESSION DES COLONNES INUTILES")

    colonnes_presentes = [c for c in COLONNES_A_SUPPRIMER if c in df.columns]

    if colonnes_presentes:
        df = df.drop(columns=colonnes_presentes)
        print(f"✓ Colonnes supprimées ({len(colonnes_presentes)}) :")
        for col in colonnes_presentes:
            print(f"    - {col}")
    else:
        print("  (aucune colonne à supprimer)")

    print(f"\n→ Colonnes restantes : {list(df.columns)}")
    return df


# ============================================================
# 5. ANALYSE DU PRIX
# ============================================================

def analyse_prix(df: pd.DataFrame) -> None:
    titre("5 — ANALYSE DU PRIX")

    prix = df[CIBLE]
    stats = {
        "Nombre":        f"{len(prix):,}",
        "Minimum":       f"{prix.min():,.0f} FCFA",
        "Q1":            f"{prix.quantile(0.25):,.0f} FCFA",
        "Médiane":       f"{prix.median():,.0f} FCFA",
        "Moyenne":       f"{prix.mean():,.0f} FCFA",
        "Q3":            f"{prix.quantile(0.75):,.0f} FCFA",
        "Maximum":       f"{prix.max():,.0f} FCFA",
        "Écart-type":    f"{prix.std():,.0f} FCFA",
        "Ratio moy/méd": f"{prix.mean() / prix.median():.2f}",
        "Skewness":      f"{prix.skew():.2f}",
    }
    for k, v in stats.items():
        print(f"  {k:<15} : {v}")

    # ─── Graphique 1 : Histogramme ───
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(prix, bins=40, edgecolor="white", color="#1E3A5F")
    ax.axvline(prix.mean(), color="#C53030", linestyle="--", linewidth=2,
               label=f"Moyenne = {prix.mean():,.0f}")
    ax.axvline(prix.median(), color="#D4A24C", linestyle="--", linewidth=2,
               label=f"Médiane = {prix.median():,.0f}")
    ax.set_title("Distribution des prix")
    ax.set_xlabel("Prix (FCFA)")
    ax.set_ylabel("Nombre de logements")
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(formatter_fcfa))
    sauvegarder(fig, "01_distribution_prix.png")

    # ─── Graphique 2 : Boxplot ───
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.boxplot(prix, orientation="horizontal")
    ax.set_title("Boxplot des prix")
    ax.set_xlabel("Prix (FCFA)")
    sauvegarder(fig, "02_boxplot_prix.png")

    # ─── Graphique 3 : Log ───
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log1p(prix), bins=40, edgecolor="white", color="#2E7D5B")
    ax.set_title("Distribution de log(1 + prix)")
    ax.set_xlabel("log(1 + prix)")
    sauvegarder(fig, "03_distribution_log_prix.png")

    # ═══════════════════════════════════════════════════════════
    # NOUVEAUX GRAPHIQUES
    # ═══════════════════════════════════════════════════════════

    # ─── Graphique 4 : Violin plot ───
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(prix, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#E8EEF5")
        pc.set_edgecolor("#1E3A5F")
        pc.set_alpha(0.8)
    ax.set_ylabel("Prix (FCFA)")
    ax.set_title("Distribution en violon des prix")
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(FuncFormatter(formatter_fcfa))
    sauvegarder(fig, "04_violin_prix.png")

    # ─── Graphique 5 : QQ-plot ───
    from scipy import stats as scipy_stats
    fig, ax = plt.subplots(figsize=(8, 6))
    scipy_stats.probplot(prix, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("#1E3A5F")
    ax.get_lines()[0].set_markeredgecolor("#1E3A5F")
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[1].set_color("#C53030")
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title("QQ-plot : prix vs distribution normale")
    sauvegarder(fig, "05_qqplot_prix.png")

    # ─── Graphique 6 : CDF (distribution cumulative) ───
    fig, ax = plt.subplots(figsize=(10, 5))
    prix_sorted = np.sort(prix)
    cdf = np.arange(1, len(prix_sorted) + 1) / len(prix_sorted) * 100
    ax.plot(prix_sorted, cdf, color="#1E3A5F", linewidth=2.5)
    ax.axhline(50, color="#D4A24C", linestyle="--", label="Médiane (50%)")
    ax.axvline(prix.median(), color="#D4A24C", linestyle="--")
    ax.set_xlabel("Prix (FCFA)")
    ax.set_ylabel("Pourcentage cumulé (%)")
    ax.set_title("Distribution cumulative des prix")
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(formatter_fcfa))
    ax.grid(alpha=0.3)
    sauvegarder(fig, "06_cdf_prix.png")

    # ─── Graphique 7 : Boxplot avec valeurs (strip) ───
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(prix, orientation="horizontal",
               patch_artist=True,
               boxprops=dict(facecolor="#E8EEF5", edgecolor="#1E3A5F"),
               medianprops=dict(color="#D4A24C", linewidth=2))
    y_jitter = np.random.normal(1, 0.03, size=len(prix))
    ax.scatter(prix, y_jitter, alpha=0.3, s=15, color="#C53030")
    ax.set_xlabel("Prix (FCFA)")
    ax.set_title("Boxplot + points individuels (strip plot)")
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(formatter_fcfa))
    sauvegarder(fig, "07_boxplot_strip_prix.png")


# ============================================================
# 6. ANALYSE PAR GROUPE
# ============================================================

def analyse_par_groupe(df: pd.DataFrame, colonne: str) -> None:
    if colonne not in df.columns:
        return

    print(f"\nPrix par {colonne} :")
    stats = (
        df.groupby(colonne)[CIBLE]
        .agg(nombre="count", moyenne="mean", mediane="median",
             minimum="min", maximum="max")
        .sort_values("mediane", ascending=False)
    )
    print(stats.round(0).to_string())
    stats.to_csv(DOSSIER_ANALYSE / f"prix_par_{colonne}.csv")


def graphique_groupe(df: pd.DataFrame, colonne: str, rotation: int = 0) -> None:
    if colonne not in df.columns or CIBLE not in df.columns:
        return

    # ─── Boxplot ───
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=colonne, y=CIBLE, ax=ax)
    ax.set_title(f"Prix selon {colonne}")
    ax.tick_params(axis="x", rotation=rotation)
    sauvegarder(fig, f"04_boxplot_{colonne}.png")

    # ═══════════════════════════════════════════════════════════
    # NOUVEAUX GRAPHIQUES PAR GROUPE
    # ═══════════════════════════════════════════════════════════

    # ─── Barplot moyenne + médiane ───
    stats = (df.groupby(colonne)[CIBLE]
             .agg(nombre="count", moyenne="mean", mediane="median")
             .sort_values("mediane", ascending=False))

    if len(stats) >= 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(stats))
        width = 0.35
        ax.bar(x - width/2, stats["moyenne"], width,
               label="Moyenne", color="#1E3A5F")
        ax.bar(x + width/2, stats["mediane"], width,
               label="Médiane", color="#D4A24C")
        ax.set_xticks(x)
        ax.set_xticklabels(stats.index, rotation=rotation or 0, ha="right")
        ax.set_ylabel("Prix (FCFA)")
        ax.set_title(f"Prix moyen et médian par {colonne}")
        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(formatter_fcfa))
        ax.grid(axis="y", alpha=0.3)
        sauvegarder(fig, f"05_barplot_{colonne}.png")

        # ─── Camembert des effectifs ───
        if len(stats) <= 10:
            fig, ax = plt.subplots(figsize=(9, 9))
            colors = sns.color_palette("Blues_r", len(stats))
            wedges, texts, autotexts = ax.pie(
                stats["nombre"], labels=stats.index, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
            ax.set_title(f"Répartition par {colonne}")
            sauvegarder(fig, f"06_camembert_{colonne}.png")


# ============================================================
# 7. CORRÉLATIONS
# ============================================================

def analyse_correlations(df: pd.DataFrame) -> None:
    titre("6 — CORRÉLATIONS")

    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return

    corr = num.corr()
    print(corr.round(3).to_string())
    corr.to_csv(DOSSIER_ANALYSE / "matrice_correlation.csv")

    # ─── Heatmap ───
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice de corrélation")
    sauvegarder(fig, "05_correlation.png")

    # ─── NOUVEAU : Scatter matrix ───
    if num.shape[1] >= 2 and num.shape[1] <= 6:
        try:
            fig = sns.pairplot(num, diag_kind="hist",
                               plot_kws={"alpha": 0.5, "color": "#1E3A5F"})
            fig.fig.suptitle("Pairplot des variables numériques",
                             y=1.02, fontsize=14)
            fig.savefig(DOSSIER_ANALYSE / "07_pairplot.png",
                        dpi=120, bbox_inches="tight")
            plt.close("all")
            print(f"  → 07_pairplot.png")
        except Exception as e:
            print(f"  ⚠ Pairplot échoué : {e}")


# ============================================================
# 8. TESTS STATISTIQUES
# ============================================================

def tests_statistiques(df: pd.DataFrame) -> None:
    titre("7 — TESTS STATISTIQUES")

    prix = df[CIBLE].dropna()
    if len(prix) >= 3:
        if len(prix) > 5000:
            prix = prix.sample(5000, random_state=42)
        stat, p = shapiro(prix)
        print(f"\nShapiro-Wilk : W = {stat:.4f}, p = {p:.2e}")
        print("  → Distribution NON normale" if p < 0.05
              else "  → Distribution normale")

    if "nb_chambres" in df.columns:
        temp = df[["nb_chambres", CIBLE]].dropna()
        rho, p = spearmanr(temp["nb_chambres"], temp[CIBLE])
        print(f"\nSpearman chambres/prix : rho = {rho:.4f}, p = {p:.2e}")

    for col in ("type_bien", "commune", "arrondissement", "quartier"):
        if col not in df.columns:
            continue
        groupes = [
            g[CIBLE].dropna()
            for _, g in df.groupby(col)
            if len(g[CIBLE].dropna()) >= 2
        ]
        if len(groupes) >= 2:
            stat, p = kruskal(*groupes)
            print(f"\nKruskal-Wallis {col} : H = {stat:.4f}, p = {p:.2e}")


# ============================================================
# 9. NOUVELLE SECTION : GRAPHIQUES AVANCÉS
# ============================================================

def graphiques_avances(df: pd.DataFrame) -> None:
    """Graphiques supplémentaires : Pareto, treemap, heatmap croisée."""
    titre("8 — GRAPHIQUES AVANCÉS")

    # ═══════════════════════════════════════════════════════════
    # Graphique A : Pareto des quartiers
    # ═══════════════════════════════════════════════════════════
    if "quartier" in df.columns:
        stats = df["quartier"].value_counts()
        cumul = stats.cumsum() / stats.sum() * 100

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(range(len(stats)), stats.values,
                color="#1E3A5F", alpha=0.7)
        ax1.set_xlabel("Quartiers (triés par nombre d'annonces)")
        ax1.set_ylabel("Nombre d'annonces", color="#1E3A5F")
        ax1.tick_params(axis="y", labelcolor="#1E3A5F")

        ax2 = ax1.twinx()
        ax2.plot(range(len(stats)), cumul.values,
                 color="#D4A24C", linewidth=2.5, marker="o", markersize=3)
        ax2.axhline(80, color="#C53030", linestyle="--",
                    label="80% du volume")
        ax2.set_ylabel("Pourcentage cumulé (%)", color="#D4A24C")
        ax2.tick_params(axis="y", labelcolor="#D4A24C")
        ax2.set_ylim(0, 105)
        ax2.legend(loc="lower right")

        ax1.set_title("Loi de Pareto : concentration des annonces par quartier")
        ax1.grid(axis="y", alpha=0.3)
        sauvegarder(fig, "08_pareto_quartiers.png")

    # ═══════════════════════════════════════════════════════════
    # Graphique B : Heatmap type_bien × commune
    # ═══════════════════════════════════════════════════════════
    if "type_bien" in df.columns and "commune" in df.columns:
        pivot = df.pivot_table(
            values=CIBLE, index="type_bien", columns="commune",
            aggfunc="median"
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                        ax=ax, cbar_kws={"label": "Prix médian (FCFA)"})
            ax.set_title("Prix médian : type_bien × commune")
            sauvegarder(fig, "09_heatmap_type_commune.png")

    # ═══════════════════════════════════════════════════════════
    # Graphique C : Count plot par quartier (top 20)
    # ═══════════════════════════════════════════════════════════
    if "quartier" in df.columns:
        top20 = df["quartier"].value_counts().head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 8))
        bars = ax.barh(top20.index, top20.values, color="#1E3A5F")
        ax.set_xlabel("Nombre d'annonces")
        ax.set_title("Top 20 des quartiers par nombre d'annonces")
        for bar, val in zip(bars, top20.values):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        sauvegarder(fig, "10_top20_quartiers_volume.png")

    # ═══════════════════════════════════════════════════════════
    # Graphique D : Stacked barplot type_bien × commune
    # ═══════════════════════════════════════════════════════════
    if "type_bien" in df.columns and "commune" in df.columns:
        cross = pd.crosstab(df["type_bien"], df["commune"])
        if not cross.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            cross.plot(kind="bar", stacked=True, ax=ax,
                       color=["#1E3A5F", "#D4A24C", "#2E7D5B"])
            ax.set_title("Répartition des types de bien par commune")
            ax.set_xlabel("")
            ax.set_ylabel("Nombre d'annonces")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Commune", loc="upper right")
            ax.grid(axis="y", alpha=0.3)
            sauvegarder(fig, "11_stacked_type_commune.png")

    # ═══════════════════════════════════════════════════════════
    # Graphique E : Scatter prix vs nb_chambres (coloré par commune)
    # ═══════════════════════════════════════════════════════════
    if "nb_chambres" in df.columns and "commune" in df.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        for commune, color in zip(df["commune"].unique(),
                                   ["#1E3A5F", "#D4A24C"]):
            subset = df[df["commune"] == commune]
            ax.scatter(subset["nb_chambres"], subset[CIBLE],
                       alpha=0.5, s=50, label=commune,
                       color=color, edgecolor="white", linewidth=0.5)

        # Ligne de tendance
        z = np.polyfit(df["nb_chambres"], df[CIBLE], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["nb_chambres"].min(),
                              df["nb_chambres"].max(), 100)
        ax.plot(x_line, p(x_line), color="#C53030",
                linewidth=2, linestyle="--", label="Tendance globale")

        ax.set_xlabel("Nombre de chambres")
        ax.set_ylabel("Prix (FCFA)")
        ax.set_title("Relation entre nombre de chambres et prix")
        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(formatter_fcfa))
        ax.grid(alpha=0.3)
        sauvegarder(fig, "12_scatter_chambres_prix.png")

    # ═══════════════════════════════════════════════════════════
    # Graphique F : Distribution par arrondissement (top 10)
    # ═══════════════════════════════════════════════════════════
    if "arrondissement" in df.columns:
        top10 = (df.groupby("arrondissement")[CIBLE]
                 .agg(["count", "median"])
                 .sort_values("count", ascending=False)
                 .head(10))

        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.barh(top10.index[::-1], top10["median"][::-1],
                       color="#1E3A5F")
        ax.set_xlabel("Prix médian (FCFA)")
        ax.set_title("Top 10 arrondissements les plus représentés "
                     "(par volume)")
        for bar, val, n in zip(bars, top10["median"][::-1],
                                top10["count"][::-1]):
            ax.text(bar.get_width() + 1000,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:,.0f} F (n={n})", va="center", fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(formatter_fcfa))
        ax.grid(axis="x", alpha=0.3)
        sauvegarder(fig, "13_top10_arrondissements_volume.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("ANALYSE EXPLORATOIRE LOCAPAY")
    print("=" * 78)

    df_brut = charger(FICHIER_SOURCE)
    audit(df_brut)

    df_brut = extraire_features(df_brut)

    df = nettoyer(df_brut)

    df = supprimer_colonnes_inutiles(df)

    chemin_clean = DOSSIER_RACINE / FICHIER_CLEAN
    df.to_csv(chemin_clean, index=False)
    print(f"\n💾 Dataset nettoyé : {chemin_clean} ({len(df)} lignes)")

    analyse_prix(df)

    titre("6 — ANALYSE PAR GROUPE")
    analyse_par_groupe(df, "type_bien")
    analyse_par_groupe(df, "commune")
    analyse_par_groupe(df, "arrondissement")
    analyse_par_groupe(df, "quartier")
    analyse_par_groupe(df, "nb_chambres")

    graphique_groupe(df, "type_bien")
    graphique_groupe(df, "commune", rotation=20)
    graphique_groupe(df, "arrondissement", rotation=45)
    graphique_groupe(df, "nb_chambres")

    analyse_correlations(df)
    tests_statistiques(df)

    # ═══════════════════════════════════════════════════════════
    # NOUVELLE SECTION
    # ═══════════════════════════════════════════════════════════
    graphiques_avances(df)

    titre("RÉSUMÉ FINAL")
    print(f"\n  Dataset propre : {chemin_clean}")
    print(f"  Lignes         : {len(df)}")
    print(f"  Colonnes       : {list(df.columns)}")
    print(f"  Quartiers      : {df['quartier'].nunique()} modalités")
    print(f"  Arrondissements: {df['arrondissement'].nunique()} modalités")
    print(f"  Types de bien  : {df['type_bien'].nunique()} modalités")
    print(f"\n  Prix : médiane = {df[CIBLE].median():,.0f} FCFA | "
          f"moyenne = {df[CIBLE].mean():,.0f} FCFA")

    # Compter les graphiques générés
    n_png = len(list(DOSSIER_ANALYSE.glob("*.png")))
    print(f"\n  📊 {n_png} graphiques générés dans {DOSSIER_ANALYSE}")

    titre("ANALYSE TERMINÉE")
    print(f"\nRésultats dans : {DOSSIER_ANALYSE.resolve()}")
    print(f"Dataset nettoyé : {chemin_clean}")


if __name__ == "__main__":
    main()