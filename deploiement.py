"""
deploiement.py
==============

API FastAPI + Frontend pour LocaPay.

Sert :
- L'API REST (endpoints /predire, /quartiers, etc.)
- Le frontend HTML/CSS/JS (dossier static/)

Usage :
    python deploiement.py
    # puis ouvrir http://localhost:8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse                          # ← AJOUT 1
from fastapi.staticfiles import StaticFiles                         # ← AJOUT 1
from pydantic import BaseModel, Field, ConfigDict

from config import (
    DOSSIER_MODELE,
    DOSSIER_RACINE,
    FICHIER_CLEAN,
    FICHIER_MODELE,
    CIBLE,
    API_HOST,
    API_PORT,
    INTERVALLE_CONFIANCE,
)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

CHEMIN_MODELE = DOSSIER_MODELE / FICHIER_MODELE

if not CHEMIN_MODELE.exists():
    print(f"❌ Modèle introuvable : {CHEMIN_MODELE}")
    print("   Lance d'abord : python entrainement.py")
    sys.exit(1)

print(f"Chargement du modèle : {CHEMIN_MODELE}")
contenu = joblib.load(CHEMIN_MODELE)

pipeline = contenu["pipeline"]
modeles_bootstrap = contenu.get("modeles_bootstrap", [])
nom_modele = contenu.get("nom_modele", "?")
config = contenu.get("config", "?")
colonnes_attendues = contenu.get("colonnes", [])
transformation_cible = contenu.get("transformation_cible", "aucune")
metriques = contenu.get("metriques", {})

log_cible = transformation_cible == "log1p"

print(f"✓ Modèle chargé : {nom_modele}")
print(f"  Config : {config}")
print(f"  Log-transform : {log_cible}")
print(f"  Bootstrap : {len(modeles_bootstrap)} modèles")
print(f"  MAE : {metriques.get('MAE', 0):,.0f} FCFA")
print(f"  R²  : {metriques.get('R2', 0):.4f}")


# ============================================================
# CHARGEMENT DES MODALITÉS
# ============================================================

df_clean = pd.read_csv(DOSSIER_RACINE / FICHIER_CLEAN)

QUARTIERS_CONNUS = sorted(df_clean["quartier"].dropna().unique().tolist())
COMMUNES_CONNUES = sorted(df_clean["commune"].dropna().unique().tolist())
ARRONDISSEMENTS_CONNUS = sorted(df_clean["arrondissement"].dropna().unique().tolist())
TYPES_BIEN_CONNUS = sorted(df_clean["type_bien"].dropna().unique().tolist())

QUARTIER_DEFAUT = str(df_clean["quartier"].mode()[0])
COMMUNE_DEFAUT = str(df_clean["commune"].mode()[0])
ARRONDISSEMENT_DEFAUT = str(df_clean["arrondissement"].mode()[0])
TYPE_BIEN_DEFAUT = str(df_clean["type_bien"].mode()[0])

print(f"✓ {len(QUARTIERS_CONNUS)} quartiers, {len(COMMUNES_CONNUES)} communes, "
      f"{len(ARRONDISSEMENTS_CONNUS)} arrondissements, "
      f"{len(TYPES_BIEN_CONNUS)} types de bien")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="LocaPay Price API",
    description=(
        "Prédiction du prix mensuel de logements simples "
        "à Cotonou et Abomey-Calavi (Bénin). "
        f"Modèle : {nom_modele} | MAE : {metriques.get('MAE', 0):,.0f} FCFA"
    ),
    version="3.2.0",
)


# ══════════════════════════════════════════════════════════════
# ← AJOUT 2 : Monter le dossier static pour servir le frontend
# ══════════════════════════════════════════════════════════════

STATIC_DIR = DOSSIER_RACINE / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print(f"✓ Frontend servi depuis {STATIC_DIR}")
else:
    print(f"⚠ Dossier static introuvable : {STATIC_DIR}")
    print(f"   Crée-le avec : mkdir -p {STATIC_DIR}")
    print(f"   Puis ajoute index.html, style.css, app.js")


# ============================================================
# SCHÉMAS PYDANTIC
# ============================================================

class Logement(BaseModel):
    """Caractéristiques d'un logement à évaluer."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type_bien": "2 CHAMBRES SALON",
                "nb_chambres": 2,
                "commune": "COTONOU",
                "arrondissement": "13EME ARRONDISSEMENT",
                "quartier": "AGLA MAISON DU PEUPLE",
            }
        }
    )

    type_bien: str = Field(..., description="Type de bien")
    nb_chambres: float = Field(..., ge=0, le=10, description="Nombre de chambres")
    commune: str = Field(..., description="Commune")
    arrondissement: str = Field(..., description="Arrondissement")
    quartier: str = Field(..., description="Quartier")


class Prediction(BaseModel):
    """Résultat d'une prédiction."""
    prix_estime: float
    intervalle_90: List[float]
    devise: str = "FCFA"
    fiabilite: str
    modele: str
    avertissements: List[str] = []


class InfoModele(BaseModel):
    """Informations sur le modèle."""
    nom: str
    config: str
    log_transform: bool
    n_bootstrap: int
    mae: float
    r2: float
    n_features: int


class ValidationResult(BaseModel):
    """Résultat d'une validation."""
    valide: bool
    erreurs: List[str]
    avertissements: List[str]


# ============================================================
# VALIDATION
# ============================================================

def valider_entree(logement: Logement) -> tuple[list[str], list[str]]:
    """Retourne (erreurs, avertissements)."""
    erreurs = []
    avertissements = []

    if logement.type_bien not in TYPES_BIEN_CONNUS:
        erreurs.append(f"type_bien '{logement.type_bien}' inconnu")

    if logement.commune not in COMMUNES_CONNUES:
        erreurs.append(f"commune '{logement.commune}' inconnue")

    if logement.arrondissement not in ARRONDISSEMENTS_CONNUS:
        if logement.arrondissement == "AUTRE":
            avertissements.append("arrondissement 'AUTRE' → moins précis")
        else:
            erreurs.append(f"arrondissement '{logement.arrondissement}' inconnu")

    if logement.quartier not in QUARTIERS_CONNUS:
        if logement.quartier == "AUTRE":
            avertissements.append("quartier 'AUTRE' → moins précis")
        else:
            erreurs.append(f"quartier '{logement.quartier}' inconnu")

    return erreurs, avertissements


# ============================================================
# UTILITAIRES
# ============================================================

def construire_dataframe(logement: Logement, strict: bool = True) -> pd.DataFrame:
    """Construit le DataFrame attendu par le pipeline."""
    data = {
        "type_bien": [str(logement.type_bien).upper().strip()],
        "nb_chambres": [float(logement.nb_chambres)],
        "commune": [str(logement.commune).upper().strip()],
        "arrondissement": [str(logement.arrondissement).upper().strip()],
        "quartier": [str(logement.quartier).upper().strip()],
    }

    X = pd.DataFrame(data)

    if not strict:
        if X.loc[0, "type_bien"] not in TYPES_BIEN_CONNUS:
            X.loc[0, "type_bien"] = TYPE_BIEN_DEFAUT
        if X.loc[0, "commune"] not in COMMUNES_CONNUES:
            X.loc[0, "commune"] = COMMUNE_DEFAUT
        if X.loc[0, "arrondissement"] not in ARRONDISSEMENTS_CONNUS:
            X.loc[0, "arrondissement"] = ARRONDISSEMENT_DEFAUT
        if X.loc[0, "quartier"] not in QUARTIERS_CONNUS:
            X.loc[0, "quartier"] = QUARTIER_DEFAUT

    for col in ["type_bien", "commune", "arrondissement", "quartier"]:
        X[col] = X[col].astype("object")

    if colonnes_attendues:
        X = X[[c for c in colonnes_attendues if c in X.columns]]

    return X


def predire_avec_intervalle(X: pd.DataFrame) -> tuple[float, float, float, str]:
    """Retourne (prix, borne_basse, borne_haute, fiabilite).

    ⚠️ Le pipeline (TransformedTargetRegressor) applique DÉJÀ
    la transformation inverse (expm1). Ne PAS la réappliquer.
    """
    pred = float(pipeline.predict(X)[0])

    if not np.isfinite(pred):
        raise ValueError(f"Prédiction non finie : {pred}")

    if modeles_bootstrap:
        predictions_boot = []
        for m in modeles_bootstrap:
            try:
                p = float(m.predict(X)[0])
                if np.isfinite(p):
                    predictions_boot.append(p)
            except Exception:
                continue

        if predictions_boot:
            alpha = (1 - INTERVALLE_CONFIANCE) / 2
            lower = float(np.percentile(predictions_boot, alpha * 100))
            upper = float(np.percentile(predictions_boot, (1 - alpha) * 100))
            fiabilite = "haute" if len(predictions_boot) >= 30 else "moyenne"
        else:
            lower = float(pred * 0.8)
            upper = float(pred * 1.2)
            fiabilite = "faible"
    else:
        lower = float(pred * 0.8)
        upper = float(pred * 1.2)
        fiabilite = "faible (basée sur ±20%)"

    return pred, lower, upper, fiabilite


# ============================================================
# ENDPOINTS
# ============================================================

# ══════════════════════════════════════════════════════════════
# ← AJOUT 3 : Endpoint "/" sert le frontend HTML
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Frontend"])
def racine():
    """Retourne la page d'accueil (frontend HTML)."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "service": "LocaPay Price API",
        "message": "Frontend introuvable. Créez static/index.html",
        "documentation": "/docs",
    }


@app.get("/sante", tags=["Info"])
def sante():
    """Health check."""
    return {
        "status": "ok",
        "modele_charge": True,
        "n_bootstrap": len(modeles_bootstrap),
    }


@app.get("/modele", response_model=InfoModele, tags=["Info"])
def info_modele():
    """Informations sur le modèle."""
    return InfoModele(
        nom=nom_modele,
        config=config,
        log_transform=log_cible,
        n_bootstrap=len(modeles_bootstrap),
        mae=float(metriques.get("MAE", 0)),
        r2=float(metriques.get("R2", 0)),
        n_features=len(colonnes_attendues),
    )


@app.get("/quartiers", tags=["Référentiel"])
def liste_quartiers():
    """Liste les modalités connues."""
    return {
        "quartiers": QUARTIERS_CONNUS,
        "communes": COMMUNES_CONNUES,
        "arrondissements": ARRONDISSEMENTS_CONNUS,
        "types_bien": TYPES_BIEN_CONNUS,
        "total": {
            "quartiers": len(QUARTIERS_CONNUS),
            "communes": len(COMMUNES_CONNUES),
            "arrondissements": len(ARRONDISSEMENTS_CONNUS),
            "types_bien": len(TYPES_BIEN_CONNUS),
        },
    }


@app.post("/valider", response_model=ValidationResult, tags=["Prédiction"])
def valider(logement: Logement):
    """Valide une entrée sans prédire."""
    erreurs, avertissements = valider_entree(logement)
    return ValidationResult(
        valide=len(erreurs) == 0,
        erreurs=erreurs,
        avertissements=avertissements,
    )


@app.post("/predire", response_model=Prediction, tags=["Prédiction"])
def predire(
    logement: Logement,
    strict: bool = Query(
        True,
        description="Si true : erreur 400 si modalité inconnue. Si false : fallback.",
    ),
):
    """
    Prédit le prix mensuel d'un logement.

    - **strict=true** : refuse les modalités inconnues
    - **strict=false** : remplace par la modalité la plus fréquente
    """
    erreurs, avertissements = valider_entree(logement)

    if strict and erreurs:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Entrée invalide",
                "erreurs": erreurs,
                "conseil": "Utilisez /quartiers ou ?strict=false",
            },
        )

    if not strict and erreurs:
        avertissements.extend([f"Fallback : {e}" for e in erreurs])

    try:
        X = construire_dataframe(logement, strict=strict)
        prix, lower, upper, fiabilite = predire_avec_intervalle(X)

        return Prediction(
            prix_estime=float(round(prix, -2)),
            intervalle_90=[float(round(lower, -2)), float(round(upper, -2))],
            devise="FCFA",
            fiabilite=fiabilite,
            modele=nom_modele,
            avertissements=avertissements,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 78)
    print("DÉMARRAGE DE L'API LOCAPAY")
    print("=" * 78)
    print(f"  Frontend : http://localhost:{API_PORT}")
    print(f"  API doc  : http://localhost:{API_PORT}/docs")
    print(f"  Modèle   : {nom_modele}")
    print("=" * 78 + "\n")

    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False, log_level="info")