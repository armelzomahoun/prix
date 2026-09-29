"""
config.py
=========

Configuration centrale partagée entre tous les scripts.

Contient :
- Chemins des fichiers
- Constantes de filtrage
- Options d'entraînement
- Paramètres du modèle
"""

from pathlib import Path


# ============================================================
# CHEMINS
# ============================================================

DOSSIER_RACINE   = Path(__file__).parent
DOSSIER_ANALYSE  = DOSSIER_RACINE / "analyse_output"
DOSSIER_MODELE   = DOSSIER_RACINE / "modele_output"
DOSSIER_EXPLIC   = DOSSIER_RACINE / "explicabilite_output"
DOSSIER_EXPER    = DOSSIER_RACINE / "experimentations_output"

# Fichiers source et dérivés
FICHIER_SOURCE      = "locapay_logements.csv"
FICHIER_CLEAN       = "locapay_clean.csv"
FICHIER_GEOCODE     = "locapay_geocode.csv"
FICHIER_MODELE      = "meilleur_modele_locapay.pkl"

# Création automatique des dossiers
for dossier in (DOSSIER_ANALYSE, DOSSIER_MODELE, DOSSIER_EXPLIC, DOSSIER_EXPER):
    dossier.mkdir(exist_ok=True)


# ============================================================
# FILTRES DE NETTOYAGE
# ============================================================

# Mots à exclure dans type_bien
MOTS_EXCLUS = (
    "VILLA", "MAISON", "DUPLEX", "BOUTIQUE", "BUREAU",
    "MAGASIN", "ENTREPOT", "ENTREPÔT", "TERRAIN",
    "IMMEUBLE", "CHAMBRE DE PASSAGE",
)

# Colonnes à supprimer AVANT l'entraînement
COLONNES_A_SUPPRIMER = (
    "id", "nom_maison", "url_annonce", "date_creation",
    "date_maj", "photo_principale", "toutes_photos",
    "videos", "commission_fcfa", "est_externe", "est_active",
    "nb_photos", "nb_videos",
)

# Colonnes catégorielles conservées pour le modèle
COLONNES_CATEGORIELLES = (
    "type_bien", "commune", "arrondissement", "quartier",
)

# Colonne numérique conservée
COLONNES_NUMERIQUES = ("nb_chambres",)

# Cible
CIBLE = "prix_fcfa"

# Seuil de quantification pour les outliers de prix
QUANTILE_OUTLIER = 0.99


# ============================================================
# OPTIONS D'ENTRAÎNEMENT
# ============================================================

# Transformation de la cible : "aucune" ou "log1p"
TRANSFORMATION_CIBLE = "log1p"

# Seuil pour regrouper les quartiers rares dans "AUTRE"
SEUIL_QUARTIER_RARE = 3

# Seuil pour regrouper les arrondissements rares
SEUIL_ARRONDISSEMENT_RARE = 2

# Faut-il supprimer nb_chambres ?
SUPPRIMER_NB_CHAMBRES = False


# ============================================================
# PARAMÈTRES DU MODÈLE
# ============================================================

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# Bootstrap pour intervalles de confiance
N_BOOTSTRAP = 50
INTERVALLE_CONFIANCE = 0.90


# ============================================================
# PARAMÈTRES API
# ============================================================

API_HOST = "0.0.0.0"
API_PORT = 8000