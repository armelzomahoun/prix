/* ============================================================
   LOGIPRIX — FRONTEND LOGIC
   ============================================================ */

const API_URL = window.location.origin;

// Références DOM
const form = document.getElementById("form-prediction");
const selectTypeBien = document.getElementById("type_bien");
const selectCommune = document.getElementById("commune");
const selectArrondissement = document.getElementById("arrondissement");
const selectQuartier = document.getElementById("quartier");
const inputNbChambres = document.getElementById("nb_chambres");
const btnSubmit = document.getElementById("btn-submit");
const btnReset = document.getElementById("btn-reset");

// États
const resultEmpty = document.getElementById("result-empty");
const resultLoading = document.getElementById("result-loading");
const resultError = document.getElementById("result-error");
const resultSuccess = document.getElementById("result-success");

// Résultats
const errorMessage = document.getElementById("error-message");
const priceValue = document.getElementById("price-value");
const intervalLow = document.getElementById("interval-low");
const intervalHigh = document.getElementById("interval-high");
const fiabilite = document.getElementById("fiabilite");
const fiabiliteBadge = document.getElementById("fiabilite-badge");
const modele = document.getElementById("modele");
const avertissementsBox = document.getElementById("avertissements-box");
const avertissementsList = document.getElementById("avertissements-list");

// Encadré de description
const typeHint = document.getElementById("type-hint");

/* ============================================================
   DESCRIPTIONS CONTEXTUELLES DES TYPES DE BIEN
   ============================================================ */

const DESCRIPTIONS_TYPES = {
    "1 CHAMBRE SALON": {
        emoji: "🏠",
        titre: "Appartement standard",
        description: "1 chambre + 1 salon. Non meublé.",
        note: "Logement économique pour un couple ou une personne seule."
    },
    "1 CHAMBRE": {
        emoji: "🏨",
        titre: "Studio meublé haut de gamme",
        description: "Chambre unique meublée et équipée.",
        note: "Souvent climatisé, dans un quartier standing."
    },
    "2 CHAMBRES SALON": {
        emoji: "🏡",
        titre: "Appartement 2 chambres",
        description: "2 chambres + 1 salon. Non meublé.",
        note: "Idéal pour une petite famille ou un couple."
    },
    "3 CHAMBRES SALON": {
        emoji: "🏘️",
        titre: "Appartement familial",
        description: "3 chambres + 1 salon. Non meublé.",
        note: "Pour une famille de 4 à 5 personnes."
    },
    "4 CHAMBRES SALON": {
        emoji: "🏛️",
        titre: "Grand appartement",
        description: "4 chambres + 1 salon. Non meublé.",
        note: "Pour une grande famille."
    },
    "5 CHAMBRES SALON": {
        emoji: "🏰",
        titre: "Très grand logement",
        description: "5 chambres + 1 salon. Non meublé.",
        note: "Rare, souvent dans des quartiers premium."
    },
    "ENTREE COUCHEE": {
        emoji: "🛏️",
        titre: "Entrée couchée",
        description: "Chambre simple avec entrée indépendante.",
        note: "Solution la plus économique du marché."
    }
};

/* ============================================================
   MISE À JOUR DE L'ENCADRÉ AU CHANGEMENT DE TYPE
   ============================================================ */

function mettreAJourTypeHint() {
    const typeSelectionne = selectTypeBien.value;
    
    if (!typeSelectionne || !DESCRIPTIONS_TYPES[typeSelectionne]) {
        typeHint.innerHTML = "";
        typeHint.classList.remove("visible");
        return;
    }

    const info = DESCRIPTIONS_TYPES[typeSelectionne];
    typeHint.innerHTML = `
        <div class="hint-header">
            <span class="hint-emoji">${info.emoji}</span>
            <span class="hint-titre">${info.titre}</span>
        </div>
        <div class="hint-description">${info.description}</div>
        <div class="hint-note">💡 ${info.note}</div>
    `;
    typeHint.classList.add("visible");
}

/* ============================================================
   CHARGEMENT DES MODALITÉS
   ============================================================ */

async function chargerModalites() {
    try {
        const response = await fetch(`${API_URL}/quartiers`);
        if (!response.ok) throw new Error("Erreur de chargement");

        const data = await response.json();

        remplirSelect(selectTypeBien, data.types_bien);
        remplirSelect(selectCommune, data.communes);
        remplirSelect(selectArrondissement, data.arrondissements);
        remplirSelect(selectQuartier, data.quartiers);

        console.log(`✓ Modalités chargées : ${data.types_bien.length} types, ` +
                    `${data.communes.length} communes, ` +
                    `${data.arrondissements.length} arrondissements, ` +
                    `${data.quartiers.length} quartiers`);
    } catch (error) {
        console.error("Erreur chargement :", error);
        afficherErreur(
            "Impossible de charger les données. Vérifiez que l'API est en ligne."
        );
    }
}

function remplirSelect(select, valeurs) {
    while (select.options.length > 1) select.remove(1);
    valeurs.forEach(valeur => {
        const option = document.createElement("option");
        option.value = valeur;
        option.textContent = valeur;
        select.appendChild(option);
    });
}

/* ============================================================
   INFO MODÈLE
   ============================================================ */

async function chargerInfoModele() {
    try {
        const response = await fetch(`${API_URL}/modele`);
        if (!response.ok) return;

        const data = await response.json();
        document.getElementById("info-mae").textContent =
            `${data.mae.toLocaleString("fr-FR")} F`;
        document.getElementById("info-r2").textContent = data.r2.toFixed(3);
    } catch (error) {
        console.error("Erreur info modèle :", error);
    }
}

/* ============================================================
   PRÉDICTION
   ============================================================ */

async function predire(event) {
    event.preventDefault();
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const body = {
        type_bien: selectTypeBien.value,
        nb_chambres: parseInt(inputNbChambres.value),
        commune: selectCommune.value,
        arrondissement: selectArrondissement.value,
        quartier: selectQuartier.value,
    };

    afficherLoading();

    try {
        const response = await fetch(`${API_URL}/predire?strict=true`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        if (!response.ok) {
            const detail = data.detail;
            if (typeof detail === "object" && detail.erreurs) {
                afficherErreur(detail.erreurs.join("\n"));
            } else {
                afficherErreur(typeof detail === "string"
                    ? detail : "Erreur de prédiction");
            }
            return;
        }

        afficherResultat(data);
    } catch (error) {
        console.error("Erreur prédiction :", error);
        afficherErreur("Impossible de contacter l'API. Vérifiez votre connexion.");
    }
}

/* ============================================================
   ÉTATS D'AFFICHAGE
   ============================================================ */

function cacherTous() {
    resultEmpty.classList.add("hidden");
    resultLoading.classList.add("hidden");
    resultError.classList.add("hidden");
    resultSuccess.classList.add("hidden");
}

function afficherLoading() {
    cacherTous();
    resultLoading.classList.remove("hidden");
    btnSubmit.disabled = true;
}

function afficherErreur(message) {
    cacherTous();
    errorMessage.textContent = message;
    resultError.classList.remove("hidden");
    btnSubmit.disabled = false;
}

function afficherResultat(data) {
    cacherTous();

    priceValue.textContent = data.prix_estime.toLocaleString("fr-FR");
    intervalLow.textContent = data.intervalle_90[0].toLocaleString("fr-FR");
    intervalHigh.textContent = data.intervalle_90[1].toLocaleString("fr-FR");

    fiabilite.textContent = data.fiabilite;
    modele.textContent = data.modele;

    // Badge fiabilité
    fiabiliteBadge.textContent = data.fiabilite;
    fiabiliteBadge.className = "confidence-badge " + data.fiabilite.toLowerCase();

    // Avertissements
    if (data.avertissements && data.avertissements.length > 0) {
        avertissementsList.innerHTML = "";
        data.avertissements.forEach(a => {
            const li = document.createElement("li");
            li.textContent = a;
            avertissementsList.appendChild(li);
        });
        avertissementsBox.classList.remove("hidden");
    } else {
        avertissementsBox.classList.add("hidden");
    }

    resultSuccess.classList.remove("hidden");
    btnSubmit.disabled = false;
}

/* ============================================================
   RESET
   ============================================================ */

function reinitialiser() {
    form.reset();
    inputNbChambres.value = 2;
    cacherTous();
    resultEmpty.classList.remove("hidden");
    btnSubmit.disabled = false;
    // Cacher l'encadré de description
    typeHint.innerHTML = "";
    typeHint.classList.remove("visible");
}

/* ============================================================
   INITIALISATION
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    chargerModalites();
    chargerInfoModele();

    // Écouter le changement de type
    selectTypeBien.addEventListener("change", mettreAJourTypeHint);

    form.addEventListener("submit", predire);
    btnReset.addEventListener("click", reinitialiser);
});
