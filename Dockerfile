# Image Python légère
FROM python:3.11-slim

# Éviter les fichiers .pyc et activer les logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système (utile pour CatBoost/scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python en premier (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Port exposé (Render utilise 10000 par défaut, mais on lit $PORT)
EXPOSE 10000

# Lancer l'API
CMD ["sh", "-c", "uvicorn deploiement:app --host 0.0.0.0 --port ${PORT:-10000}"]
