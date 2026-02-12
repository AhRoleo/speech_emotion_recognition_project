##  Reconnaissance des Émotions Vocales (Speech Emotion Recognition)
## 📌 Description

Ce projet a pour objectif de réaliser une Reconnaissance des Émotions à partir de la voix (SER) en utilisant des techniques de traitement du signal audio et le framework distribué PySpark.

## ⚙️ Pré-requis

- Python 3.11
- Java 17 (required for PySpark)
  (Pas obligatoire, mais Conda est recommandé)

## ☕ Installation de Java 17 (WSL / Ubuntu)

sudo apt update
sudo apt install openjdk-17-jdk

Vérifier l’installation :
java -version

Vous devez obtenir :
openjdk version "17"

## Mise en place de l’environnement

# Création d’un environnement virtuel

Avec conda:
conda create -n ser_env python=3.11 -y
conda activate ser_env

Si vous n’utilisez pas conda :
python3 -m venv ser_env
source ser_env/bin/activate

# Installation des dépendances

pip install -r requirements.txt

# Structure du projet

Le notebook est à la racine du projet. Le dossier qui contient les données, Dataset/, est placé dans le dossier parent du repo. 

## Lancement du projet

Si vous utilisez Jupyter Notebook:

pip install jupyter ipykernel
python -m ipykernel install --user --name ser_env --display-name "Python (ser_env)"
jupyter notebook
