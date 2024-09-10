# Résolution du Challenge "Actuarial Loss Prediction"

https://www.kaggle.com/competitions/actuarial-loss-estimation

## 1. Définition de la problématique business

Le challenge consiste à prédire le coût total des réclamations de compensation des travailleurs (**Ultimate Incurred Claim Cost**). Ces réclamations concernent les compensations que les employés doivent recevoir après des accidents ou des incidents au travail. L'objectif principal est d'aider les compagnies d'assurance et les entreprises à mieux estimer les coûts futurs des réclamations pour ajuster leurs politiques d'assurance.

Le challenge se déroule sur des données synthétiques, sans lien direct avec un pays ou une juridiction spécifique. Les participants doivent également tenir compte de l'inflation des réclamations, un facteur influençant le coût total au fil du temps.

### Problème business :
- **Prédire les coûts futurs des réclamations** permet une meilleure gestion des risques financiers.
- **Optimisation des provisions** : les entreprises peuvent allouer suffisamment de fonds pour couvrir les réclamations futures.
- **Meilleure gestion des primes d'assurance** en fonction des prédictions de coûts, pour éviter les pertes financières imprévues.

---

## 2. Démarche méthodologique pour résoudre le challenge

### a. Exploration des données
- **Compréhension des variables** : Analyse des caractéristiques telles que l'âge, le salaire, le statut matrimonial, les heures travaillées, etc.
- **Visualisation des données** : Exploration des distributions, corrélations, et identification des patterns temporels.
- **Analyse des variables temporelles** : Les dates d'accidents et de réclamations doivent être exploitées pour créer des variables comme le délai entre l'accident et la réclamation.

### b. Préparation des données
- **Nettoyage des données** : Gestion des valeurs manquantes, traitement des outliers et encodage des variables catégorielles.
- **Feature engineering** : 
  - Créer des variables comme le délai entre l'accident et la déclaration.
  - Créer des groupes de sinistres en appliquant la technique du word2vec sur la colonne **ClaimDescription** complètée d'un clustering
- **Encodage des variables** : Utilisation de l'encodage **OneHotEncoder** pour les variables catégorielles (ex. : statut marital).
- **Standardisation** : Les variables numériques sont standardisées pour éviter les écarts d’échelle.

### c. Modélisation

Compte tenu du délai, voici la méthodologie appliquée

- Modèle baseline

  - NLP (Word2Vec) + Clustering (Kmeans) pour classer les types de sinitres
  - **Gradient Boosting Regressor**
  - **XGBoost Regressor**

- Optimisation
  
- **Recherche d'hyperparamètres** : Utilisation de **GridSearchCV** pour optimiser les hyperparamètres des modèles.
- Récupération du modèle avec les meilleures performances

### d. Évaluation des modèles
- **Indicateur de performance** : Le modèle est évalué avec le **Root Mean Squared Error (RMSE)**, qui pénalise plus fortement les grandes erreurs. L'objectif est de minimiser cette métrique sur l'ensemble de validation et de test.


### f. Soumission
- **Génération des prédictions** : Prédictions sur le jeu de test, soumises au format CSV avec les colonnes `ClaimNumber` et `UltimateIncurredClaimCost`.

---

## 3. Conclusion sur les résultats

Une fois les résultats obtenus, voici les étapes à suivre :

- **Analyse des erreurs** : Interprétation des erreurs et identification des scénarios où les prédictions sont incorrectes (ex. : réclamations surévaluées ou sous-évaluées).
- **Optimisations potentielles** :
  - Réévaluation des **features** pour capturer des informations plus pertinentes.
  - Envisager des modèles plus complexes ou utiliser des techniques d'ensemble pour améliorer les performances.

L'objectif final de cette compétition est d'appliquer des techniques d'analyse de données avancées pour mieux prédire les coûts de réclamations, tout en prenant en compte l'inflation et les risques financiers associés aux accidents de travail. Les solutions les plus pertinentes seront celles qui réduisent le RMSE tout en fournissant des insights exploitables pour la gestion des risques dans le secteur des assurances.

---

## 4. Aller plus loin

- **Pousser un peu plus la data prep/analysis**
    - Détecter les outliers et les traiter
    - Approfondir les liens entre les variables (correlations, chi2 etc.)
    - Discrétiser certaines variables
    - affinier le word2vec + clustering

- **Mettre en place l'optimisation des hyperpamatres avec optuna par exemple**
- **Mettre en place un monitoring des soumissions avec mlfow**

---

## 5. Résultats après soumission

Après les différentes tentattives et optimisations, mon meilleur **private score** est de **23399.68398** et le classement aurait été entre le **5eme** et **6eme**.

![Leadboard](/actuarial_leadboard.PNG "Lewis Leadboard")

Dans ce repo github, on peut y trouver les dossiers et les fichiers suivants :

- data contenant les fichiers en input et les soumissions
- models
- src contenant les fichiers :
    - data_preprocessing pour la data prep
    - model_training_evaluation pour l'entrainement et l'evaluation
    - soumission pour sortir le fichier à soumettre
- notebooks pour les fichers notebook
- requirements.txt pour la gestion des packages
- Makefile pour orchestrer les installations d'environnement virtuels, packages etc.
