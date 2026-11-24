# Bank Marketing — MLOps

Projet académique réalisé dans le cadre du module DevOps / MLOps de quatrième année à ESIEE Paris.

**Année universitaire :** 2026-2027  
**Filière :** Data Science & Intelligence Artificielle

---

## 1. Auteurs

Projet réalisé en trinôme par :

- Aaron ADA
- Yassine AROUS
- Célya BAKALA-YAME

### Répartition du travail — TP2

Afin de garantir une répartition équilibrée du travail, nous avons adopté une organisation tournante permettant à chaque membre du trinôme de participer au développement et à la revue du code.

| Étape | Développement | Relecture |
|---|---|---|
| Contrat de données | Aaron ADA | Yassine AROUS |
| Module d'ingestion | Yassine AROUS | Célya BAKALA-YAME |
| Jeu de features réutilisable | Célya BAKALA-YAME | Aaron ADA |

### Répartition des exercices complémentaires

| Exercice | Responsable |
|---|---|
| Q1 — Nouvelle règle de validation | Aaron ADA |
| Q2 — Création et vérification d'un fichier invalide | Célya BAKALA-YAME |
| Q3 — Nouvelle feature et test unitaire | Yassine AROUS |

Les développements ont été réalisés sur des branches dédiées, avec des Pull Requests et des revues de code avant leur intégration dans la branche principale `main`.

---

## 2. Objectif du projet

Ce projet consiste à construire progressivement un pipeline Machine Learning fiable, reproductible et maintenable à partir du jeu de données Bank Marketing.

L'objectif métier est de prédire la souscription d'un client à un dépôt bancaire à terme dans le cadre d'une campagne marketing.

Ce projet met en application plusieurs bonnes pratiques MLOps :

- Gestion collaborative du code avec Git et GitHub.
- Gestion des environnements et des dépendances.
- Validation des données à l'aide d'un contrat.
- Mise en place de tests unitaires.
- Feature Engineering réutilisable.
- Prévention des fuites de données.
- Stockage des données transformées au format Parquet.
- Revue de code et Pull Requests.
- Versionnement et publication de releases.

---

## 3. Jeu de données

Le projet utilise le dataset Bank Marketing issu de UCI Machine Learning Repository.

| Caractéristique | Valeur |
|---|---|
| Nombre d'observations | 45 211 |
| Nombre de colonnes initiales | 17 |
| Format initial | CSV |
| Séparateur | Point-virgule |
| Variable cible | `y` |
| Taux de souscription | Environ 11,7 % |

Les données contiennent des informations relatives aux clients de la banque, à leur situation financière et aux campagnes marketing.

Les fichiers de données sont stockés localement dans le dossier `data/` et ne sont pas versionnés sur GitHub.

**Source officielle :**  
https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## 4. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.11 | Langage de programmation |
| uv | Gestion de l'environnement et des dépendances |
| Pandas | Manipulation des données |
| Pandera | Validation du contrat de données |
| Pytest | Tests unitaires |
| PyArrow | Export au format Parquet |
| Git | Gestion des versions |
| GitHub | Collaboration et revue de code |

---

## 5. Structure principale du projet

```text
bank-marketing-mlops/
├── data/
│   ├── bank-full.csv
│   └── features.parquet
├── docs/
│   └── data_contract.md
├── src/
│   └── mlops_projet/
│       ├── data/
│       │   ├── load.py
│       │   ├── schema.py
│       │   └── check.py
│       └── features/
│           └── build.py
├── tests/
│   ├── test_load.py
│   ├── test_schema.py
│   └── test_features.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Le dossier `data/` contient les données locales nécessaires à l'exécution du projet ainsi que les fichiers générés.

---

## 6. TP1 — Socle du projet

Le premier TP a permis de mettre en place les fondations du projet.

### Travaux réalisés

- Initialisation du dépôt GitHub.
- Création de la structure du projet Python.
- Configuration de l'environnement avec uv.
- Installation des dépendances.
- Création de la fonction `load_bank_data()`.
- Vérification des dimensions du dataset.
- Vérification de la présence de la variable cible.
- Mise en place des premières Pull Requests.

### Résultats

- 2 tests unitaires réussis.
- Dataset de dimensions `(45211, 17)`.

---

## 7. TP2 — Module d'ingestion et jeu de features

### 7.1. Contrat de données

Un contrat de données a été défini afin de garantir la conformité des informations transmises par l'équipe CRM de la banque.

Le contrat est documenté dans :

`docs/data_contract.md`

Son implémentation avec Pandera est disponible dans :

`src/mlops_projet/data/schema.py`

Le schéma vérifie notamment :

- La présence des 17 colonnes attendues.
- L'absence de colonnes supplémentaires.
- Les types des données.
- Les catégories autorisées.
- Les intervalles de valeurs numériques.
- Les valeurs de la variable cible.

L'utilisation de `lazy=True` permet de collecter les violations du contrat en une seule validation.

Les données non conformes sont rejetées afin d'éviter leur utilisation dans le pipeline Machine Learning.

### 7.2. Module d'ingestion

Le module d'ingestion permet de charger et de vérifier les données.

Une commande de vérification a été créée afin de détecter les violations du contrat.

```bash
uv run python -m mlops_projet.data.check data/bank-full.csv
```

Cette étape contribue à prévenir l'utilisation de données invalides lors de l'entraînement du modèle.

### 7.3. Feature Engineering

La fonction `build_features()` permet d'appliquer les mêmes transformations lors de l'entraînement et de la prédiction.

Les quatre features initialement introduites sont :

| Feature | Description |
|---|---|
| `deja_contacte` | Indique si le client a déjà été contacté lors d'une campagne précédente |
| `solde_negatif` | Indique si le solde bancaire est négatif |
| `nb_contacts_total` | Nombre total de contacts |
| `tranche_age` | Catégorie d'âge du client |

### 7.4. Prévention des fuites de données

La colonne `duration` est supprimée des variables explicatives, car sa valeur n'est connue qu'après l'appel téléphonique.

Son utilisation pour décider quels clients contacter constituerait une fuite de données (data leakage).

La colonne `y` est également retirée des variables explicatives et transformée séparément en une cible binaire :

- 1 : souscription.
- 0 : absence de souscription.

### 7.5. Génération du jeu de features

La commande suivante permet de générer le jeu de features au format Parquet :

```bash
uv run python -m mlops_projet.features.build
```

Le format Parquet permet notamment de conserver les types des données et de bénéficier d'un stockage efficace.

### Résultats du TP2 avant les exercices complémentaires

- 10 tests unitaires réussis.
- 45 211 observations.
- 20 colonnes dans le fichier Parquet.
- Publication de la release `v0.2.0`.

---

## 8. Exercices complémentaires — À vous de réfléchir

Les trois exercices complémentaires ont été réalisés sur la branche `feat/my-rules`.

### Question 1 — Ajouter une nouvelle règle de validation

**Responsable : Aaron ADA**

Une nouvelle règle a été ajoutée à la colonne `campaign` dans le fichier `schema.py`.

Le nombre de contacts effectués auprès d'un même client pendant une campagne doit être compris entre 1 et 100.

```python
"campaign": pa.Column(int, pa.Check.in_range(1, 100)),
```

Dans le dataset réel, la valeur maximale observée est de 63. Les données initiales respectent donc cette nouvelle règle.

Une valeur telle que 5000 serait considérée comme une anomalie potentiellement liée à une erreur de saisie ou d'export.

Cette règle permet de détecter les valeurs aberrantes susceptibles de fausser l'entraînement du modèle. Le contrat bloque les données non conformes avant leur utilisation et permet d'identifier un problème à signaler à l'équipe CRM.

Un test unitaire supplémentaire a été ajouté afin de vérifier le rejet d'une valeur invalide.

### Question 2 — Fabriquer un autre fichier invalide

**Responsable : Célya BAKALA-YAME**

Nous avons volontairement modifié l'âge du premier client pour lui attribuer une valeur de -5 ans.

Le contrat impose pourtant un âge compris entre 18 et 100 ans.

#### Hypothèse avant vérification

Nous attendons exactement une violation du contrat, puisque seule la valeur de l'âge du premier client est modifiée.

#### Création du fichier invalide

```bash
sed '2s/^[0-9]*/-5/' data/bank-full.csv > data/bank-age-casse.csv
```

Cette commande modifie uniquement l'âge du premier client et enregistre les données dans un nouveau fichier, sans modifier le dataset original.

#### Vérification du contrat

```bash
uv run python -m mlops_projet.data.check data/bank-age-casse.csv
```

#### Résultats obtenus

| Indicateur | Résultat |
|---|---|
| Nombre de violations attendu | 1 |
| Nombre de violations détecté | 1 |
| Colonne concernée | `age` |
| Valeur invalide | -5 |
| Règle violée | `in_range(18, 100)` |
| Validation | Données rejetées |

Pandera a correctement levé une exception `SchemaErrors` contenant la violation suivante :

```text
Column 'age' failed element-wise validator:
in_range(18, 100) failure cases: -5
```

Notre hypothèse est donc confirmée : le contrat détecte et refuse les données non conformes.

Le fichier invalide reste stocké localement dans le dossier `data/` et n'est pas envoyé sur GitHub.

### Question 3 — Proposer une nouvelle feature

**Responsable : Célya BAKALA-YAME**

Nous avons ajouté une nouvelle feature nommée `a_un_credit`.

Elle indique si le client possède au moins un crédit immobilier ou personnel.

Cette variable est construite à partir des colonnes `housing` et `loan`.

#### Implémentation

```python
features["a_un_credit"] = (
    (df["housing"] == "yes") |
    (df["loan"] == "yes")
).astype(int)
```

#### Valeurs possibles

| Crédit immobilier | Crédit personnel | a_un_credit |
|---|---|---|
| yes | yes | 1 |
| yes | no | 1 |
| no | yes | 1 |
| no | no | 0 |

#### Justification

Cette feature repose sur une règle fixe qui ne nécessite aucun apprentissage à partir des données. Les informations relatives aux crédits du client sont disponibles avant l'appel téléphonique et ne constituent donc pas une fuite de données.

Cette variable peut apporter au modèle une information synthétique concernant les engagements financiers du client.

#### Validation

Un test unitaire supplémentaire a été ajouté dans `test_features.py` afin de vérifier les quatre combinaisons possibles de crédits immobiliers et personnels.

Le test vérifie que les résultats obtenus correspondent exactement aux valeurs attendues :

```python
[1, 1, 1, 0]
```

Après l'ajout de cette feature, le jeu de données final contient désormais 21 colonnes.

---

## 9. Installation et utilisation

### Prérequis

- Git.
- Python 3.11.
- uv.

### Cloner le dépôt

```bash
git clone https://github.com/aaron-ada91/bank-marketing-mlops.git
cd bank-marketing-mlops
```

### Installer les dépendances

```bash
uv sync
```

### Récupérer les données

Télécharger le fichier `bank-full.csv` depuis la source officielle UCI ou les ressources du cours.

Créer le dossier de données :

```bash
mkdir -p data
```

Placer ensuite le fichier `bank-full.csv` dans ce dossier.

### Vérifier le contrat de données

```bash
uv run python -m mlops_projet.data.check data/bank-full.csv
```

### Exécuter les tests unitaires

```bash
uv run pytest
```

### Générer le jeu de features

```bash
uv run python -m mlops_projet.features.build
```

Le fichier généré est disponible dans :

```text
data/features.parquet
```

---

## 10. Résultats finaux

L'ensemble des tests unitaires a été exécuté avec succès sur la branche principale `main` après la fusion des Pull Requests.

### Tests unitaires

| Catégorie | Tests réussis |
|---|---|
| Chargement des données | 2/2 |
| Contrat de données | 5/5 |
| Feature Engineering | 5/5 |
| **Total** | **12/12** |

Résultat final :

```text
==================== 12 passed ====================
```

### Évolution du jeu de données

| Étape | Observations | Colonnes |
|---|---|---|
| Dataset initial | 45 211 | 17 |
| Jeu de features du TP2 | 45 211 | 20 |
| Jeu de features final | 45 211 | 21 |

### Composition du jeu de features final

- 15 variables explicatives d'origine.
- 5 nouvelles features.
- 1 variable cible.

Le pipeline permet ainsi de produire un jeu de données validé et transformé, destiné aux prochaines étapes du projet Machine Learning.

---

## 12. Perspectives d'amélioration

Les prochaines étapes du projet pourront notamment porter sur :

- L'entraînement de modèles Machine Learning.
- L'évaluation des performances.
- La reproductibilité des entraînements.
- L'automatisation des tests.
- L'intégration continue (CI/CD).
- Le déploiement d'un modèle de prédiction.

---

## 13. Conclusion

Les deux premières séances ont permis de construire les fondations d'un projet MLOps structuré et collaboratif.

Nous disposons désormais d'un contrat de données, d'un module de vérification et d'un jeu de features réutilisable lors de l'entraînement et de la prédiction.

La validation des 12 tests unitaires sur la branche principale confirme le bon fonctionnement des fonctionnalités testées.

Ce travail constitue une base pour les prochaines étapes de développement, d'entraînement et de mise en production d'un modèle Machine Learning.

---

**ESIEE Paris — E4 Data Science & Intelligence Artificielle**

**Aaron ADA · Yassine AROUS · Célya BAKALA-YAME**

**2026-2027**
