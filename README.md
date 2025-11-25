# signauxIII

## 📋 Description du projet

Projet de détection et d'analyse vidéo utilisant YOLO pour la reconnaissance d'objets et de personnes.

## 🔧 Installation

Pour la gestion de fichiers lourds, il faut télécharger git Large File Storage (git lfs) :

```bash
git lfs install
```

Dans le `.gitattributes`, il faut ajouter les types de fichiers à gérer avec `git lfs`.

## 📁 Description des fichiers

### 🎥 Capture vidéo et photos

#### `capture_video_photos.py`

Script principal pour enregistrer une vidéo et prendre des photos périodiques avec la webcam.

- Utilise des arguments en ligne de commande
- Enregistre une vidéo au format AVI
- Prend des photos à intervalles réguliers
- Paramètres configurables : durée, FPS, codec, backend OpenCV

**Utilisation :**

```bash
python capture_video_photos.py --duration 30 --photo-interval 5
```

#### `capture_video_photos_interactive.py`

Version interactive du script de capture avec interface utilisateur via prompts.

- Interface conviviale avec emojis
- Demande les paramètres un par un
- Génération automatique des noms de fichiers
- Confirmation avant démarrage

**Utilisation :**

```bash
python capture_video_photos_interactive.py
```

#### `interface_user.py`

Module contenant les fonctions pour l'interface interactive.

- `prompt_float()` : demande une valeur décimale
- `prompt_int()` : demande un nombre entier
- `prompt_string()` : demande une chaîne de caractères
- `get_next_video_name()` : génère le nom de vidéo suivant (video_X.avi)
- `get_capture_config()` : collecte toute la configuration
- `confirm_config()` : affiche un récapitulatif et demande confirmation

### 🤖 Détection avec YOLO

#### `detection_person.py`

Script de base pour détecter si une personne est présente sur un canapé.

- Utilise YOLOv8n pour la détection
- Détecte les personnes et les canapés
- Vérifie l'intersection des boîtes de détection
- Affiche le résultat dans la console

#### `detection_person_V2.py`

Version améliorée avec choix de source et historique.

- Choix entre webcam ou vidéo enregistrée
- Filtre temporel pour stabiliser la détection
- Enregistrement des événements dans un fichier CSV
- Affichage en temps réel avec code couleur

**Utilisation :**

```bash
python detection_person_V2.py
# Puis choisir : 1 pour webcam, 2 pour vidéo
```

#### `recup_vid.py`

Script de détection en temps réel sur webcam avec filtre temporel.

- Détection personne/canapé en direct
- Filtre de 5 frames pour éviter les faux positifs
- Affichage visuel de l'état (vert/rouge)
- Messages console horodatés

#### `recup_vid_V2.py`

Version avec historique des événements.

- Toutes les fonctionnalités de `recup_vid.py`
- Enregistrement automatique dans `historique_presence.csv`
- Horodatage précis des événements
- Création automatique du fichier CSV avec en-têtes

### 📸 Traitement d'images

#### `recup_photo.py`

Script simple pour afficher une image recadrée.

- Charge une image depuis `MEDIA/IMG/`
- Recadrage manuel de la région d'intérêt
- Affichage avec matplotlib

#### `recup_canap_V3.py`

Extraction automatique des canapés/lits détectés dans une image.

- Détecte les objets "couch" et "bed"
- Découpe et sauvegarde chaque détection
- Organisation en sous-dossiers par type d'objet
- Nommage automatique incrémental

**Utilisation :**

```bash
python recup_canap_V3.py
```

#### `recup_plus_grand.py`

Extraction uniquement du plus grand objet par catégorie.

- Détecte canapés et lits
- Compare les surfaces de détection
- Sauvegarde seulement le plus grand de chaque type
- Utilise `fonction.next_filename()` pour nommage intelligent

### 🛠️ Utilitaires

#### `fonction.py`

Module contenant des fonctions utilitaires.

- `next_filename()` : génère un nom de fichier libre avec numérotation automatique
- Crée automatiquement les dossiers si nécessaires
- Évite l'écrasement de fichiers existants

## 📊 Structure des dossiers

```
signauxIII/
├── Media/
│   ├── VID/          # Vidéos enregistrées
│   └── IMG/          # Photos capturées
├── IMG/              # Images de test
│   ├── couch/        # Canapés extraits
│   └── bed/          # Lits extraits
└── VID/
    └── IMG/          # Images extraites des vidéos
```

## 🎯 Fonctionnalités principales

### Détection d'objets

- YOLOv8n pour la détection
- Classes : personne, canapé, lit
- Détection d'intersection entre objets

### Filtrage temporel

- Seuil de 5 frames consécutives
- Évite les faux positifs
- États stables (présence/absence)

### Enregistrement

- Vidéo + photos simultanées
- Horodatage automatique
- Historique CSV des événements
- Nommage intelligent des fichiers

## 📝 Logs et historiques

### `historique_presence.csv`

Format : `Heure, Événement`

- 🟢 Personne assise
- 🔴 Personne partie

## ⚙️ Configuration

### Paramètres par défaut

- Durée d'enregistrement : 30 secondes
- Intervalle photos : 5 secondes
- FPS vidéo : 30
- Codec : XVID
- Backend : auto (dshow sur Windows)
- Caméra : index 0

## 🚀 Démarrage rapide

1. **Capture interactive :**

   ```bash
   python capture_video_photos_interactive.py
   ```

2. **Détection en temps réel :**

   ```bash
   python recup_vid_V2.py
   ```

3. **Analyse d'une image :**
   ```bash
   python recup_plus_grand.py
   ```

## 📦 Dépendances

- OpenCV (`cv2`)
- Ultralytics YOLO
- PIL (Pillow)
- matplotlib
- datetime, csv, os (modules standard)
