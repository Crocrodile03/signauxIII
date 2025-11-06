import os

def next_filename(folder, label, ext="jpg"):
    """
    Retourne un nom de fichier libre dans le dossier 'folder' pour l'objet 'label'.
    Exemple : si 'couch_1.jpg' existe déjà, retourne 'couch_2.jpg'.

    Arguments :
        folder : chemin du dossier où sauvegarder
        label  : nom de l'objet ('couch', 'bed', etc.)
        ext    : extension du fichier (default: 'jpg')

    Retour :
        Chemin complet du fichier libre
    """
    os.makedirs(folder, exist_ok=True)  # Crée le dossier si nécessaire
    i = 1
    while True:
        filename = os.path.join(folder, f"{label}_{i}.{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1
